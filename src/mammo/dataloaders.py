# src/dataloaders.py
"""
CBIS-DDSM loader with on-the-fly Kaggle download and dataframe creation.

cfg keys required:
    dataset        : 'cbis_ddsm'  (only option for now)
    view           : 'ROI' or 'FF'
    data_root      : where to unzip images  (e.g. /content/cbis_ddsm/jpeg)
    input_size     : 224 or 512
    batch_size
    csv_path       : optional cache; if absent we regenerate each run

The code mirrors the Colab notebook logic exactly.
"""

import os, subprocess, zipfile, random, pathlib, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception


KAGGLE_SLUG = "awsaf49/cbis-ddsm-breast-cancer-image-dataset"
ZIP_NAME    = "cbis-ddsm-breast-cancer-image-dataset.zip"

IMG_SIZE = {224: (224, 224), 512: (512, 512)}

# Updated to match baseline training augmentation exactly
_AUG = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),           # horizontal_flip=True
    tf.keras.layers.RandomRotation(0.056),              # rotation_range=20 degrees (20/360 = 0.056)
    tf.keras.layers.RandomTranslation(0.1, 0.1),        # width_shift_range=0.1, height_shift_range=0.1
    tf.keras.layers.RandomShear(0.2),                   # shear_range=0.2
    tf.keras.layers.RandomZoom(0.2),                    # zoom_range=0.2
], name="baseline_training_augment")

# --------------------------------------------------------------------- #
# 1. Download + unzip                                                   #
# --------------------------------------------------------------------- #
def _download_and_unzip(dest: pathlib.Path):
    if (dest/"jpeg").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("[dataloaders] Downloading CBIS-DDSM from Kaggle …")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(dest)],
        check=True,
    )
    print("[dataloaders] Unzipping …")
    zipfile.ZipFile(dest/ZIP_NAME).extractall(dest)
    (dest/ZIP_NAME).unlink()                    # remove zip to save space

# --------------------------------------------------------------------- #
# 2. Build dataframe (ROI or full-frame)                                #
# --------------------------------------------------------------------- #
def _make_dataframe(root: pathlib.Path, view: str):
    img_dir   = root/"jpeg"
    csv_dir   = root/"csv"
    dicom_csv = csv_dir/"dicom_info.csv"
    mass_csv  = csv_dir/"mass_case_description_train_set.csv"
    calc_csv  = csv_dir/"calc_case_description_train_set.csv"

    dicom_df = pd.read_csv(dicom_csv)

    if view.lower() == "roi":
        roi_df = dicom_df[dicom_df['SeriesDescription'] == 'cropped images'].copy()
    else:  # full frame
        roi_df = dicom_df[dicom_df['SeriesDescription'] != 'cropped images'].copy()

    roi_df['uid'] = roi_df['image_path'].apply(lambda x: x.split('/')[-2])

    mass_df = pd.read_csv(mass_csv)[['cropped image file path', 'pathology']].copy()
    calc_df = pd.read_csv(calc_csv)[['cropped image file path', 'pathology']].copy()

    mass_df['uid'] = mass_df['cropped image file path'].apply(lambda x: x.split('/')[-2])
    calc_df['uid'] = calc_df['cropped image file path'].apply(lambda x: x.split('/')[-2])
    mass_df['label'] = (mass_df['pathology'] == 'MALIGNANT').astype(int)
    calc_df['label'] = (calc_df['pathology'] == 'MALIGNANT').astype(int)

    case_df = pd.concat([mass_df[['uid', 'label']], calc_df[['uid', 'label']]], ignore_index=True)
    merged  = pd.merge(case_df, roi_df, on='uid')
    merged['image_path'] = merged['image_path'].str.replace(
        'CBIS-DDSM/jpeg', str(img_dir), regex=False)
    merged = merged[['image_path', 'label']].rename(columns={'image_path': 'filepath'})
    merged['label_str'] = merged['label'].map({0: 'benign', 1: 'malignant'})

    # Updated to match baseline split: 80/10/10 instead of 70/15/15
    train_df, temp_df = train_test_split(
        merged, test_size=0.20, stratify=merged['label_str'], random_state=42)  # 20% for val+test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label_str'], random_state=42)  # 10% each

    train_df['split'] = 'train'
    val_df['split']   = 'val'
    test_df['split']  = 'test'

    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    return df.drop(columns=['label_str'])

# --------------------------------------------------------------------- #
# 3. tf.data builder                                                    #
# --------------------------------------------------------------------- #
def _build_tfds(df, img_size, batch, is_train, preprocess_fn, model_type):
    """Build TensorFlow dataset with correct preprocessing and shuffling."""
    # Apply a DataFrame-level shuffle first for VGG16 models
    # This ensures proper class distribution before creating the dataset
    df_shuffled = df.sample(frac=1.0).reset_index(drop=True)
    
    # Debug: print class distribution
    malignant_count = df_shuffled['label'].sum()
    total_count = len(df_shuffled)
    print(f"Dataset class distribution: {malignant_count}/{total_count} malignant "
          f"({100*malignant_count/total_count:.1f}%)")
    
    paths = df_shuffled['filepath'].values
    labels = df_shuffled['label'].values
    
    # Ensure labels are float32 for both VGG16 and InceptionV3
    labels = labels.astype('float32')
    
    # Debug: print first few labels to verify shuffle
    print(f"First 20 labels after shuffle: {labels[:20]}")
    
    ds_paths = tf.data.Dataset.from_tensor_slices(paths)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)

    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE[img_size])
        img = preprocess_fn(img)
        
        # Reshape label explicitly to match model expectations
        # This is critical for compatibility across models
        return img, tf.reshape(y, [1])

    ds = tf.data.Dataset.zip((ds_paths, ds_labels))
    
    # Print the dataset structure for debugging
    print(f"Dataset structure: {ds.element_spec}")
    
    if is_train:
        # Use a large buffer size for better shuffling
        buffer_size = min(len(paths), 4096)
        ds = ds.shuffle(buffer_size, seed=42, reshuffle_each_iteration=True)
        
        # Then do expensive mapping
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Then augmentation
        ds = ds.map(lambda x, y: (_AUG(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # For validation: just mapping, no shuffle
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Print the dataset structure after mapping for debugging
    print(f"Dataset structure after mapping: {ds.element_spec}")
    
    # Create batches and prefetch
    batched_ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    
    # Print the batched dataset structure for debugging
    print(f"Batched dataset structure: {batched_ds.element_spec}")
    
    # Debug: check first few batches for class distribution if training
    if is_train:
        print("\nChecking class distribution in first 5 batches:")
        batch_data = []
        for i, (imgs, batch_y) in enumerate(batched_ds.take(5)):
            # Print the raw tensor for debugging
            print(f"Batch {i+1} label tensor: {batch_y}")
            print(f"Batch {i+1} label shape: {batch_y.shape}")
            
            # Calculate malignant rate
            malignant_rate = tf.reduce_mean(tf.cast(batch_y, tf.float32))
            print(f"Batch {i+1}: {malignant_rate:.3f} malignant rate")
            
            # Extract the actual label values for detailed inspection
            y_values = batch_y.numpy().flatten()
            batch_data.append((i+1, y_values, malignant_rate.numpy()))
            
            # Count unique values
            unique_vals, counts = np.unique(y_values, return_counts=True)
            print(f"  Unique values: {unique_vals}, counts: {counts}")
        
        # Print detailed batch information
        print("\nDetailed batch analysis:")
        for batch_num, y_values, rate in batch_data:
            print(f"Batch {batch_num}: {len(y_values)} samples, {rate:.3f} malignant rate")
            if len(y_values) > 0:
                print(f"  First 10 labels: {y_values[:10]}")
    
    return batched_ds


# --------------------------------------------------------------------- #
# 4. Public API                                                         #
# --------------------------------------------------------------------- #
def get_loaders(cfg):
    """Return train_ds, val_ds according to cfg."""
    if cfg['dataset'].lower() != 'cbis_ddsm':
        raise ValueError("Only cbis_ddsm supported for now.")

    root = pathlib.Path(cfg['data_root'])
    _download_and_unzip(root)

    # dataframe cache
    csv_path = cfg.get('csv_path')
    if csv_path and pathlib.Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        df = _make_dataframe(root, cfg['view'])
        if csv_path:
            pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)

    img_size  = cfg.get('input_size', 224)
    batch     = cfg.get('batch_size', 16)
    model_type = cfg["model"].lower()

    # choose preprocessing function based on model
    if model_type == "vgg16":
        print("Using VGG16 preprocessing")
        preprocess_fn = preprocess_vgg
    elif model_type == "inceptionv3":
        print("Using InceptionV3 preprocessing")
        preprocess_fn = preprocess_inception
    else:
        raise ValueError("Unknown model architecture")
    
    # Add debug mode to analyze labels directly before building dataset
    train_subset = df[df.split == 'train']
    print("\nDirect inspection of training labels:")
    print(f"Label type: {train_subset['label'].dtype}")
    print(f"Unique label values: {train_subset['label'].unique()}")
    print(f"Label counts: {train_subset['label'].value_counts()}")
    
    # Convert labels to float32 before dataset creation
    df['label'] = df['label'].astype('float32')

    train_ds = _build_tfds(df[df.split == 'train'], img_size, batch, is_train=True, 
                          preprocess_fn=preprocess_fn, model_type=model_type)
    val_ds   = _build_tfds(df[df.split == 'val'],   img_size, batch, is_train=False, 
                          preprocess_fn=preprocess_fn, model_type=model_type)
    
    # Verify dataset structure
    print("\nDataset verification:")
    for images, labels in train_ds.take(1):
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(f"Label data type: {labels.dtype}")
        print(f"First few labels: {labels.numpy().flatten()[:10]}")
    
    return train_ds, val_ds

# Export the augmentation layer
AUG_LAYER = _AUG