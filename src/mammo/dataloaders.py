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
import numpy as np
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
def _build_tfds(df, img_size, batch, is_train, preprocess_fn, model_type=""):
    """Build TensorFlow dataset with correct preprocessing and shuffling."""
    # Apply a DataFrame-level shuffle first
    # This ensures proper class distribution before creating the dataset
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Debug: print class distribution
    malignant_count = df_shuffled['label'].sum()
    total_count = len(df_shuffled)
    print(f"Dataset class distribution: {malignant_count}/{total_count} malignant "
          f"({100*malignant_count/total_count:.1f}%)")
    
    # CRITICAL: Ensure labels are float32 before creating the dataset
    df_shuffled['label'] = df_shuffled['label'].astype('float32')
    
    paths = df_shuffled['filepath'].values
    labels = df_shuffled['label'].values
    
    # Debug: print first few labels to verify shuffle
    print(f"First 20 labels after shuffle: {labels[:20]}")
    
    # Create tensor slices from PREPROCESSED numpy arrays to avoid issues
    # This is crucial to prevent label corruption in the pipeline
    ds_paths = tf.data.Dataset.from_tensor_slices(paths)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)

    def _load(path, y):
        # Load and preprocess image
        img_data = tf.io.read_file(path)
        img_data = tf.image.decode_png(img_data, channels=3)
        img_data = tf.image.resize(img_data, IMG_SIZE[img_size])
        img_data = preprocess_fn(img_data)
        
        # Convert to float32
        y_data = tf.cast(y, tf.float32)
        
        # For VGG16, provide labels as flat scalars
        if 'vgg16' in model_type.lower():
            return img_data, tf.squeeze(y_data)  # Remove extra dimensions
        else:
            return img_data, tf.reshape(y_data, [1])

    # Create dataset zip
    ds = tf.data.Dataset.zip((ds_paths, ds_labels))
    
    if is_train:
        # Use a large buffer size for better shuffling
        buffer_size = len(paths)  # Use full dataset size for perfect shuffling
        ds = ds.shuffle(buffer_size, seed=42, reshuffle_each_iteration=True)
        
        # Then do expensive mapping
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Then augmentation
        ds = ds.map(lambda x, y: (_AUG(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # For validation: just mapping, no shuffle
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Create batches without further shuffling
    batched_ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    
    # Debug: check batch properties and content
    if is_train:
        print("\nChecking class distribution in first 5 batches:")
        
        batch_data = []
        for i, (_, batch_y) in enumerate(batched_ds.take(5)):
            # Calculate malignant rate
            batch_labels = batch_y.numpy().flatten()
            rate = np.mean(batch_labels)
            print(f"Batch {i+1}: {rate:.3f} malignant rate")
            
            # Show actual label values
            if i < 2:  # For first two batches only
                print(f"  Batch {i+1} labels: {batch_labels}")
                # Count classes
                unique, counts = np.unique(batch_labels, return_counts=True)
                print(f"  Classes: {dict(zip(unique, counts))}")
    
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

    # Choose preprocessing function based on model

    if model_type == "vgg16":
        print("Using VGG16 preprocessing")
        # Define a custom preprocessing function that does simple rescaling
        preprocess_fn = preprocess_vgg
    elif model_type == "inceptionv3":
        print("Using InceptionV3 preprocessing")
        preprocess_fn = preprocess_inception
    else:
        raise ValueError("Unknown model architecture")
    
    # IMPORTANT: Convert all labels to float32 in the dataframe itself
    # This ensures consistent label types before creating the dataset
    df['label'] = df['label'].astype('float32')
    
    # Print info about dataset split sizes
    print(f"Train set: {len(df[df.split == 'train'])} samples")
    print(f"Val set: {len(df[df.split == 'val'])} samples")
    print(f"Test set: {len(df[df.split == 'test'])} samples")

    train_ds = _build_tfds(df[df.split == 'train'], img_size, batch, is_train=True, 
                          preprocess_fn=preprocess_fn, model_type=model_type)
                          
    val_ds = _build_tfds(df[df.split == 'val'], img_size, batch, is_train=False, 
                        preprocess_fn=preprocess_fn, model_type=model_type)
    
    return train_ds, val_ds

# Add this function to the end of dataloaders.py (before the AUG_LAYER export)

def get_dataframe_subset(cfg, split='val'):
    """
    Load dataframe subset for TTA evaluation.
    
    Args:
        cfg: Configuration dictionary containing data paths and settings
        split: Dataset split to return ('train', 'val', 'test')
    
    Returns:
        DataFrame subset for the specified split
    """
    import pandas as pd
    import pathlib
    
    csv_path = cfg.get('csv_path')
    if csv_path and pathlib.Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        # Recreate dataframe if CSV cache doesn't exist
        root = pathlib.Path(cfg['data_root'])
        df = _make_dataframe(root, cfg['view'])
        
        # Cache the dataframe if csv_path is specified
        if csv_path:
            pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
    
    return df[df['split'] == split].reset_index(drop=True)

# Export the augmentation layer (existing line)
AUG_LAYER = _AUG