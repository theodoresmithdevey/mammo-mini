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
# Updated to match baseline training augmentation exactly with proper fill_mode
_AUG = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),                                    # horizontal_flip=True
    tf.keras.layers.RandomRotation(20/360, fill_mode='nearest'),                 # rotation_range=20 degrees exactly, fill_mode='nearest'
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='nearest'),           # width_shift_range=0.1, height_shift_range=0.1, fill_mode='nearest'  
    tf.keras.layers.RandomShear(0.2, fill_mode='nearest'),                       # shear_range=0.2, fill_mode='nearest'
    tf.keras.layers.RandomZoom(0.2, fill_mode='nearest'),                        # zoom_range=0.2, fill_mode='nearest'
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
# Replace the _build_tfds function in dataloaders.py with this corrected version:

def _build_tfds(df, img_size, batch, is_train, preprocess_fn, model_type=""):
    """Build TensorFlow dataset with EXACT baseline preprocessing approach."""
    
    # ✅ FOR VALIDATION: Use ImageDataGenerator approach like baseline
    if not is_train:
        print(f"Using baseline-style validation preprocessing (ImageDataGenerator)")
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Create exact baseline validation generator
        val_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)
        
        # Prepare dataframe with baseline-compatible format
        df_val = df.copy()
        df_val['label_str'] = df_val['label'].map({0: 'benign', 1: 'malignant'})
        
        # Create validation generator exactly like baseline
        val_generator = val_gen.flow_from_dataframe(
            df_val,
            x_col='filepath',
            y_col='label_str', 
            target_size=(img_size, img_size),
            batch_size=batch,
            class_mode='binary',
            shuffle=False  # No shuffling for validation
        )
        
        # Convert to tf.data.Dataset for compatibility
        def generator_to_dataset():
            for batch_x, batch_y in val_generator:
                # Ensure correct label format for model type
                if 'vgg16' in model_type.lower():
                    batch_y = batch_y.flatten()  # VGG16 expects flat labels
                else:
                    batch_y = batch_y.reshape(-1, 1)  # InceptionV3 expects reshaped
                yield batch_x, batch_y
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            generator_to_dataset,
            output_signature=(
                tf.TensorSpec(shape=(None, img_size, img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1) if 'inception' in model_type.lower() else (None,), dtype=tf.float32)
            )
        )
        
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    # ✅ FOR TRAINING: Use the corrected approach (augmentation before preprocessing)
    # Apply a DataFrame-level shuffle first
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
    
    # Create tensor slices
    ds_paths = tf.data.Dataset.from_tensor_slices(paths)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)

    def _load_raw(path, y):
        """Load and resize image WITHOUT preprocessing - for training augmentation"""
        img_data = tf.io.read_file(path)
        img_data = tf.image.decode_png(img_data, channels=3)
        img_data = tf.image.resize(img_data, (img_size, img_size))
        # Keep raw image data [0,255] for augmentation
        img_data = tf.cast(img_data, tf.float32)
        
        # Convert labels to float32
        y_data = tf.cast(y, tf.float32)
        
        # For VGG16, provide labels as flat scalars
        if 'vgg16' in model_type.lower():
            return img_data, tf.squeeze(y_data)
        else:
            return img_data, tf.reshape(y_data, [1])

    def _apply_preprocessing(img, y):
        """Apply preprocessing AFTER augmentation"""
        processed_img = preprocess_fn(img)
        return processed_img, y

    # Create dataset zip
    ds = tf.data.Dataset.zip((ds_paths, ds_labels))
    
    # Use a large buffer size for better shuffling
    buffer_size = len(paths)
    ds = ds.shuffle(buffer_size, seed=42, reshuffle_each_iteration=True)
    
    # 1. Load raw images (no preprocessing yet)
    ds = ds.map(_load_raw, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 2. Apply augmentation to raw image data (0-255 range)
    ds = ds.map(lambda x, y: (_AUG(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    # 3. Apply preprocessing AFTER augmentation
    ds = ds.map(_apply_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Create batches without further shuffling
    batched_ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    
    # Debug: check batch properties and content
    print("\nChecking class distribution in first 5 batches:")
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

# ============================================================================
# COMPLETE BASELINE IMAGEDATAGENERATOR REPLACEMENT
# Replace your entire dataloaders.py get_loaders function with this
# ============================================================================

def get_loaders_baseline_style(cfg):
    """
    EXACT baseline ImageDataGenerator approach
    This completely replaces the tf.data pipeline with baseline approach
    """
    import pandas as pd
    import pathlib
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
    from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
    
    if cfg['dataset'].lower() != 'cbis_ddsm':
        raise ValueError("Only cbis_ddsm supported for now.")

    root = pathlib.Path(cfg['data_root'])
    _download_and_unzip(root)  # Your existing function

    # Load or create dataframe (your existing logic)
    csv_path = cfg.get('csv_path')
    if csv_path and pathlib.Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        df = _make_dataframe(root, cfg['view'])  # Your existing function
        if csv_path:
            pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)

    # ✅ EXACT BASELINE SETUP
    img_size = cfg.get('input_size', 512)
    batch_size = cfg.get('batch_size', 8)
    model_type = cfg["model"].lower()

    # Choose preprocessing function (baseline approach)
    if model_type == "vgg16":
        print("Using VGG16 preprocessing")
        preprocess_fn = preprocess_vgg
    elif model_type == "inceptionv3":
        print("Using InceptionV3 preprocessing")
        preprocess_fn = preprocess_inception
    else:
        raise ValueError("Unknown model architecture")

    # Prepare dataframes with baseline-compatible format
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    
    # ✅ CRITICAL: Add label_str column exactly like baseline
    train_df['label_str'] = train_df['label'].map({0: 'benign', 1: 'malignant'})
    val_df['label_str'] = val_df['label'].map({0: 'benign', 1: 'malignant'})
    
    # ✅ EXACT BASELINE IMAGE DATA GENERATORS
    
    # Training generator with augmentation (EXACT baseline parameters)
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,  # ✅ Applied here like baseline
        rotation_range=20,                     # ✅ Exact baseline
        width_shift_range=0.1,                 # ✅ Exact baseline  
        height_shift_range=0.1,                # ✅ Exact baseline
        shear_range=0.2,                       # ✅ Exact baseline
        zoom_range=0.2,                        # ✅ Exact baseline
        horizontal_flip=True,                  # ✅ Exact baseline
        fill_mode='nearest'                    # ✅ Exact baseline
    )
    
    # Validation generator with NO augmentation (EXACT baseline)
    val_test_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    
    # ✅ EXACT BASELINE FLOW_FROM_DATAFRAME CALLS
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Val set: {len(val_df)} samples")
    
    # Training data generator
    train_generator = train_gen.flow_from_dataframe(
        train_df,
        x_col='filepath',                      # ✅ Your column name
        y_col='label_str',                     # ✅ Baseline format
        target_size=(img_size, img_size),      # ✅ Baseline format
        batch_size=batch_size,                 # ✅ Baseline batch size
        class_mode='binary'                    # ✅ Exact baseline
    )
    
    # Validation data generator  
    val_generator = val_test_gen.flow_from_dataframe(
        val_df,
        x_col='filepath',                      # ✅ Your column name  
        y_col='label_str',                     # ✅ Baseline format
        target_size=(img_size, img_size),      # ✅ Baseline format
        batch_size=batch_size,                 # ✅ Baseline batch size
        class_mode='binary'                    # ✅ Exact baseline
    )
    
    print(f"✅ Created baseline-style generators:")
    print(f"   Training: {train_generator.samples} samples, {len(train_generator)} batches")
    print(f"   Validation: {val_generator.samples} samples, {len(val_generator)} batches")
    
    return train_generator, val_generator

# ============================================================================
# UPDATE YOUR TRAIN_HELPERS.PY
# ============================================================================

def train_once_baseline_style(cfg, outdir):
    """
    Updated train_once to work with baseline ImageDataGenerator approach
    """
    outdir = pathlib.Path(outdir)
    set_seed(cfg["seed"])

    # ✅ Use baseline data loading
    print(f"\n{'='*40}\nLoading data for {cfg['model']} model (BASELINE STYLE)\n{'='*40}")
    train_generator, val_generator = get_loaders_baseline_style(cfg)

    # Build model (your existing function)
    print(f"\n{'='*40}\nBuilding {cfg['model']} model\n{'='*40}")
    model = build_model(cfg)  

    # ✅ EXACT BASELINE CALLBACKS (val_loss monitoring)
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",        # ✅ Baseline monitors val_loss
            mode="min",                # ✅ Minimize loss
            patience=6,
            restore_best_weights=True, 
            verbose=1
        ),
        
        tf.keras.callbacks.ModelCheckpoint(
            filepath=outdir / "best.weights.h5",
            monitor="val_loss",        # ✅ Baseline monitors val_loss
            mode="min",                # ✅ Minimize loss
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",        # ✅ Baseline monitors val_loss
            mode="min",                # ✅ Minimize loss
            factor=0.5, 
            patience=4,
            min_lr=1e-6, 
            verbose=1
        ),
    ]   

    # ✅ EXACT BASELINE TRAINING CALL
    hist = model.fit(
        train_generator,                       # ✅ Use generator directly
        validation_data=val_generator,         # ✅ Use generator directly  
        epochs=cfg.get("epochs", 15),
        verbose=2,
        callbacks=cb,
    )

    # ✅ BASELINE-STYLE EVALUATION (simplified)
    # For now, just get basic metrics from training history
    final_val_auc = max(hist.history.get('val_AUC', [0]))
    final_val_acc = hist.history.get('val_accuracy', [0])[-1]
    
    metrics = {
        'val_auc': float(final_val_auc),
        'val_acc': float(final_val_acc),
        'history': {k: [float(v) for v in hist.history[k]] for k in hist.history}
    }

    # Save artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir/"config.json")
    save_json(metrics, outdir/"metrics.json")
    model.save(outdir / "model.keras")

    return metrics

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
1. Add get_loaders_baseline_style() to your dataloaders.py

2. Replace your train_once function call in the ablation sweep with:
   metrics_train = train_once_baseline_style(cfg, outdir)

3. For comprehensive evaluation, you can still use the baseline TTA approach:
   - The generators can be used with your existing TTA functions
   - Just make sure to create test_generator the same way

4. Expected outcome:
   - Epoch 1 val_AUC should jump from ~0.77 to ~0.82 (matching baseline)
   - Training dynamics should match baseline exactly
   - LR reduction should happen around epoch 14 (not epoch 6)
"""

# Export the augmentation layer (existing line)
AUG_LAYER = _AUG