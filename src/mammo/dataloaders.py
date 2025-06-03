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
def _build_tfds(df, img_size, batch, is_train, preprocess_fn):
    paths = df['filepath'].values
    labels = df['label'].values

    ds_paths = tf.data.Dataset.from_tensor_slices(paths)
    ds_labels = tf.data.Dataset.from_tensor_slices(labels)

    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE[img_size])
        img = preprocess_fn(img)
        return img, tf.expand_dims(y, -1)

    ds = tf.data.Dataset.zip((ds_paths, ds_labels))
    
    if is_train:
        # 🎯 CRITICAL FIX: Shuffle BEFORE expensive operations!
        ds = ds.shuffle(4096, seed=None, reshuffle_each_iteration=True)
        
        # Then do expensive mapping
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Then augmentation
        ds = ds.map(lambda x, y: (_AUG(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        
        # Optional: Light shuffle after augmentation
        ds = ds.shuffle(512, seed=None, reshuffle_each_iteration=True)
    else:
        # For validation: just mapping, no shuffle
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


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

    # choose preprocessing function based on model
    if cfg["model"].lower() == "vgg16":
        preprocess_fn = preprocess_vgg
    elif cfg["model"].lower() == "inceptionv3":
        preprocess_fn = preprocess_inception
    else:
        raise ValueError("Unknown model architecture")

    train_ds = _build_tfds(df[df.split == 'train'], img_size, batch, is_train=True,  preprocess_fn=preprocess_fn)
    val_ds   = _build_tfds(df[df.split == 'val'],   img_size, batch, is_train=False, preprocess_fn=preprocess_fn)
    return train_ds, val_ds

# Export the augmentation layer
AUG_LAYER = _AUG