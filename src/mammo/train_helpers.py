"""
Core training wrapper for mammo-mini.

train_helpers.py

Supports ONLY:
  • model  : 'vgg16' | 'inceptionv3'
  • weights: 'imagenet' | 'random'
  • optimiser: 'Adam' | 'SGD'
  • optional TTA (horizontal flip)

Public:
    train_once(cfg: dict, outdir: str|Path) -> metrics dict
"""

from tensorflow.keras import layers, models, optimizers, applications, regularizers
import pathlib, json, numpy as np, tensorflow as tf, sklearn.metrics as skm
from .dataloaders import get_loaders, AUG_LAYER
from .utils import set_seed, save_json

# ───────────────────────────────────────────────────────────────────── #
#  Model builder                                                       #
# ───────────────────────────────────────────────────────────────────── #

def build_model(cfg):
    
    img_size = cfg["input_size"]
    arch = cfg["model"]
    weights = None if cfg["weights"] == "random" else "imagenet"

    if arch == "vgg16":
        base = applications.VGG16(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
        # Print model input shape
        print(f"VGG16 input shape: {base.input_shape}")
    elif arch == "inceptionv3":
        base = applications.InceptionV3(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
        print(f"InceptionV3 input shape: {base.input_shape}")
    else:
        raise ValueError(f"Unsupported model: {arch}")

    # 🎯 FREEZING STRATEGY: Only apply to ImageNet pretrained models
    if cfg["weights"] == "imagenet":
        # Freeze everything first
        for layer in base.layers:
            layer.trainable = False
        
        # Unfreeze last 2 conv blocks (Chougrad strategy)
        if arch == "vgg16":
            # For VGG16, explicitly unfreeze block4 and block5
            for layer in base.layers:
                if 'block4' in layer.name or 'block5' in layer.name:
                    layer.trainable = True
        elif arch == "inceptionv3":
            for layer in base.layers[-44:]:
                layer.trainable = True
    
    # For random weights: all layers trainable (no freezing)
    # This happens automatically since layers are trainable by default

    # 🧠 Classification head
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(base.input, out)
    
    # Print model output shape for debugging
    print(f"Model output shape: {model.output_shape}")

    # 🛠 OPTIMIZER + LOSS: Different strategies for random vs pretrained
    lr = 1e-5 if cfg["model"].lower() == "vgg16" else 1e-3
    
    if cfg["optimiser"].lower() == "adam":
        opt = optimizers.Adam(learning_rate=lr)
    else:
        opt = optimizers.SGD(learning_rate=lr, momentum=0.9)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="AUC")]
    )
    
    return model


# ───────────────────────────────────────────────────────────────────── #
#  Optional test-time augmentation (horizontal flip)                   #
# ───────────────────────────────────────────────────────────────────── #

def _tta_predict(model, df_subset, cfg, tta_passes=10):
    """TTA with exact baseline replication using ImageDataGenerator"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
    from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
    import numpy as np
    
    # Choose preprocessing function based on model type (matching baseline)
    model_type = cfg.get("model", "inceptionv3").lower()
    if model_type == "vgg16":
        preprocess_fn = preprocess_vgg
    else:
        preprocess_fn = preprocess_inception
    
    # Create exact baseline TTA generator
    tta_gen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,          # Exact baseline: ±15 degrees
        zoom_range=0.1,             # Exact baseline: ±10% zoom
        horizontal_flip=True,       # Exact baseline: random horizontal flip
        fill_mode='nearest'         # Exact baseline: fill mode
    )
    
    # Create dataframe copy with baseline-compatible label format
    df_tta = df_subset.copy()
    df_tta['label_str'] = df_tta['label'].map({0: 'benign', 1: 'malignant'})
    
    n = len(df_tta)
    all_preds = np.zeros((n, tta_passes))
    
    # Exact baseline TTA loop: recreate data pipeline for each pass
    for t in range(tta_passes):
        print(f"TTA pass {t+1}/{tta_passes}")  # Match baseline logging
        
        tta_batch = tta_gen.flow_from_dataframe(
            df_tta,
            x_col='filepath',           # Column name in ablation dataframes
            y_col='label_str',          # Binary string labels like baseline
            target_size=(cfg.get('input_size', 512), cfg.get('input_size', 512)),
            class_mode='binary',        # Exact baseline: binary classification
            batch_size=cfg.get('batch_size', 16),
            shuffle=False               # Exact baseline: no shuffling for consistent ordering
        )
        
        preds = model.predict(tta_batch, verbose=0)  # Exact baseline: silent prediction
        all_preds[:, t] = preds.flatten()           # Exact baseline: flatten and store
    
    return np.mean(all_preds, axis=1)  # Exact baseline: average across TTA passes


def evaluate(model, val_ds, tta=False, tta_passes=10, cfg=None, df_subset=None):
    """
    Backward compatible evaluate function
    
    Args:
        model: Trained model
        val_ds: Validation dataset (tf.data.Dataset)
        tta: Whether to use test-time augmentation
        tta_passes: Number of TTA passes
        cfg: Configuration dict (optional - if None and tta=True, will skip TTA)
        df_subset: DataFrame subset for TTA (optional - if None and tta=True, will skip TTA)
    """
    import numpy as np
    import sklearn.metrics as skm

    # ✅ BACKWARD COMPATIBILITY: If TTA is requested but parameters are missing, disable TTA
    if tta and (cfg is None or df_subset is None):
        print("⚠️  Warning: TTA requested but cfg/df_subset not provided. Falling back to non-TTA evaluation.")
        tta = False

    if tta:
        # Use baseline-style TTA approach
        print(f"\nUsing baseline-style TTA with {tta_passes} passes")
        y_pred = _tta_predict(model, df_subset, cfg, tta_passes)
        y_true = df_subset['label'].values.astype(float)
        
        print(f"\nFinal evaluation arrays:")
        print(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
        print(f"y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
        print(f"y_true unique values: {np.unique(y_true)}")
        print(f"y_pred range: {np.min(y_pred):.4f} to {np.max(y_pred):.4f}")
        
    else:
        # Original batch-by-batch approach for non-TTA
        y_true, y_pred = [], []
        
        print("\nValidation dataset inspection:")
        for batch_x, batch_y in val_ds.take(1):
            print(f"Validation batch_x shape: {batch_x.shape}")
            print(f"Validation batch_y shape: {batch_y.shape}")
            print(f"Validation batch_y dtype: {batch_y.dtype}")
            print(f"First few validation labels: {batch_y.numpy().flatten()[:10]}")
        
        for batch_x, batch_y in val_ds:
            # Print the raw batch_y for debugging
            if len(y_true) == 0:
                print(f"Raw batch_y: {batch_y}")
            
            # Ensure batch_y is properly converted to a numpy array
            batch_y_np = batch_y.numpy().flatten()
            y_true.extend(batch_y_np)

            batch_preds = model.predict(batch_x, verbose=0)

            # Flatten predictions properly
            batch_preds_np = batch_preds.flatten()
            y_pred.extend(batch_preds_np)
            
            # Early debugging - print first batch
            if len(y_true) <= len(batch_y_np):
                print(f"First batch true labels: {batch_y_np}")
                print(f"First batch predictions: {batch_preds_np}")

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Verify final arrays
        print(f"\nFinal evaluation arrays:")
        print(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
        print(f"y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
        print(f"y_true unique values: {np.unique(y_true)}")
        print(f"y_pred range: {np.min(y_pred):.4f} to {np.max(y_pred):.4f}")

    auc = skm.roc_auc_score(y_true, y_pred)
    acc = skm.accuracy_score(y_true, np.round(y_pred))  # threshold 0.5
    return dict(val_auc=float(auc), val_acc=float(acc),
                y_true=y_true.tolist(), y_pred=y_pred.tolist())


# ───────────────────────────────────────────────────────────────────── #
#  Freeze strategy helpers                                                #
# ───────────────────────────────────────────────────────────────────── #

def freeze_until(model, stop_substring: str):
    """
    Set layer.trainable = False for every layer whose name does NOT
    contain `stop_substring`.  Everything at and after the first match
    stays trainable.

    Example:
        freeze_until(base_model, "block4_conv1")   # VGG16
        freeze_until(base_model, "mixed9")         # InceptionV3
    """
    reached = False
    for layer in model.layers:
        if stop_substring in layer.name:
            reached = True
        layer.trainable = reached  # False until we "reach" the block

def debug_trainable_layers(model, n_last=10):
    """Print last n layer names + trainable flag for sanity check."""
    for l in model.layers[-n_last:]:
        print(f"{l.name:25s}  trainable={l.trainable}")


# ------------------------------------------------------------------
#  Compile model function (needed for the ablation sweep)
# ------------------------------------------------------------------

def compile_model(model, cfg):
    """Recompile an existing model with the appropriate optimizer and loss."""
    lr = 1e-3 if cfg["optimiser"].lower() == "adam" else 1e-2
    
    if cfg["optimiser"].lower() == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)

    # Use different loss strategies based on weights

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="AUC")]
    )
    
    return model


# ───────────────────────────────────────────────────────────────────── #
#  Main wrapper                                                         #
# ───────────────────────────────────────────────────────────────────── #

def train_once(cfg, outdir):
    outdir = pathlib.Path(outdir)
    set_seed(cfg["seed"])

    # data
    print(f"\n{'='*40}\nLoading data for {cfg['model']} model\n{'='*40}")
    train_ds, val_ds = get_loaders(cfg)

    # model - build once
    print(f"\n{'='*40}\nBuilding {cfg['model']} model\n{'='*40}")
    model = build_model(cfg)  
    
    # Debug trainable layers to verify VGG16 has proper layer freezing
    if cfg["model"] == "vgg16":
        print("Verifying VGG16 trainable layers:")
        for l in model.layers[-20:]:
            print(f"{l.name:25s}  trainable={l.trainable}")
    
    # Fix early stopping callback
    cb = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",           
        mode="min",                
        patience=6,                 # ✅ matches baseline patience
        restore_best_weights=True, 
        verbose=1
    ),
    
    tf.keras.callbacks.ModelCheckpoint(
        filepath=outdir / "best.weights.h5",
        monitor="val_loss",           
        mode="min",       
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    ),
    
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",           
        mode="min",   
        factor=0.5,                 # ✅ matches baseline
        patience=4,                 # ✅ matches baseline  
        min_lr=1e-6,                # ✅ matches baseline
        verbose=1
    ),
]

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.get("epochs", 15),
        verbose=2,
        callbacks=cb,
    )

    # evaluation
    metrics = evaluate(model, val_ds, tta=cfg.get("tta", False))
    metrics["history"] = {k: [float(v) for v in hist.history[k]]
                          for k in hist.history}

    # persist artefacts
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, outdir/"config.json")
    save_json(metrics, outdir/"metrics.json")
    model.save(outdir / "model.keras")

    return metrics