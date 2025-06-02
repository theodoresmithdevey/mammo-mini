"""
Core training wrapper for mammo-mini.

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
from .dataloaders import get_loaders
from .utils import set_seed, save_json

# ───────────────────────────────────────────────────────────────────── #
#  Model builder                                                       #
# ───────────────────────────────────────────────────────────────────── #

def build_model(cfg):
    
    img_size = cfg["input_size"]
    arch     = cfg["model"]
    weights  = None if cfg["weights"] == "random" else "imagenet"

    if arch == "vgg16":
        base = applications.VGG16(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    elif arch == "inceptionv3":
        base = applications.InceptionV3(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    else:
        raise ValueError(f"Unsupported model: {arch}")

    # 🔒 Apply conditional freezing logic
    if cfg.get("freeze_scheme") == "chougrad" and cfg["weights"] == "imagenet":
        if arch == "vgg16":
            freeze_until(base, "block4_conv1")  # unfreezes last two conv blocks
        else:
            freeze_until(base, "mixed9")        # unfreezes final inception modules

    # 🧠 Classification head (updated)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)


    model = models.Model(base.input, out)

    # 🛠 Optimizer + LR
    lr = 1e-3 if cfg["optimiser"].lower() == "adam" else 1e-2
    opt = optimizers.Adam(learning_rate=lr) if cfg["optimiser"].lower() == "adam" else optimizers.SGD(learning_rate=lr, momentum=0.9)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc", "AUC"])
    return model


def compile_model(model, cfg):
    lr = 1e-3 if cfg["optimiser"]=="Adam" else 1e-2
    if cfg["optimiser"].lower() == "adam":
        opt = tf.keras.optimizers.Adam(lr)
    else:
        opt = tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model

# ───────────────────────────────────────────────────────────────────── #
#  Optional test-time augmentation (horizontal flip)                   #
# ───────────────────────────────────────────────────────────────────── #

def _tta_predict(model, batch, tta_passes=10):
    preds = []
    for _ in range(tta_passes):
        augmented = _AUG(batch, training=True)
        p = model.predict(augmented, verbose=0)
        preds.append(p)
    return np.mean(preds, axis=0)

def evaluate(model, val_ds, tta=False, tta_passes=10):
    y_true, y_pred = [], []
    for x, y in val_ds:
        y_true.extend(y.numpy().ravel())
        if tta:
            y_pred.extend(_tta_predict(model, x, tta_passes).ravel())
        else:
            y_pred.extend(model.predict(x, verbose=0).ravel())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Optional: threshold sweep here
    auc = skm.roc_auc_score(y_true, y_pred)
    acc = skm.accuracy_score(y_true, np.round(y_pred))  # default threshold=0.5
    return dict(val_auc=float(auc), val_acc=float(acc),
                y_true=y_true.tolist(), y_pred=y_pred.tolist())

# ------------------------------------------------------------------
#  Freeze strategy helpers
# ------------------------------------------------------------------

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


# ───────────────────────────────────────────────────────────────────── #
#  Main wrapper                                                         #
# ───────────────────────────────────────────────────────────────────── #

def train_once(cfg, outdir):
    outdir = pathlib.Path(outdir)
    set_seed(cfg["seed"])

    # data
    train_ds, val_ds = get_loaders(cfg)

    # model
    model = compile_model(build_model(cfg), cfg)

    cb = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="max", patience=6,
        restore_best_weights=True, verbose=1),
    
    tf.keras.callbacks.ModelCheckpoint(
        filepath=outdir / "best.h5", monitor="val_auc",
        mode="max", save_best_only=True, verbose=1),
    
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4,
        min_lr=1e-6, verbose=1),
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
    save_json(cfg,   outdir/"config.json")
    save_json(metrics, outdir/"metrics.json")
    model.save(outdir/"model.h5")

    return metrics



