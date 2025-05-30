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

import pathlib, json, numpy as np, tensorflow as tf, sklearn.metrics as skm
from .dataloaders import get_loaders
from .utils import set_seed, save_json

# ───────────────────────────────────────────────────────────────────── #
#  Model builder                                                       #
# ───────────────────────────────────────────────────────────────────── #

def build_model(cfg):
    arch      = cfg["model"].lower()
    img_size  = cfg["input_size"]
    in_shape  = (img_size, img_size, 3)
    weights   = "imagenet" if cfg["weights"]=="imagenet" else None
    freeze_bn = cfg.get("freeze_blocks", 0)

    if arch == "vgg16":
        base = tf.keras.applications.VGG16(include_top=False,
                                           weights=weights,
                                           input_shape=in_shape,
                                           pooling="avg")
    elif arch == "inceptionv3":
        base = tf.keras.applications.InceptionV3(include_top=False,
                                                 weights=weights,
                                                 input_shape=in_shape,
                                                 pooling="avg")
    else:
        raise ValueError("arch must be vgg16 or inceptionv3")

    # optional fine-tuning depth
    if freeze_bn > 0:
        for layer in base.layers[:-freeze_bn]:
            layer.trainable = False

    x   = tf.keras.layers.Dropout(0.5)(base.output)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(base.input, out)

def compile_model(model, cfg):
    lr = cfg.get("lr", 1e-4)
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

def _tta_predict(model, batch):
    p1 = model.predict(batch, verbose=0)
    p2 = model.predict(tf.image.flip_left_right(batch), verbose=0)
    return (p1 + p2) / 2.0

def evaluate(model, val_ds, tta=False):
    y_true, y_pred = [], []
    for x, y in val_ds:
        y_true.extend(y.numpy().ravel())
        if tta:
            y_pred.extend(_tta_predict(model, x).ravel())
        else:
            y_pred.extend(model.predict(x, verbose=0).ravel())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auc = skm.roc_auc_score(y_true, y_pred)
    acc = skm.accuracy_score(y_true, np.round(y_pred))
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
            monitor="val_auc", mode="max", patience=3,
            restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            outdir/"best.h5", monitor="val_auc",
            mode="max", save_best_only=True, verbose=0),
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
