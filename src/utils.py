import json, random, pathlib, numpy as np, tensorflow as tf

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

def save_json(obj, path):
    """Write dict→pretty JSON."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
