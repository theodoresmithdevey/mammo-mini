"""
Shared analysis helpers for mammo-mini.

Typical usage in a Colab notebook:
    from mammo.analysis import load_runs_df, plot_roc, plot_cm, plot_history

    df = load_runs_df('/content/drive/MyDrive/runs')
    df[['model','view','weights','optimiser','val_auc']]

    # plot ROC for a single run
    plot_roc('/content/drive/MyDrive/runs/7f2a9bce/metrics.json')
"""

import json, pathlib, numpy as np, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns, sklearn.metrics as skm

# ------------------------------------------------------------------ #
# 1.  Load all runs into a DataFrame                                  #
# ------------------------------------------------------------------ #
def load_runs_df(root):
    """Scan root/*/metrics.json & config.json -> tidy DataFrame."""
    rows = []
    root = pathlib.Path(root)
    for metrics_path in root.glob('*/metrics.json'):
        cfg_path = metrics_path.with_name('config.json')
        if not cfg_path.exists():
            continue
        m = json.load(open(metrics_path))
        c = json.load(open(cfg_path))
        rows.append({**c, **m, "run_id": metrics_path.parent.name})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# 2.  ROC curve                                                       #
# ------------------------------------------------------------------ #
def plot_roc(metrics_path, ax=None):
    """Plot ROC curve given metrics.json path or metrics dict."""
    if isinstance(metrics_path, (str, pathlib.Path)):
        m = json.load(open(metrics_path))
    else:
        m = metrics_path
    y_true = np.array(m["y_true"])
    y_pred = np.array(m["y_pred"])
    fpr, tpr, _ = skm.roc_curve(y_true, y_pred)
    auc       = skm.roc_auc_score(y_true, y_pred)
    ax = ax or plt.gca()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set_xlabel("False-Positive Rate")
    ax.set_ylabel("True-Positive Rate")
    ax.legend()
    ax.set_title("ROC Curve")
    plt.show()


# ------------------------------------------------------------------ #
# 3.  Confusion matrix                                                #
# ------------------------------------------------------------------ #
def plot_cm(metrics_path, threshold=0.5, ax=None):
    if isinstance(metrics_path, (str, pathlib.Path)):
        m = json.load(open(metrics_path))
    else:
        m = metrics_path
    y_true = np.array(m["y_true"])
    y_pred = np.array(m["y_pred"]) >= threshold
    cm = skm.confusion_matrix(y_true, y_pred)
    ax = ax or plt.gca()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (th = {threshold})")
    plt.show()


# ------------------------------------------------------------------ #
# 4.  Training curves                                                 #
# ------------------------------------------------------------------ #
def plot_history(metrics_path, ax=None):
    if isinstance(metrics_path, (str, pathlib.Path)):
        m = json.load(open(metrics_path))
    else:
        m = metrics_path
    hist = m["history"]
    ax = ax or plt.gca()
    ax.plot(hist["loss"], label="train-loss")
    ax.plot(hist["val_loss"], label="val-loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")
    ax.legend()
    plt.show()
