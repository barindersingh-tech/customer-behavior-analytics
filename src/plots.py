"""Matplotlib figures saved to static/plots (used by the dashboard and reports)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.features import FEATURE_LABELS


def _save(fig, name):
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / name, dpi=130)
    plt.close(fig)


def silhouette_plot(sil):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(list(sil.keys()), list(sil.values()), marker="o", color="#2563eb")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Choosing the number of segments")
    ax.grid(alpha=0.3)
    _save(fig, "silhouette.png")


def segment_scatter(scores):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, grp in scores.groupby("segment"):
        ax.scatter(grp["recency"], np.log10(grp["monetary"].clip(lower=1)), s=8, alpha=0.5, label=name)
    ax.set_xlabel("Recency (days since last order)")
    ax.set_ylabel("log10 total spend")
    ax.set_title("Customer segments")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "segments.png")


def roc_plot(curves, results):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test snapshot)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "roc.png")


def importance_plot(importance):
    items = list(importance.items())[:8][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.barh([FEATURE_LABELS[k] for k, _ in items], [v for _, v in items], color="#2563eb")
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Most important features")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "importance.png")


def confusion_plot(cm):
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], ["Stays", "Churns"])
    ax.set_yticks([0, 1], ["Stays", "Churns"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    _save(fig, "confusion.png")
