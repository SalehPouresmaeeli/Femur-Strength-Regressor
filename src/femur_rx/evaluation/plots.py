"""Figures saved to disk (non-interactive backend, safe on headless machines)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_predicted_vs_actual(predictions: pd.DataFrame, path: Path, title: str = "Predicted vs actual"):
    """Scatter of ``y_pred`` against ``y_true``, one colour per split, with the identity line."""
    fig, ax = plt.subplots(figsize=(6, 5))
    splits = predictions["split"].unique()
    for split in splits:
        part = predictions[predictions["split"] == split]
        ax.scatter(part["y_true"], part["y_pred"], s=12, alpha=0.6, label=split if len(splits) <= 10 else None)
    lo = min(predictions["y_true"].min(), predictions["y_pred"].min())
    hi = max(predictions["y_true"].max(), predictions["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", lw=1, label="y_pred = y_true")
    ax.set(xlabel="Actual", ylabel="Predicted", title=title)
    ax.legend(fontsize="small")
    save_figure(fig, path)


def plot_percent_error(predictions: pd.DataFrame, path: Path, title: str = "Relative error"):
    """Signed relative error (%) against the actual value."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(predictions["y_true"], predictions["error_percent"], s=12, alpha=0.5)
    ax.axhline(0, color="grey", linestyle="--", lw=1)
    ax.set(xlabel="Actual", ylabel="Error (%)", title=title)
    save_figure(fig, path)


def plot_residual_histogram(predictions: pd.DataFrame, path: Path, title: str = "Residuals"):
    """Histogram of ``y_pred - y_true``."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(predictions["y_pred"] - predictions["y_true"], bins=30, color="grey")
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set(xlabel="Predicted - actual", ylabel="Count", title=title)
    save_figure(fig, path)


def plot_metric_boxplot(per_split: pd.DataFrame, columns: list[str], path: Path, title: str = ""):
    """Box plot of metrics across splits (e.g. a robustness check over many seeds)."""
    columns = [c for c in columns if c in per_split.columns]
    fig, ax = plt.subplots(figsize=(1.6 * len(columns) + 2, 4))
    ax.boxplot([per_split[c].dropna() for c in columns])
    ax.set_xticks(range(1, len(columns) + 1), columns)
    ax.set(ylabel="Value", title=title)
    save_figure(fig, path)


def plot_image_grid(images: dict[str, np.ndarray], path: Path, cols: int = 3, flip: bool = True):
    """Show named images side by side (flipped vertically with ``np.flipud`` when ``flip`` is set)."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, min(cols, n), figsize=(4 * min(cols, n), 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, img) in zip(axes.ravel(), images.items()):
        ax.imshow(np.flipud(img) if flip else img, cmap="gray")
        ax.set_title(title)
    save_figure(fig, path)
