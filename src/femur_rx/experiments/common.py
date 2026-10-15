"""Shared bookkeeping for experiment runners: per-split results and reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from femur_rx.data.dataset import KEY_COLUMNS, FemurDataset
from femur_rx.evaluation import compute_metrics, percent_error, plots, summarise_splits

log = logging.getLogger(__name__)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def format_metrics(metrics: dict) -> str:
    return f"R2 {metrics['r2']:.4f} | RMSE {metrics['rmse']:.4g} | MAE {metrics['mae']:.4g}"


class SplitResults:
    """Collects test-set predictions split by split and writes the reports."""

    def __init__(self, dataset: FemurDataset, out_dir: Path):
        self.dataset = dataset
        self.out_dir = out_dir
        self.metrics: list[dict] = []
        self.predictions: list[pd.DataFrame] = []

    def add(self, split_name: str, test_idx: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = self.dataset.y[test_idx]
        metrics = {"split": split_name, "n_test": len(test_idx), **compute_metrics(y_true, y_pred)}
        self.metrics.append(metrics)
        log.info("[%s] %s", split_name, format_metrics(metrics))

        pred = self.dataset.metadata.iloc[test_idx][KEY_COLUMNS].copy()
        pred.insert(0, "split", split_name)
        pred["y_true"] = y_true
        pred["y_pred"] = y_pred
        pred["error_percent"] = percent_error(y_true, y_pred)
        self.predictions.append(pred)
        return metrics

    def finalise(self) -> dict:
        per_split = summarise_splits(self.metrics)
        per_split.to_csv(self.out_dir / "metrics_per_split.csv")
        predictions = pd.concat(self.predictions, ignore_index=True)
        predictions.to_csv(self.out_dir / "predictions.csv", index=False)
        pooled = compute_metrics(predictions["y_true"], predictions["y_pred"])

        plots.plot_predicted_vs_actual(predictions, self.out_dir / "predicted_vs_actual.png",
                                       f"Predicted vs actual (all test splits), R2 = {pooled['r2']:.3f}")
        plots.plot_percent_error(predictions, self.out_dir / "percent_error.png", "Relative error (all test splits)")
        plots.plot_residual_histogram(predictions, self.out_dir / "residuals_histogram.png",
                                      "Residuals (all test splits)")
        if len(self.metrics) > 1:
            plots.plot_metric_boxplot(per_split.drop(index=["mean", "std"]), ["r2"],
                                      self.out_dir / "r2_boxplot.png", "R2 across splits")

        summary = {
            "mean": per_split.loc["mean"].to_dict(),
            "std": per_split.loc["std"].to_dict(),
            "pooled": pooled,  # all test predictions together
            "splits": self.metrics,
        }
        write_json(self.out_dir / "metrics.json", summary)
        log.info("Mean over %d split(s):\n%s", len(self.metrics), per_split.loc[["mean", "std"]].T.to_string())
        return summary
