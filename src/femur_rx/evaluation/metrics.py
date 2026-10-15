"""Regression metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def percent_error(y_true, y_pred) -> np.ndarray:
    """Signed relative error ``100 * (y_true - y_pred) / y_true`` per sample."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return 100 * (y_true - y_pred) / y_true


def compute_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan"),
        "mape_percent": 100 * mean_absolute_percentage_error(y_true, y_pred),
    }


def summarise_splits(split_metrics: list[dict]) -> pd.DataFrame:
    """One row per split plus ``mean`` and ``std`` rows (scalar metrics only)."""
    rows = [{k: v for k, v in m.items() if not isinstance(v, list)} for m in split_metrics]
    df = pd.DataFrame(rows).set_index("split")
    numeric = df.select_dtypes("number")
    summary = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=0)}).T
    return pd.concat([df, summary])
