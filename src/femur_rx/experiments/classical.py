"""Hyper-parameter search + grouped evaluation for the regression pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV

from femur_rx.config import ExperimentConfig, SearchConfig
from femur_rx.data.dataset import FemurDataset
from femur_rx.evaluation import compute_metrics, group_train_test_split, iter_splits
from femur_rx.experiments.common import SplitResults, format_metrics, write_json
from femur_rx.models import build_pipeline, set_random_state

log = logging.getLogger(__name__)

_DISTRIBUTIONS = {"uniform": stats.uniform, "randint": stats.randint, "loguniform": stats.loguniform}


def _to_search_space(value):
    """Turn ``{distribution: uniform, loc: .., scale: ..}`` specs into scipy distributions."""
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        return [_to_search_space(v) for v in value]
    if isinstance(value, dict) and "distribution" in value:
        spec = dict(value)
        kind = spec.pop("distribution")
        try:
            return _DISTRIBUTIONS[kind](**spec)
        except KeyError:
            raise ValueError(f"Unknown distribution {kind!r}; choose from {sorted(_DISTRIBUTIONS)}") from None
    if isinstance(value, dict):
        return {k: _to_search_space(v) for k, v in value.items()}
    return value


def run_search(pipeline, X, dataset: FemurDataset, cfg: SearchConfig, out_dir: Path) -> dict:
    """Grid/random search on a grouped train split; report the held-out test score."""
    train, test = group_train_test_split(dataset.y, dataset.groups, cfg.test_size, cfg.random_state)
    space = _to_search_space(cfg.param_grid)
    common = dict(cv=GroupKFold(n_splits=cfg.cv_splits), scoring=cfg.scoring, n_jobs=cfg.n_jobs,
                  verbose=1, refit=True)
    if cfg.kind == "grid":
        search = GridSearchCV(pipeline, space, **common)
    elif cfg.kind == "random":
        search = RandomizedSearchCV(pipeline, space, n_iter=cfg.n_iter, random_state=cfg.random_state, **common)
    else:
        raise ValueError(f"Unknown search kind {cfg.kind!r}; use 'grid' or 'random'")

    log.info("Running %s search (%s) on %d training rows", cfg.kind, cfg.scoring, len(train))
    search.fit(X[train], dataset.y[train], groups=dataset.groups[train])

    test_metrics = compute_metrics(dataset.y[test], search.best_estimator_.predict(X[test]))
    log.info("Best params: %s | CV %s = %.4f | held-out %s",
             search.best_params_, cfg.scoring, search.best_score_, format_metrics(test_metrics))

    search_dir = out_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).drop(columns="params").sort_values("rank_test_score").to_csv(
        search_dir / "cv_results.csv", index=False)
    write_json(search_dir / "best.json", {"best_params": search.best_params_, "best_cv_score": search.best_score_,
                                          "scoring": cfg.scoring, "held_out_test": test_metrics})
    return search.best_params_


def run_classical_experiment(cfg: ExperimentConfig, dataset: FemurDataset, out_dir: Path) -> dict:
    X = dataset.to_matrix()
    pipeline = build_pipeline(cfg.model, cfg.preprocessing, dataset.image_shape)
    log.info("Pipeline: %s", " -> ".join(name for name, _ in pipeline.steps))

    if cfg.search is not None:
        best = run_search(pipeline, X, dataset, cfg.search, out_dir)
        if cfg.search.apply_best:
            pipeline.set_params(**best)

    results = SplitResults(dataset, out_dir)
    reseed = cfg.evaluation.strategy == "group_shuffle_split"
    for split in iter_splits(cfg.evaluation, dataset.y, dataset.groups, cfg.seed):
        estimator = clone(pipeline)
        if reseed:
            set_random_state(estimator, split.seed)
        estimator.fit(X[split.train], dataset.y[split.train])
        results.add(split.name, split.test, estimator.predict(X[split.test]))
    summary = results.finalise()

    if cfg.evaluation.refit_full:
        final = clone(pipeline).fit(X, dataset.y)
        joblib.dump(final, out_dir / "model.joblib")
        log.info("Saved pipeline refitted on all %d samples to %s", len(X), out_dir / "model.joblib")
    return summary
