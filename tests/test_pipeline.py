"""End-to-end runs of the experiment runner on the synthetic dataset."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from femur_rx.cli import main
from femur_rx.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessingConfig,
    SearchConfig,
)
from femur_rx.data import build_dataset
from femur_rx.evaluation import compute_metrics, iter_splits, percent_error
from femur_rx.experiments import run_experiment
from femur_rx.models import build_pipeline


def _cfg(tmp_path, data_config: DataConfig, **kwargs) -> ExperimentConfig:
    return ExperimentConfig(name="test", output_dir=str(tmp_path / "out"), data=data_config, **kwargs)


def test_group_splits_never_share_patients(dataset):
    for strategy in ("group_kfold", "group_shuffle_split"):
        cfg = EvaluationConfig(strategy=strategy, n_splits=5, seeds=[1, 2, 2])
        splits = list(iter_splits(cfg, dataset.y, dataset.groups, 0))
        assert len({s.name for s in splits}) == len(splits)  # duplicate seeds get distinct names
        for s in splits:
            assert not set(dataset.groups[s.train]) & set(dataset.groups[s.test])


def test_metrics():
    y = np.array([100.0, 200.0, 400.0])
    perfect = compute_metrics(y, y)
    assert perfect["r2"] == 1.0 and perfect["mae"] == 0.0 and perfect["mape_percent"] == 0.0
    m = compute_metrics(y, y + [10.0, -20.0, 40.0])
    assert m["mae"] == pytest.approx(70 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(2100 / 3))
    assert m["mape_percent"] == pytest.approx(10.0)
    np.testing.assert_allclose(percent_error(y, [80.0, 220.0, 400.0]), [20.0, -10.0, 0.0])


def test_lbp_and_wavelet_together_are_rejected():
    preprocessing = PreprocessingConfig(lbp={"P": 8, "R": 1}, wavelet={"wavelet": "sym4", "level": 1})
    with pytest.raises(ValueError, match="not both"):
        build_pipeline(ModelConfig(), preprocessing, (16, 20))


def test_unknown_regressor():
    with pytest.raises(ValueError, match="Unknown regressor"):
        build_pipeline(ModelConfig(type="svm"), PreprocessingConfig(), (16, 20))


def test_random_forest_search_and_kfold(tmp_path, data_config, dataset):
    cfg = _cfg(
        tmp_path, data_config,
        model=ModelConfig(type="random_forest", params={"n_estimators": 20}),
        preprocessing=PreprocessingConfig(
            denoise={"method": "nl_means", "h_coef": 1.0, "patch_size": 3, "patch_distance": 4},
        ),
        search=SearchConfig(kind="grid", scoring="r2", cv_splits=3, n_jobs=1,
                            param_grid={"reg__max_depth": [2, None]}),
        evaluation=EvaluationConfig(strategy="group_kfold", n_splits=3),
    )
    run_dir = run_experiment(cfg, dataset)

    for name in ("config.yaml", "run.log", "dataset_summary.json", "metrics.json", "metrics_per_split.csv",
                 "predictions.csv", "predicted_vs_actual.png", "percent_error.png", "residuals_histogram.png",
                 "r2_boxplot.png", "model.joblib", "search/cv_results.csv", "search/best.json"):
        assert (run_dir / name).exists(), name

    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert len(predictions) == len(dataset)  # k-fold: every sample tested exactly once
    np.testing.assert_allclose(predictions.sort_values(["Patient_id", "Rotation around y-axis",
                                                        "Rotation around z-axis"])["y_true"], dataset.y)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert len(metrics["splits"]) == 3
    assert metrics["pooled"]["r2"] > 0.5  # the synthetic target is easy to learn

    model = joblib.load(run_dir / "model.joblib")
    assert [name for name, _ in model.steps] == ["denoise", "reg"]
    assert model.predict(dataset.to_matrix()[:5]).shape == (5,)


def test_wavelet_with_seeds(tmp_path, data_config, dataset):
    cfg = _cfg(
        tmp_path, data_config,
        model=ModelConfig(type="random_forest", params={"n_estimators": 20}),
        preprocessing=PreprocessingConfig(wavelet={"wavelet": "sym4", "level": 1, "mask": True}),
        search=SearchConfig(kind="random", n_iter=2, cv_splits=3, n_jobs=1,
                            param_grid={"wavelet__level": {"distribution": "randint", "low": 1, "high": 3}}),
        evaluation=EvaluationConfig(strategy="group_shuffle_split", seeds=[3, 4], test_size=0.3, refit_full=False),
    )
    run_dir = run_experiment(cfg, dataset)
    per_split = pd.read_csv(run_dir / "metrics_per_split.csv", index_col=0)
    assert list(per_split.index) == ["run_01_seed_3", "run_02_seed_4", "mean", "std"]
    assert {"mae", "rmse", "mse", "r2", "mape_percent"} <= set(per_split.columns)
    assert not (run_dir / "model.joblib").exists()


def test_both_image_mode_with_lbp(tmp_path, data_config):
    data_config.image_mode = "both"
    dataset = build_dataset(data_config)
    cfg = _cfg(
        tmp_path, data_config,
        model=ModelConfig(type="random_forest", params={"n_estimators": 10}),
        preprocessing=PreprocessingConfig(lbp={"P": 8, "R": 1, "method": "uniform"}),
        evaluation=EvaluationConfig(strategy="group_kfold", n_splits=2),
    )
    run_dir = run_experiment(cfg, dataset)
    model = joblib.load(run_dir / "model.joblib")
    assert model.named_steps["reg"].n_features_in_ == 2 * 10 + 3  # two LBP histograms + loading direction


def test_cli_run_with_overrides(tmp_path, synthetic_data):
    config = tmp_path / "exp.yaml"
    config.write_text(
        "name: cli_test\n"
        "model: {type: random_forest, params: {n_estimators: 10}}\n"
        "data: {excluded_patients: [11], tabular_features: [Loading_x]}\n"
        "evaluation: {strategy: group_kfold, n_splits: 2}\n"
    )
    code = main(["run", str(config),
                 "--set", f"data.excel_path={synthetic_data['excel_path']}",
                 "--set", f"data.image_dir={synthetic_data['image_dir']}",
                 "--set", f"data.cache_dir={tmp_path / 'cache'}",
                 "--set", f"output_dir={tmp_path / 'out'}"])
    assert code == 0
    assert len(list((tmp_path / "out" / "cli_test").glob("*/metrics.json"))) == 1


def test_cli_rejects_unknown_only_name(tmp_path):
    config = tmp_path / "exp.yaml"
    config.write_text("name: real_name\n")
    assert main(["run", str(config), "--only", "real_name", "typo_name"]) == 1


def test_run_dirs_never_collide(tmp_path):
    from femur_rx.experiments.runner import make_run_dir

    cfg = ExperimentConfig(name="same", output_dir=str(tmp_path))
    dirs = {make_run_dir(cfg) for _ in range(3)}
    assert len(dirs) == 3


def test_cli_reports_failure(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(f"name: bad\noutput_dir: {tmp_path.as_posix()}/out\ndata: {{image_dir: {tmp_path.as_posix()}/missing}}\n")
    assert main(["run", str(config)]) == 1
