"""Regression model pipelines (sklearn).

Every pipeline consumes the ``[pixels..., tabular...]`` matrix produced by
:meth:`FemurDataset.to_matrix` and is built from optional steps::

    denoise -> lbp | wavelet -> reg

LBP and wavelet features both replace the pixels, so at most one of them is used.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from femur_rx.config import ModelConfig, PreprocessingConfig
from femur_rx.preprocessing import ImageDenoiser, LBPTransformer, WaveletTransformer

REGRESSORS = {
    "random_forest": (RandomForestRegressor, {"random_state": 0, "n_jobs": -1}),
}


def make_regressor(model_type: str, params: dict | None = None):
    try:
        cls, defaults = REGRESSORS[model_type]
    except KeyError:
        raise ValueError(f"Unknown regressor {model_type!r}; choose from {sorted(REGRESSORS)}") from None
    return cls(**{**defaults, **(params or {})})


def build_pipeline(model: ModelConfig, preprocessing: PreprocessingConfig, image_shape: tuple[int, ...]) -> Pipeline:
    if preprocessing.lbp is not None and preprocessing.wavelet is not None:
        raise ValueError("Use either preprocessing.lbp or preprocessing.wavelet, not both")
    steps = []
    if preprocessing.denoise is not None:
        steps.append(("denoise", ImageDenoiser(image_shape=image_shape, **preprocessing.denoise)))
    if preprocessing.lbp is not None:
        steps.append(("lbp", LBPTransformer(image_shape=image_shape, **preprocessing.lbp)))
    if preprocessing.wavelet is not None:
        steps.append(("wavelet", WaveletTransformer(image_shape=image_shape, **preprocessing.wavelet)))
    steps.append(("reg", make_regressor(model.type, model.params)))
    return Pipeline(steps)


def set_random_state(estimator, seed: int):
    """Set every ``random_state`` parameter of a (nested) estimator to ``seed``."""
    params = {k: seed for k in estimator.get_params(deep=True) if k == "random_state" or k.endswith("__random_state")}
    return estimator.set_params(**params)
