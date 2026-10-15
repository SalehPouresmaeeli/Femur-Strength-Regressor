"""Wavelet texture features.

Each image is decomposed with :func:`pywt.wavedec2`; the features are the mean
and standard deviation of every detail sub-band (horizontal, vertical,
diagonal) at every level, so an image gives ``6 * level`` features.

There are no default settings: ``wavelet`` and ``level`` must be given, e.g. in
the ``preprocessing.wavelet`` section of a config::

    wavelet: {wavelet: ..., level: ..., mask: false}
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from femur_rx.preprocessing.layout import join_matrix, map_images, split_matrix


def wavelet_features(image: np.ndarray, wavelet: str, level: int, mask: bool = False) -> np.ndarray:
    """Mean and std of each detail sub-band, coarsest level first.

    With ``mask`` set, pixels <= 0 are set to zero before the decomposition
    (this only changes images with negative values).
    """
    import pywt

    image = np.asarray(image, dtype=np.float64)
    if mask:
        image = np.where(image > 0, image, 0.0)
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    features = []
    for details in coeffs[1:]:  # coeffs[0] is the approximation
        for band in details:
            features.extend((band.mean(), band.std()))
    return np.asarray(features)


class WaveletTransformer(BaseEstimator, TransformerMixin):
    """Replace each image of a ``[pixels..., tabular...]`` matrix by its wavelet features.

    Tabular columns pass through unchanged. ``wavelet`` and ``level`` have no
    defaults and must be set (they can be tuned with ``wavelet__wavelet`` etc.).
    """

    def __init__(self, image_shape=None, wavelet=None, level=None, mask=False):
        self.image_shape = image_shape
        self.wavelet = wavelet
        self.level = level
        self.mask = mask

    def _check_settings(self) -> None:
        missing = [name for name in ("wavelet", "level") if getattr(self, name) is None]
        if missing:
            raise ValueError(f"Wavelet features need setting(s) {missing} (e.g. in preprocessing.wavelet)")

    def fit(self, X, y=None):
        self._check_settings()  # fail early on missing settings
        return self

    def transform(self, X):
        self._check_settings()
        images, tabular = split_matrix(X, self.image_shape)
        features = map_images(lambda img: wavelet_features(img, self.wavelet, self.level, self.mask), images)
        return join_matrix(features, tabular)
