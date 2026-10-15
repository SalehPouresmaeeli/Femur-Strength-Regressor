import numpy as np
import pytest

from femur_rx.preprocessing import (
    ImageDenoiser,
    LBPTransformer,
    WaveletTransformer,
    denoise_image,
    lbp_histogram,
    lbp_n_bins,
    wavelet_features,
)

SHAPE = (16, 20)

# Arbitrary test settings for each denoising method.
SETTINGS = {
    "median": {"disk_radius": 1},
    "wiener": {"window": 3},
    "nl_means": {"h_coef": 1.0, "patch_size": 3, "patch_distance": 4},
}


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    return rng.gamma(4.0, 50.0, (6, *SHAPE)).astype(np.float32)


@pytest.mark.parametrize("method, P", [("uniform", 8), ("nri_uniform", 8), ("uniform", 16), ("default", 8)])
def test_lbp_histogram_has_fixed_length(images, method, P):
    hists = [lbp_histogram(img, P, 1, method) for img in images]
    assert {len(h) for h in hists} == {lbp_n_bins(P, method)}
    for h in hists:
        assert h.sum() == pytest.approx(1.0)


def test_lbp_transformer_keeps_tabular(images):
    tab = np.arange(12, dtype=float).reshape(6, 2)
    X = np.hstack([images.reshape(6, -1), tab])
    out = LBPTransformer(image_shape=SHAPE, P=8, R=1).fit_transform(X)
    assert out.shape == (6, lbp_n_bins(8, "uniform") + 2)
    np.testing.assert_array_equal(out[:, -2:], tab)


def test_lbp_constant_image_does_not_crash():
    assert np.isfinite(lbp_histogram(np.ones(SHAPE))).all()


@pytest.mark.parametrize("method", ["median", "wiener", "nl_means"])
def test_denoiser_treats_images_independently(images, method):
    """Each image is filtered on its own; the result must not depend on the other images in the batch."""
    X = images.reshape(len(images), -1)
    stacked = ImageDenoiser(image_shape=SHAPE, method=method, **SETTINGS[method]).fit_transform(X)
    single = ImageDenoiser(image_shape=SHAPE, method=method, **SETTINGS[method]).fit_transform(X[:1])
    np.testing.assert_allclose(stacked[:1], single, rtol=1e-5)
    np.testing.assert_allclose(stacked[0], denoise_image(images[0], method, **SETTINGS[method]).ravel(), rtol=1e-5)
    assert np.isfinite(stacked).all()


def test_denoiser_passes_tabular_through(images):
    tab = np.array([[1.0, -1.0]] * 6)
    X = np.hstack([images.reshape(6, -1), tab])
    out = ImageDenoiser(image_shape=SHAPE, method="median", **SETTINGS["median"]).fit_transform(X)
    np.testing.assert_array_equal(out[:, -2:], tab)


def test_denoiser_parallel_matches_serial(images):
    X = images.reshape(len(images), -1)
    serial = ImageDenoiser(image_shape=SHAPE, method="nl_means", **SETTINGS["nl_means"]).fit_transform(X)
    parallel = ImageDenoiser(image_shape=SHAPE, method="nl_means", n_jobs=2, **SETTINGS["nl_means"]).fit_transform(X)
    np.testing.assert_allclose(serial, parallel)


def test_unknown_denoiser():
    with pytest.raises(ValueError, match="Unknown denoising method"):
        denoise_image(np.ones(SHAPE), "gaussian_blur")


@pytest.mark.parametrize("method, missing", [("median", "disk_radius"), ("wiener", "window"),
                                             ("nl_means", "patch_distance")])
def test_denoiser_settings_are_required(images, method, missing):
    """There are no default filter settings; a missing one must be named in the error."""
    settings = {k: v for k, v in SETTINGS[method].items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        denoise_image(images[0], method, **settings)
    with pytest.raises(ValueError, match=missing):
        ImageDenoiser(image_shape=SHAPE, method=method, **settings).fit(images.reshape(len(images), -1))


def test_denoiser_needs_method_and_image_shape(images):
    X = images.reshape(len(images), -1)
    with pytest.raises(ValueError, match="method"):
        ImageDenoiser(image_shape=SHAPE).fit(X)
    with pytest.raises(ValueError, match="image_shape"):
        ImageDenoiser(method="median", **SETTINGS["median"]).fit_transform(X)
    with pytest.raises(ValueError, match="image_shape"):
        LBPTransformer().fit_transform(X)
    with pytest.raises(ValueError, match="image_shape"):
        WaveletTransformer(wavelet="sym4", level=1).fit_transform(X)


# Arbitrary test settings for the wavelet features.
WAVELET = {"wavelet": "sym4", "level": 1}


@pytest.mark.parametrize("level", [1, 2])
def test_wavelet_feature_count(images, level):
    # mean and std of three detail sub-bands per level
    assert wavelet_features(images[0], "sym4", level).shape == (6 * level,)


def test_wavelet_transformer_keeps_tabular(images):
    tab = np.arange(12, dtype=float).reshape(6, 2)
    X = np.hstack([images.reshape(6, -1), tab])
    out = WaveletTransformer(image_shape=SHAPE, **WAVELET).fit_transform(X)
    assert out.shape == (6, 6 + 2)
    np.testing.assert_array_equal(out[:, -2:], tab)
    np.testing.assert_allclose(out[0, :-2], wavelet_features(images[0], **WAVELET))


def test_wavelet_transformer_without_tabular_columns(images):
    out = WaveletTransformer(image_shape=SHAPE, **WAVELET).fit_transform(images.reshape(6, -1))
    assert out.shape == (6, 6)


def test_wavelet_mask_only_changes_negative_pixels(images):
    np.testing.assert_allclose(wavelet_features(images[0], **WAVELET, mask=True),
                               wavelet_features(images[0], **WAVELET))
    with_negatives = images[0] - images[0].mean()
    assert not np.allclose(wavelet_features(with_negatives, **WAVELET, mask=True),
                           wavelet_features(with_negatives, **WAVELET))


@pytest.mark.parametrize("missing", ["wavelet", "level"])
def test_wavelet_settings_are_required(images, missing):
    settings = {k: v for k, v in WAVELET.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        WaveletTransformer(image_shape=SHAPE, **settings).fit(images.reshape(len(images), -1))


def test_two_images_per_sample_are_processed_separately(images):
    """With image_shape (2, rows, cols) each transformer handles the two images independently."""
    pairs = images.reshape(3, 2, *SHAPE)
    tab = np.ones((3, 1))
    X = np.hstack([pairs.reshape(3, -1), tab])
    shape = (2, *SHAPE)

    lbp = LBPTransformer(image_shape=shape).fit_transform(X)
    n_bins = lbp_n_bins(8, "uniform")
    assert lbp.shape == (3, 2 * n_bins + 1)
    np.testing.assert_allclose(lbp[1, n_bins:2 * n_bins], lbp_histogram(pairs[1, 1]))

    wl = WaveletTransformer(image_shape=shape, **WAVELET).fit_transform(X)
    assert wl.shape == (3, 2 * 6 + 1)
    np.testing.assert_allclose(wl[2, :6], wavelet_features(pairs[2, 0], **WAVELET))

    denoised = ImageDenoiser(image_shape=shape, method="median", **SETTINGS["median"]).fit_transform(X)
    assert denoised.shape == X.shape
    expected = denoise_image(pairs[0, 1], "median", **SETTINGS["median"]).ravel()
    np.testing.assert_allclose(denoised[0, SHAPE[0] * SHAPE[1]:-1], expected)
    parallel = ImageDenoiser(image_shape=shape, method="median", n_jobs=2, **SETTINGS["median"]).fit_transform(X)
    np.testing.assert_allclose(parallel, denoised)
