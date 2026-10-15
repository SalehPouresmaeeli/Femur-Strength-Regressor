import numpy as np
import pandas as pd
import pytest

from conftest import IMAGE_SHAPE, target
from femur_rx.data import build_dataset, parse_vti_filename, read_vti
from femur_rx.data.vti import load_image_directory


def test_parse_vti_filename():
    assert parse_vti_filename("Hip016_[0.0, -10.0, 20.0].vti") == (16, (0.0, -10.0, 20.0))
    assert parse_vti_filename("Hip016_scan_v2_[0.0, -10.0, 20.0].vti") == (16, (0.0, -10.0, 20.0))


def test_read_vti_keeps_orientation(tmp_path):
    from conftest import write_vti

    image = np.arange(np.prod(IMAGE_SHAPE), dtype=np.float32).reshape(IMAGE_SHAPE)
    write_vti(tmp_path / "Hip001_x_[0.0, 0.0, 0.0].vti", image)
    np.testing.assert_array_equal(read_vti(tmp_path / "Hip001_x_[0.0, 0.0, 0.0].vti"), image)


def test_image_cache_roundtrip(synthetic_data, tmp_path):
    meta1, imgs1 = load_image_directory(synthetic_data["image_dir"], tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    meta2, imgs2 = load_image_directory(synthetic_data["image_dir"], tmp_path)
    pd.testing.assert_frame_equal(meta1, meta2)
    np.testing.assert_array_equal(imgs1, imgs2)


def test_build_dataset(dataset):
    # 10 kept patients x 6 rotations
    assert len(dataset) == 10 * 6
    assert dataset.image_shape == IMAGE_SHAPE
    assert 11 not in dataset.groups
    md = dataset.metadata
    # each image belongs to its row: patient id and rotation are encoded in the pixels
    np.testing.assert_array_equal(dataset.images[:, 0, 0], md["Patient_id"])
    np.testing.assert_allclose(
        dataset.images[:, 0, 1], 1000 + md["Rotation around y-axis"] * 10 + md["Rotation around z-axis"])
    expected = [target(p, ry, rz) for p, ry, rz in
                md[["Patient_id", "Rotation around y-axis", "Rotation around z-axis"]].itertuples(index=False)]
    np.testing.assert_allclose(dataset.y, expected)
    assert dataset.y.dtype == float


def test_rows_without_target_are_dropped(synthetic_data, data_config, tmp_path):
    table = pd.read_excel(synthetic_data["excel_path"])
    table.loc[table["Patient_id"] == 1, "Strength"] = np.nan
    data_config.excel_path = str(tmp_path / "table.xlsx")
    table.to_excel(data_config.excel_path, index=False)
    ds = build_dataset(data_config)
    assert 1 not in ds.groups
    assert len(ds) == 9 * 6


def test_summary(dataset):
    s = dataset.summary()
    assert s["n_samples"] == 60
    assert s["n_patients"] == 10
    assert s["target"]["column"] == "Strength"
    assert s["target"]["min"] == pytest.approx(dataset.y.min())
    assert s["target"]["mean"] == pytest.approx(dataset.y.mean())


def test_loading_direction(dataset):
    md = dataset.metadata
    neutral = md[(md["Rotation around y-axis"] == 0) & (md["Rotation around z-axis"] == 0)]
    np.testing.assert_allclose(neutral[["Loading_x", "Loading_y", "Loading_z"]], [[-1, 0, 0]] * len(neutral), atol=1e-12)
    norms = np.linalg.norm(dataset.tabular.to_numpy(), axis=1)
    np.testing.assert_allclose(norms, 1.0)


def test_to_matrix_layout(dataset):
    X = dataset.to_matrix()
    n_pixels = int(np.prod(IMAGE_SHAPE))
    assert X.shape == (len(dataset), n_pixels + 3)
    np.testing.assert_allclose(X[:, n_pixels:], dataset.tabular.to_numpy())


def test_fixed_image_mode(data_config):
    data_config.image_mode = "fixed"
    ds = build_dataset(data_config)
    # every row shows the neutral (ry = rz = 0) image of its own patient
    np.testing.assert_array_equal(ds.images[:, 0, 0], ds.metadata["Patient_id"])
    np.testing.assert_array_equal(ds.images[:, 0, 1], 1000)


def test_both_image_mode(data_config, dataset):
    data_config.image_mode = "both"
    ds = build_dataset(data_config)
    assert ds.image_shape == (2, *IMAGE_SHAPE)
    md = ds.metadata
    # first image: the neutral image of the patient; second image: the row's own rotated image
    np.testing.assert_array_equal(ds.images[:, 0, 0, 0], md["Patient_id"])
    np.testing.assert_array_equal(ds.images[:, 0, 0, 1], 1000)
    np.testing.assert_array_equal(ds.images[:, 1], dataset.images)
    # matrix layout: fixed pixels, then rotated pixels, then tabular features
    n_pixels = int(np.prod(IMAGE_SHAPE))
    X = ds.to_matrix()
    assert X.shape == (len(ds), 2 * n_pixels + 3)
    np.testing.assert_array_equal(X[:, n_pixels:2 * n_pixels], dataset.images.reshape(len(ds), -1))


def test_unknown_image_mode(data_config):
    data_config.image_mode = "double"
    with pytest.raises(ValueError, match="image_mode"):
        build_dataset(data_config)


def test_missing_inputs_give_helpful_error(data_config, tmp_path):
    data_config.excel_path = str(tmp_path / "missing.xlsx")
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        build_dataset(data_config)


def test_unknown_tabular_feature(data_config):
    data_config.tabular_features = ["Nope"]
    with pytest.raises(KeyError):
        build_dataset(data_config)
