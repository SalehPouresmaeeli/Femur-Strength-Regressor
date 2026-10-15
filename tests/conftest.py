"""Synthetic stand-in for the real data: small .vti images + an Excel table.

Each image carries its patient id in pixel (0, 0) and a rotation-dependent
value in pixel (0, 1), so tests can check that rows and images stay aligned.
A bright block scales with the target, so models have something to learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from femur_rx.config import DataConfig

IMAGE_SHAPE = (16, 20)  # (rows, cols)
ROT_Y = [-10.0, 0.0, 10.0]
ROT_Z = [0.0, 10.0]
PATIENTS = list(range(1, 12))
EXCLUDED = [11]


def target(patient: int, ry: float, rz: float) -> float:
    return 1000.0 + 150.0 * patient + 4.0 * ry - 3.0 * rz


def write_vti(path, image: np.ndarray) -> None:
    import vtk
    from vtk.util import numpy_support

    rows, cols = image.shape
    data = vtk.vtkImageData()
    data.SetDimensions(cols, rows, 1)  # VTK is x-fastest: x = columns
    data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(image.ravel().astype(np.float32), deep=True))
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.Write()


def make_image(rng, patient: int, ry: float, rz: float) -> np.ndarray:
    img = rng.gamma(4.0, 50.0, IMAGE_SHAPE).astype(np.float32) + 100
    img[:, -3:] = 0  # background
    img[4:10, 5:12] += target(patient, ry, rz) / 2
    img[0, 0] = patient
    img[0, 1] = 1000 + ry * 10 + rz
    return img


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("femur")
    image_dir = root / "images"
    image_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for pid in PATIENTS:
        for ry in ROT_Y:
            for rz in ROT_Z:
                rows.append({
                    "Patient_id": pid, "Rotation around x-axis": 0, "Rotation around y-axis": int(ry),
                    "Rotation around z-axis": int(rz), "Strength": target(pid, ry, rz),
                })
                name = f"Hip{pid:03d}_[0.0, {ry}, {rz}].vti"
                write_vti(image_dir / name, make_image(rng, pid, ry, rz))
    excel = root / "table.xlsx"
    # Shuffle so that alignment cannot rely on row order.
    pd.DataFrame(rows).sample(frac=1, random_state=1).to_excel(excel, index=False)
    return {"excel_path": str(excel), "image_dir": str(image_dir), "root": root}


@pytest.fixture
def data_config(synthetic_data, tmp_path) -> DataConfig:
    return DataConfig(
        excel_path=synthetic_data["excel_path"],
        image_dir=synthetic_data["image_dir"],
        excluded_patients=EXCLUDED,
        tabular_features=["Loading_x", "Loading_y", "Loading_z"],
        cache_dir=str(tmp_path / "cache"),
    )


@pytest.fixture
def dataset(data_config):
    from femur_rx.data import build_dataset

    return build_dataset(data_config)
