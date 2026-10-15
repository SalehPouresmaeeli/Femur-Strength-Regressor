# Femur Strength Regressor

[![tests](https://github.com/SalehPouresmaeeli/femur-strength-regressor/actions/workflows/tests.yml/badge.svg)](https://github.com/SalehPouresmaeeli/femur-strength-regressor/actions/workflows/tests.yml)

Machine-learning pipeline that predicts **femur strength** from CT images of the femur, combined with clinical and
load-case features.

It uses random forest regression with optional image denoising and texture features (LBP or wavelets). Each
experiment is a small YAML file and runs end to end with one command, saving metrics, figures and the trained model.

The methods were developed in 2024; this repository is a refactored, documented release of that work.

## Installation

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
pip install -e .        # installs the femur_rx package from src/ in editable mode
```

Editable mode means changes to the code take effect without reinstalling.

## Data

No data is included in this repository.

1. Put the clinical table and the `.vti` images in the [data/](data/README.md) folder. It is git-ignored except for
   its README, which describes the expected format.
2. Create your dataset settings from the template and set your file names there:
   ```bash
   copy configs\data.example.yaml configs\data.yaml      # Linux/macOS: cp configs/data.example.yaml configs/data.yaml
   ```
   `configs/data.yaml` is git-ignored, so your file names and excluded patients stay private.

## Usage

```bash
# run the example experiment
python -m femur_rx run configs/example.yaml

# change settings without editing files
python -m femur_rx run configs/example.yaml --set evaluation.n_splits=10

# list / run all of your own experiments, continuing after failures
python -m femur_rx list configs/experiments
python -m femur_rx run configs/experiments --keep-going

# noise analysis of the images
python -m femur_rx noise
```

Results are written to `outputs/<experiment>/<timestamp>/`.

## Pipeline

```
config ─► data ─► [hyper-parameter search] ─► evaluation ─► metrics, figures, model
           │
           ├─ table: target, add loading direction
           └─ images: joined to table rows by patient and load case
                      (the load case's image, the un-rotated image, or both)

[image(s) | features] ─► denoise* ─► LBP* or wavelet* ─► random forest
                                                      * = optional
```

- **Splits are grouped by patient**: a model is never tested on a patient it was trained on.
- **The hyper-parameter search uses only training patients**; its score on held-out patients is reported.
- **Each image is processed separately** (denoising, texture features), so one image never affects another.

## Experiments

An experiment is a YAML file. [configs/example.yaml](configs/example.yaml) is a commented template: copy it into
`configs/experiments/` and edit it. That folder is git-ignored, so your own experiments are never committed.

An experiment can inherit from another one and list only what changes:

```yaml
base: ../example.yaml
name: example_ten_folds
evaluation: {n_splits: 10}
```

All settings are documented in [src/femur_rx/config.py](src/femur_rx/config.py).

## Outputs

Each run folder contains:
- the settings used and a log
- metrics per split (MAE, RMSE, MSE, R², MAPE), with mean and standard deviation
- all test predictions
- figures: predicted vs actual, relative error, residuals, R² spread
- the trained model

## Project layout

```
├── configs/              templates (data.example.yaml, example.yaml); your data.yaml and experiments/ are git-ignored
├── data/                 input data (git-ignored)
├── src/
│   └── femur_rx/         the package
│       ├── config.py         reads the YAML configs
│       ├── cli.py            command line: run, list, noise
│       ├── data/             reads the images and the table, builds the dataset
│       ├── preprocessing/    denoising, LBP and wavelet texture features
│       ├── models/           random forest pipeline
│       ├── evaluation/       patient-grouped splits, metrics, figures
│       ├── experiments/      runs an experiment end to end and saves the results
│       └── analysis/         noise analysis of the images
└── tests/                tests on synthetic data
```

## Testing

```bash
python -m pytest
```

The tests use a small synthetic dataset, so no patient data is needed.

## Troubleshooting

- **`Config file not found: configs/data.yaml`**: create it from the template (see [Data](#data)).
- **`Excel table not found` / `Image folder not found`**: the data is not where `configs/data.yaml` says. Copy it
  into `data/`, fix the names in `configs/data.yaml`, or set the paths with `--set data.excel_path=...` and
  `--set data.image_dir=...`.
- **`PyWavelets is not installed`**: run `pip install PyWavelets`.

## How to cite

If you use this software in your work, please cite it. Use the "Cite this repository" button on the GitHub page,
or see [CITATION.cff](CITATION.cff).

## License

No license. All rights are reserved.

## Contact

- Questions and bug reports: please [open an issue](https://github.com/SalehPouresmaeeli/femur-strength-regressor/issues).
- Collaboration and industry pilots: connect with the developer on
  [LinkedIn](https://www.linkedin.com/in/saleh-pouresmaeeli).
