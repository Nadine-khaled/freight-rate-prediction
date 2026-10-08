# Freight Rate Prediction

A forecasting pipeline for predicting freight rates from operational load attributes. The project uses a time-based validation setup to avoid leakage from same-day market conditions and fits a gradient-boosted regressor on log-transformed rates.

## What this project does

- Cleans and transforms raw freight data
- Excludes obvious corrupted target values during training
- Engineers lane and market features
- Validates on rolling historical time windows
- Fits the final model on all labeled data
- Produces validation predictions and a fixed December forecast chart
- Builds a report PDF summarizing the modeling approach and results

## Project structure

- `features.py` – data cleaning, feature engineering, and missing-value handling
- `train.py` – training pipeline, validation splits, final fit, and output generation
- `score.py` – validates prediction files and generates the December chart
- `make_report.py` – creates `report.pdf` from validation outputs
- `cv.py` – comparison notebook/script utilities for cleaning/model variants
- `tune.py` – hyperparameter tuning experiments
- `requirements.txt` – Python dependencies

## Setup

```bash
python -m pip install -r requirements.txt
```

## Required data files

Place the following files in a `data/` folder before running the pipeline:

- `train_test.csv`
- `validation.csv`
- `validation_predictions_template.csv`
- `december_chart_inputs.csv`

These are expected to be in the repository root under `data/`.

## Run the full workflow

```bash
python train.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
python make_report.py
```

## What each step produces

### `python train.py`

Runs:

- rolling time-based cross-validation over historical windows
- final model fit on all labeled training data
- writes:
  - `validation_predictions.csv`
  - `december_predictions.csv`
  - `outputs/cv_results.csv`

### `python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv`

Validates:

- final validation predictions format and IDs
- fixed-December output structure and date coverage
- generates a chart at:
  - `scorer_results/candidate_december.png`

### `python make_report.py`

Builds a PDF report with model narrative and validation results:

- `report.pdf`

## Modeling approach

The repository uses a time-aware validation design instead of random splits to avoid leakage from same-day market conditions.

The model pipeline includes:

- log-transform of target `posted_rate`
- robust detection of corrupted target rows
- median and date-based missing-value fills
- feature engineering for distance, weight, equipment, market index, and lane information
- `HistGradientBoostingRegressor` with absolute error loss
- rolling validation windows aligned to the real forecasting horizon

## Notes

- Training excludes clearly corrupted labels, but evaluation still reflects the real-world target distribution.
- Validation is intentionally performed on future periods rather than random rows so results reflect the actual forecasting task.
- The December forecast uses a fixed route and fixed load attributes while varying only the date, matching the competition specification.

## Dependencies

```text
numpy
pandas
scikit-learn>=1.4
matplotlib
reportlab
```

## Typical output files

```text
outputs/
  cv_results.csv
scorer_results/
  candidate_december.png
validation_predictions.csv
december_predictions.csv
report.pdf
```
