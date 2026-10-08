# Freight Rate Prediction

## Run
```bash
python -m pip install -r requirements.txt
# put train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv in data/
python train.py            # time-based CV, final fit, writes validation_predictions.csv + december_predictions.csv
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
python make_report.py      # builds report.pdf (uses outputs/cv_results.csv and scorer_results/candidate_december.png)
```

## Files
- `features.py` – cleaning + feature engineering
- `train.py` – corrupted-row filter, GBM, rolling time-based validation, final predictions
- `cv.py`, `tune.py` – model/cleaning comparison and small hyperparameter sweep
- `make_report.py` – report generator
