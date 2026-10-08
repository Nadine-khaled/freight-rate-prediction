import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
S = getSampleStyleSheet(); B = S["BodyText"]; H = S["Heading2"]
cv = pd.read_csv("outputs/cv_results.csv")
doc = SimpleDocTemplate("report.pdf", pagesize=letter, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
P = lambda t, s=B: Paragraph(t, s)
el = [P("Freight Rate Prediction - Validation &amp; Split Approach", S["Title"]),
 P("1. Data split / validation design", H),
 P("The labelled file (train_test.csv, 48,000 loads) covers 2025-01-01 to 2025-10-31, while the 12,000 loads to predict "
   "are dated 2025-11-01 to 2025-12-31. The task is therefore a <b>forecast into the future</b>, so a random split would leak "
   "same-day market conditions into validation and overstate accuracy. I used a <b>rolling-origin (expanding window) time split</b>: "
   "train on all data before a cut-off and test on the following two months, the same horizon as the real task. Three folds were used "
   "(test May-Jun, Jul-Aug, Sep-Oct). Tuning decisions (cleaning strategy, loss, depth, lane features) were made on these folds only; "
   "validation.csv was never used for fitting. The final model is refit on all 48,000 rows."),
 P("2. Data-quality findings", H),
 P("&bull; <b>Corrupted target values (~1.4%, 677 rows):</b> rates are 0.2-0.45x or 2.2-5.4x what distance/equipment/market imply (clean noise is about 4-5% in log terms), "
   "spread evenly across months, equipment and quote_signal, so they look like random entry errors. They are detected with a robust (Huber) fit "
   "and excluded from training only (never from test data).<br/>"
   "&bull; <b>Negative weights (292 rows):</b> sign errors (|weight| is within the normal 5,000-47,500 lb range); fixed with abs().<br/>"
   "&bull; <b>Missing values:</b> weight 300 train / 165 validation (median fill + missing flag logic); market_index 374 / 249 - it is a daily market "
   "signal with tiny per-load noise (std 0.025), so missing values are filled with that date's mean.<br/>"
   "&bull; <b>Distances / coordinates:</b> distance is consistent within each lane (within about 2%), so no repairs were needed. City coordinates are one fixed pair per city.<br/>"
   "&bull; <b>Temporal drift:</b> market_index averages 0.92-0.93 in Nov-Dec versus 1.08 overall in training, so the model leans on market_index rather than "
   "calendar month (Nov/Dec are never seen in training and trees cannot extrapolate month effects)."),
 P("3. Model", H),
 P("HistGradientBoostingRegressor on log(rate) with L1 loss (robust to the remaining outliers), numeric features: log distance, distance, weight, equipment, "
   "market_index, quote_signal, day of week, coordinates, plus pickup and delivery as categorical features. Gradient boosting reduced MAPE on the clean "
   "holdout subset from about 3.9% (ridge on log features) to about 3.2-3.3%, and the naive baseline is far worse (about 64%). Filtering corrupted training "
   "rows and adding lane categories each gave small, consistent gains."),
 P("4. Time-based validation results (final configuration)", H)]
t = Table([list(cv.columns)] + cv.values.tolist(), repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#064A56")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                       ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
el += [t, Spacer(1, 6),
 P("MAE and MAPE are measured against the raw posted_rate including corrupted rows (unavoidable error), while the *_clean columns exclude rows flagged as corrupted "
   "so they reflect pricing accuracy. Error rises slightly in later folds, consistent with more drift as the forecast moves away from the training window."),
 P("5. Fixed December prediction chart", H),
 P("Chart inputs contain only the load attributes and date, so market_index for each December day is set to the average market_index of that day's loads in validation.csv "
   "(a feature, not a label), and quote_signal is set to the training median. The weekly rhythm in the chart comes from the weekly cycle in market_index."),
 Image("scorer_results/candidate_december.png", width=500, height=222)]
doc.build(el)
