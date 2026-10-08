"""Freight rate model: cleaning -> features -> HistGradientBoosting (log-rate, L1 loss).
Run: python train.py   (writes validation_predictions.csv, data/december_chart_inputs_filled.csv, outputs/*)"""
import numpy as np, pandas as pd, json, os
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import HuberRegressor, Ridge
from features import build_features, NUM

CAT = ["pickup", "delivery"]
PARAMS = dict(loss="absolute_error", max_iter=800, learning_rate=0.03, max_leaf_nodes=63,
              min_samples_leaf=20, random_state=0)
os.makedirs("outputs", exist_ok=True)

def load(p):
    d = pd.read_csv(p); d["date"] = pd.to_datetime(d["date"]); return d

def corrupt_mask(df, k=0.25):
    """True for rows whose log-rate is >k from a robust (Huber) distance/equipment/market fit.
    Clean noise is ~0.045 log units; corrupted rows are off by >=0.8 (x0.2-0.45 or x2.2-5.4)."""
    X = np.c_[df.log_dist, df.log_dist**2, df.market_index, pd.get_dummies(df.equipment, drop_first=True).values]
    r = df.y - HuberRegressor(max_iter=500).fit(X, df.y).predict(X)
    return r.abs() >= k

def with_cats(X, ref):
    X = X.copy()
    for c in CAT:
        X[c] = X[c].astype(pd.CategoricalDtype(sorted(ref[c].unique())))
    return X

def fit(tr):
    """tr: featurised training frame with y. Drops corrupted-rate rows, fits the GBM."""
    keep = ~corrupt_mask(tr)
    X = with_cats(tr.loc[keep, NUM + CAT], tr)
    return HGB(categorical_features="from_dtype", **PARAMS).fit(X, tr.y[keep]), keep

def predict(model, tr, df):
    return np.exp(model.predict(with_cats(df[NUM + CAT], tr)))

def prep(tr, others):
    wf = tr.weight.abs().median()
    mkt = pd.concat([tr] + others).dropna(subset=["market_index"]).groupby("date").market_index.mean()
    out = [build_features(d, wf, mkt) for d in [tr] + others]
    return out

if __name__ == "__main__":
    dev = load("data/train_test.csv"); dev["y"] = np.log(dev.posted_rate)
    val = load("data/validation.csv")
    # ---- time-based validation: train on the past, test on the next two months (mirrors Nov-Dec task)
    rows = []
    for s, e in [("2025-05-01", "2025-06-30"), ("2025-07-01", "2025-08-31"), ("2025-09-01", "2025-10-31")]:
        tr, te = dev[dev.date < s].copy(), dev[(dev.date >= s) & (dev.date <= e)].copy()
        tr, te = prep(tr, [te]); m, _ = fit(tr); p = predict(m, tr, te)
        err = np.abs(p - te.posted_rate); clean = ~corrupt_mask(te)
        rows.append(dict(train_through=str((pd.Timestamp(s) - pd.Timedelta(days=1)).date()), test=f"{s}..{e}",
                         n_test=len(te), MAE=err.mean(), MAPE=(err/te.posted_rate).mean()*100,
                         MAE_clean=err[clean].mean(), MAPE_clean=(err/te.posted_rate)[clean].mean()*100,
                         R2_log=1-((np.log(p)-te.y)**2).sum()/((te.y-te.y.mean())**2).sum()))
    cv = pd.DataFrame(rows).round(3); cv.to_csv("outputs/cv_results.csv", index=False); print(cv.to_string(index=False))
    # ---- final model on all labelled data
    tr, va = prep(dev.copy(), [val.copy()]); model, keep = fit(tr)
    print("dropped corrupted training rows:", (~keep).sum())
    pred = predict(model, tr, va)
    sub = pd.read_csv("data/validation_predictions_template.csv")[["load_id"]]
    sub = sub.merge(pd.DataFrame({"load_id": val.load_id, "predicted_rate": pred.round(2)}), on="load_id", how="left")
    sub.to_csv("validation_predictions.csv", index=False)
    # ---- fixed December chart: only date varies; market_index = that date's validation-set average,
    #      quote_signal = training median (not supplied in the chart file)
    dec = pd.read_csv("data/december_chart_inputs.csv"); d = dec.copy()
    d["load_id"] = "DEC"; d["date"] = pd.to_datetime(d["date"])
    mk = val.groupby("date").market_index.mean()
    ref = tr
    for c in ["pickup_lat","pickup_lon","delivery_lat","delivery_lon"]:
        src, col = ("pickup","pickup") if c.startswith("pickup") else ("delivery","delivery")
        d[c] = d[col].map(dev.groupby(col)[c].first())
    d["market_index"] = d["date"].map(mk); d["quote_signal"] = dev.quote_signal.median()
    dd = build_features(d, dev.weight.abs().median(), mk)
    dec["predicted_rate"] = predict(model, tr, dd).round(2)
    dec.to_csv("december_predictions.csv", index=False)
    print(dec.predicted_rate.describe()); print("val pred mean/ median", pred.mean(), np.median(pred))
