import numpy as np, pandas as pd

def haversine(a, b, c, d):
    a, b, c, d = map(np.radians, [a, b, c, d])
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(x))

def clean_features(df):
    """Fix data-quality issues in feature columns (applies to train and validation alike)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weight"] = df["weight"].abs()                      # sign errors (negative weights)
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["market_missing"] = df["market_index"].isna().astype(int)
    return df

def build_features(df, weight_fill, market_by_date):
    df = clean_features(df)
    df["weight"] = df["weight"].fillna(weight_fill)
    # market_index is a daily signal with tiny per-load noise -> fill with that date's mean
    day = df["date"].map(market_by_date)
    df["market_index"] = df["market_index"].fillna(day)
    df["market_day"] = day
    hv = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    df["log_dist"] = np.log(df["distance"])
    df["dist_ratio"] = df["distance"] / hv                 # >1.4 flags suspicious distances
    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["eq"] = df["equipment"].map({"Dry Van": 0, "Flatbed": 1, "Reefer": 2})
    df["log_w"] = np.log(df["weight"])
    return df

NUM = ["log_dist", "distance", "weight", "market_index", "quote_signal", "eq", "dow",
       "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
