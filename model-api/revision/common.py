"""Shared, leakage-safe data/feature code for the revision experiments.
Everything that is *learned* (medians, category lists, airline cancel rate, route counts) is fit on TRAIN only."""
import numpy as np, pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "csv"
OUT = Path(__file__).resolve().parent / "outputs"
WEATHER = ["D1_tmin", "D1_tmax", "D1_tavg", "D1_prcp", "D1_snow", "D1_wspd", "D1_pres"]
CATS = ["AIRLINE_CODE", "ORIGIN", "DEST"]
TRAIN_END, VAL_END = pd.Timestamp("2022-07-01"), pd.Timestamp("2023-01-01")   # train < 2022-07-01 <= val < 2023-01-01 <= test


def load():
    df = pd.read_csv(D / "clean_flights.csv.gz", parse_dates=["FL_DATE"])
    df["SPLIT"] = np.select([df.FL_DATE < TRAIN_END, df.FL_DATE < VAL_END], ["train", "validation"], "test")
    return df


def _hm_to_min(s):
    s = s.fillna(0).astype(int)
    return (s // 100) * 60 + (s % 100)


def fit_stats(train, target="CANCELLED", prior_strength=200):
    p = train[target].mean()
    g = train.groupby("AIRLINE_CODE")[target].agg(["sum", "size"])
    return dict(
        cats={c: sorted(train[c].dropna().unique()) for c in CATS},
        airline_rate=((g["sum"] + prior_strength * p) / (g["size"] + prior_strength)).to_dict(),
        prior=p,
        route_count=train.groupby(["ORIGIN", "DEST"]).size().to_dict(),
        medians=None,
    )


def make_features(df, st, weather=True):
    X = pd.DataFrame(index=df.index)
    m = df.FL_DATE.dt.month
    X["MONTH_SIN"], X["MONTH_COS"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    X["DAY_OF_WEEK"] = df.FL_DATE.dt.dayofweek
    X["DAY_OF_YEAR_SIN"] = np.sin(2 * np.pi * df.FL_DATE.dt.dayofyear / 366)
    X["DAY_OF_YEAR_COS"] = np.cos(2 * np.pi * df.FL_DATE.dt.dayofyear / 366)
    X["CRS_DEP_MIN"], X["CRS_ARR_MIN"] = _hm_to_min(df.CRS_DEP_TIME), _hm_to_min(df.CRS_ARR_TIME)
    X["DISTANCE"] = df.DISTANCE
    X["AIRLINE_RATE_TRAIN"] = df.AIRLINE_CODE.map(st["airline_rate"]).fillna(st["prior"])
    rc = pd.Series(list(zip(df.ORIGIN, df.DEST)), index=df.index).map(st["route_count"]).fillna(0)
    X["ROUTE_COUNT_LOG_TRAIN"] = np.log1p(rc)
    if weather:
        for c in WEATHER: X[c] = df[c]
    num = list(X.columns)
    for c in CATS:   # integer codes w.r.t. TRAIN categories, -1 = unseen
        X[c + "_C"] = pd.Categorical(df[c], categories=st["cats"][c]).codes
    return X, num


def finalize(X, num, st, fit=False):
    """Median-impute numeric columns (train medians), turn codes into xgboost categoricals (-1 -> NaN)."""
    X = X.copy()
    if fit: st["medians"] = X[num].median()
    X[num] = X[num].fillna(st["medians"])
    for c in CATS:
        X[c + "_C"] = pd.Categorical.from_codes(X[c + "_C"].values, categories=list(range(len(st["cats"][c]))))
    return X
