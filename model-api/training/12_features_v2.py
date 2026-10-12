"""Feature table v2 = clean_flights + (a) calendar/holiday, (b) destination D-1 weather, (c) HISTORY features that use ONLY days strictly before the flight date:
 national / origin-airport / airline rates of cancellation and of >15-min departure delay on D-1 and over D-7..D-1 (shrunk towards the global prior).
All of this is information that exists at prediction time (completed previous days). A self-test proves a flight's own day never enters its features.
Output: csv/features_v2.pkl"""
import sys, numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as C

K = 30   # shrinkage strength (pseudo-flights) towards the prior


def history_features(df, prior_c, prior_d):
    """prior_c/prior_d are FIXED constants from the TRAIN period (never from the data being featurised).
    df needs FL_DATE, ORIGIN, AIRLINE_CODE, CANCELLED, DIVERTED, DEP_DELAY. Returns DataFrame (index = df.index) of history columns."""
    ok = (df.CANCELLED == 0) & (df.DIVERTED == 0)
    base = pd.DataFrame({"FL_DATE": df.FL_DATE, "ORIGIN": df.ORIGIN, "AIRLINE_CODE": df.AIRLINE_CODE, "n": 1, "c": df.CANCELLED,
                         "m": ok.astype(int), "d": (ok & (df.DEP_DELAY > 15)).astype(int), "z": np.where(ok, df.DEP_DELAY.clip(-30, 180), 0.0)})
    days = pd.date_range(df.FL_DATE.min() - pd.Timedelta(days=1), df.FL_DATE.max())
    out = pd.DataFrame(index=df.index)
    for tag, key in [("NAT", None), ("ORG", "ORIGIN"), ("AIR", "AIRLINE_CODE")]:
        keys = ["FL_DATE"] if key is None else [key, "FL_DATE"]
        g = base.groupby(keys)[["n", "c", "m", "d", "z"]].sum()
        if key is None:
            g = g.reindex(days, fill_value=0); g.index.name = "FL_DATE"
            roll = g.rolling(7, min_periods=1).sum().shift(1); lag = g.shift(1)
        else:
            idx = pd.MultiIndex.from_product([g.index.levels[0], days], names=keys)
            g = g.reindex(idx, fill_value=0)
            roll = g.groupby(level=0).transform(lambda x: x.rolling(7, min_periods=1).sum().shift(1))
            lag = g.groupby(level=0).shift(1)
        feats = pd.DataFrame(index=g.index)
        for nm, src in [("D1", lag), ("D7", roll)]:
            src = src.fillna(0)
            feats[f"H_{tag}_{nm}_CANCEL"] = (src.c + K * prior_c) / (src.n + K)
            feats[f"H_{tag}_{nm}_DELAY15"] = (src.d + K * prior_d) / (src.m + K)
            feats[f"H_{tag}_{nm}_MEANDELAY"] = src.z / src.m.clip(lower=1)
            feats[f"H_{tag}_{nm}_LOGN"] = np.log1p(src.n)
        feats = feats.drop(columns=[c for c in feats if c.endswith("_LOGN") and tag == "NAT"])  # counts differ by source density -> keep only for ORG/AIR? no: drop all counts below
        feats = feats[[c for c in feats if not c.endswith("_LOGN")]]
        if key is None: j = base[["FL_DATE"]].join(feats, on="FL_DATE")
        else: j = base[[key, "FL_DATE"]].join(feats, on=[key, "FL_DATE"])
        out = out.join(j[feats.columns])
    return out


def selftest(df):
    """Flip all labels/delays of ONE day; features of flights on that very day must be unchanged, the next day's must change."""
    d0 = df.FL_DATE.drop_duplicates().sort_values().iloc[400]
    sub = df[df.FL_DATE.between(d0 - pd.Timedelta(days=10), d0 + pd.Timedelta(days=2))].copy()
    a = history_features(sub, PC, PD)
    sub2 = sub.copy(); m = sub2.FL_DATE == d0
    sub2.loc[m, "CANCELLED"] = 1 - sub2.loc[m, "CANCELLED"]; sub2.loc[m, "DEP_DELAY"] += 500
    b = history_features(sub2, PC, PD)
    same_day = np.allclose(a[m].fillna(-1).values, b[m].fillna(-1).values)
    nxt = sub.FL_DATE == d0 + pd.Timedelta(days=1)
    changed_next = not np.allclose(a[nxt].fillna(-1).values, b[nxt].fillna(-1).values)
    assert same_day and changed_next, f"history leakage test failed: same_day_unchanged={same_day} next_day_changed={changed_next}"
    print(f"SELF-TEST OK: day {d0.date()} labels flipped -> its own features unchanged ({same_day}), next day's features reflect it ({changed_next})", flush=True)


df = C.load()
_tr = df[df.SPLIT == "train"]; _ok = (_tr.CANCELLED == 0) & (_tr.DIVERTED == 0)
PC, PD = float(_tr.CANCELLED.mean()), float((_ok & (_tr.DEP_DELAY > 15)).sum() / _ok.sum())
print(f"train-period priors: cancel={PC:.4f} delay15={PD:.4f}", flush=True)
selftest(df)
H = history_features(df, PC, PD)
df = df.join(H)
# calendar / holidays (known in advance)
hol = USFederalHolidayCalendar().holidays("2018-12-01", "2024-03-01").values
d = df.FL_DATE.values
pos = np.searchsorted(hol, d)
nxt = hol[np.clip(pos, 0, len(hol) - 1)]; prv = hol[np.clip(pos - 1, 0, len(hol) - 1)]
df["DAYS_TO_HOLIDAY"] = np.minimum(((nxt - d) / np.timedelta64(1, "D")), 14)
df["DAYS_SINCE_HOLIDAY"] = np.minimum(((d - prv) / np.timedelta64(1, "D")), 14)
# destination D-1 weather (observed previous day at destination airport)
w = pd.read_csv(C.D / "weather_daily_by_airport.csv.gz", parse_dates=["WEATHER_DATE"])
w["FL_DATE"] = w.WEATHER_DATE + pd.Timedelta(days=1)
wd = w[["ORIGIN", "FL_DATE", "tmin", "tmax", "prcp", "snow", "wspd"]].rename(columns={"ORIGIN": "DEST", **{c: f"DEST_D1_{c}" for c in ["tmin", "tmax", "prcp", "snow", "wspd"]}})
n0 = len(df); df = df.merge(wd, on=["DEST", "FL_DATE"], how="left", validate="many_to_one"); assert len(df) == n0
print("history columns:", [c for c in H.columns], flush=True)
print("dest weather non-missing:", df.DEST_D1_tmax.notna().mean().round(4), flush=True)
df.to_pickle(C.D / "features_v2.pkl")
print("saved", df.shape, flush=True)
