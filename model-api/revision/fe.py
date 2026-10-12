"""Feature groups + task definitions for the v2 experiments (reads csv/features_v2.pkl).
No label-derived statistic is learned from the data here (history features use strictly earlier days, see 12_features_v2.py)."""
import numpy as np, pandas as pd
import common as C

CAL = ["MONTH_SIN", "MONTH_COS", "DAY_OF_WEEK", "DOY_SIN", "DOY_COS"]
HOL = ["DAYS_TO_HOLIDAY", "DAYS_SINCE_HOLIDAY"]
SCHED = ["CRS_DEP_MIN", "CRS_ARR_MIN", "DISTANCE", "ROUTE_COUNT_LOG_TRAIN"]
WX_O = C.WEATHER
WX_D = ["DEST_D1_tmin", "DEST_D1_tmax", "DEST_D1_prcp", "DEST_D1_snow", "DEST_D1_wspd"]
HIST = [f"H_{l}_{w}_{m}" for l in ["NAT", "ORG", "AIR"] for w in ["D1", "D7"] for m in ["CANCEL", "DELAY15", "MEANDELAY"]]
CATS = C.CATS
SETS = {
    "A_v1 (cal+sched+origin D-1 weather)": CAL + SCHED + WX_O,
    "B_A+history": CAL + SCHED + WX_O + HIST,
    "C_full (B+holiday+dest weather)": CAL + HOL + SCHED + WX_O + WX_D + HIST,
    "D_full_no_weather": CAL + HOL + SCHED + HIST,
    "E_full_no_history": CAL + HOL + SCHED + WX_O + WX_D,
    "F_cal+sched_only": CAL + HOL + SCHED,
}
FULL = "C_full (B+holiday+dest weather)"
LIVE = [f"L_{l}_{m}" for l in ["ORG", "AIR", "NAT"] for m in ["DELAY15", "MEANDELAY", "CANCEL", "BACKLOG"]]
FULL_B = "G_full+live (day-of-departure, T-3h)"
SETS_B = {
    FULL: SETS[FULL],
    FULL_B: SETS[FULL] + LIVE,
    "H_full+live_no_weather": CAL + HOL + SCHED + HIST + LIVE,
    "I_cal+sched+live_only": CAL + HOL + SCHED + LIVE,
}


def load_v2(mode="A"):
    df = pd.read_pickle(C.D / ("features_v3.pkl" if mode == "B" else "features_v2.pkl"))
    m = df.FL_DATE.dt.month
    doy = df.FL_DATE.dt.dayofyear
    df["MONTH_SIN"], df["MONTH_COS"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    df["DAY_OF_WEEK"] = df.FL_DATE.dt.dayofweek
    df["DOY_SIN"], df["DOY_COS"] = np.sin(2 * np.pi * doy / 366), np.cos(2 * np.pi * doy / 366)
    df["CRS_DEP_MIN"], df["CRS_ARR_MIN"] = C._hm_to_min(df.CRS_DEP_TIME), C._hm_to_min(df.CRS_ARR_TIME)
    rc = df[df.SPLIT == "train"].groupby(["ORIGIN", "DEST"]).size()   # label-free, train only
    key = pd.MultiIndex.from_frame(df[["ORIGIN", "DEST"]])
    df["ROUTE_COUNT_LOG_TRAIN"] = np.log1p(rc.reindex(key).fillna(0).values)
    return df


def task_frame(df, task):
    if task == "cancel":
        d = df.copy()
        d["y"] = d.CANCELLED.astype(int)
        return d, ["no", "yes"]
    if task == "code":   # cancelled flights, DOT codes A-C (D excluded: 2020-COVID artefact; N never existed)
        d = df[(df.CANCELLED == 1) & (df.CANCELLATION_CODE.isin(list("ABC")))].copy()
        d["y"] = d.CANCELLATION_CODE.map({"A": 0, "B": 1, "C": 2}).astype(int)
        return d, ["A carrier", "B weather", "C NAS"]
    if task == "delay":  # not cancelled, not diverted; DEP_DELAY in minutes
        d = df[(df.CANCELLED == 0) & (df.DIVERTED == 0)].copy()
        d["y"] = pd.cut(d.DEP_DELAY, [-np.inf, 0, 15, 30, np.inf], labels=False).astype(int)
        return d, ["<=0 min", "1-15 min", "16-30 min", ">30 min"]
    raise ValueError(task)


def matrix(d, cols, cats_from_train):
    X = d[cols].copy()
    for c in CATS:
        X[c] = pd.Categorical(d[c], categories=cats_from_train[c])
    return X


def train_cats(df):
    tr = df[df.SPLIT == "train"]
    return {c: sorted(tr[c].dropna().unique()) for c in CATS}
