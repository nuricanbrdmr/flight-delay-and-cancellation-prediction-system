"""Roadmap steps 3+4: chronological split, per-task target/class distributions, feature & leakage control table."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "csv"
OUT = Path(__file__).resolve().parent / "outputs"
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
df = pd.read_csv(D / "clean_flights.csv.gz", parse_dates=["FL_DATE"], usecols=["FL_DATE","CANCELLED","CANCELLATION_CODE","DIVERTED","DEP_DELAY","SOURCE"])

def split_of(d):
    return np.select([d < "2022-07-01", d < "2023-01-01"], ["train", "validation"], "test")
df["SPLIT"] = split_of(df.FL_DATE)
span = df.groupby("SPLIT").FL_DATE.agg(["min", "max", "size"]).loc[["train", "validation", "test"]]
print(span, "\n")

rows = []
def add(task, sub, ycol):
    for sp in ["train", "validation", "test"]:
        s = sub[sub.SPLIT == sp][ycol]
        for cls, n in s.value_counts().sort_index().items():
            rows.append(dict(task=task, split=sp, cls=cls, n=int(n), share=round(n / len(s), 4), split_total=len(s)))
# 1) cancellation: all flights
add("cancel (all flights)", df.assign(y=df.CANCELLED), "y")
# 2) cancellation cause: cancelled flights only, A-D (no N)
c = df[df.CANCELLED == 1]; add("cancel_code (cancelled only, A-D)", c, "CANCELLATION_CODE")
# 3) delay: not cancelled, not diverted; DEP_DELAY minutes -> classes
d = df[(df.CANCELLED == 0) & (df.DIVERTED == 0)].copy()
print("delay task: DEP_DELAY missing =", int(d.DEP_DELAY.isna().sum()), "of", len(d))
d["DELAY_CLASS"] = pd.cut(d.DEP_DELAY, [-np.inf, 0, 15, 30, np.inf], labels=["0: <=0 min", "1: 1-15", "2: 16-30", "3: >30"])
add("delay (not cancelled, not diverted)", d, "DELAY_CLASS")
t = pd.DataFrame(rows); t.to_csv(OUT / "09_task_class_distributions.csv", index=False)
print(t.pivot_table(index=["task", "cls"], columns="split", values=["n", "share"], aggfunc="first")[["n", "share"]].reindex(columns=["train", "validation", "test"], level=1).to_string())

# cancel rate by split & month coverage
m = df.groupby("SPLIT").agg(rows=("CANCELLED", "size"), cancelled=("CANCELLED", "sum")); m["rate"] = (m.cancelled / m.rows).round(4)
print("\n", m.loc[["train", "validation", "test"]])
print("\nCOVID-period (2020-03..2020-12) share of train rows:", round(((df.SPLIT == "train") & df.FL_DATE.between("2020-03-01", "2020-12-31")).sum() / (df.SPLIT == "train").sum(), 3))
print("test months:", sorted(df[df.SPLIT == "test"].FL_DATE.dt.to_period("M").astype(str).unique()))

# ---- feature & leakage control table -------------------------------------------------------------
F = [
 # feature, source, known_at_prediction, cancel, code, delay, note
 ("FL_DATE -> MONTH/DAY_OF_WEEK/SEASON (+sin/cos)", "schedule", "yes", "keep", "keep", "keep", "calendar fields of the scheduled date"),
 ("YEAR / YEAR_NORMALIZED", "schedule", "yes", "DROP", "DROP", "DROP", "old data: 2016 and 2024 were 100% cancelled, so year leaked the label; also non-stationary across chronological splits"),
 ("AIRLINE_CODE", "schedule", "yes", "keep", "keep", "keep", "categorical (target/one-hot/native categorical, NOT label-encoded integers)"),
 ("ORIGIN, DEST (airport)", "schedule", "yes", "keep", "keep", "keep", "replaces ORIGIN_CITY/DEST_CITY (city merges several airports)"),
 ("FL_NUMBER", "schedule", "yes", "DROP", "DROP", "DROP", "identifier, high-cardinality; used only for dedup"),
 ("CRS_DEP_TIME, CRS_ARR_TIME (+time-of-day bins)", "schedule", "yes", "keep", "keep", "keep", "scheduled times only"),
 ("DISTANCE (+category)", "schedule", "yes", "keep", "keep", "keep", ""),
 ("DEP_TIME", "actual", "NO - post-event", "DROP", "DROP", "DROP", "actual departure; absent in 97% of old cancelled-only rows (proxy of label); already dropped in old code, stays dropped"),
 ("DEP_DELAY", "actual", "NO - post-event", "DROP", "DROP", "TARGET", "delay-model target only"),
 ("ARR_TIME, ARR_DELAY, TAXI_OUT/IN, WHEELS_ON/OFF, ELAPSED_TIME, AIR_TIME, DELAY_DUE_*", "actual", "NO - post-event", "DROP", "DROP", "DROP", "not even loaded in clean_flights (08_build_dataset.py)"),
 ("DIVERTED", "actual", "NO - post-event", "DROP", "DROP", "FILTER", "used only to exclude diverted flights from the delay task"),
 ("CANCELLED / CANCELLATION_CODE", "actual", "NO - post-event", "TARGET", "TARGET", "FILTER", "targets / row filters only"),
 ("D1_tmin, D1_tmax, D1_tavg, D1_prcp, D1_snow, D1_wspd, D1_pres", "Meteostat daily, day D-1, origin airport", "yes (completed previous day)", "keep", "keep", "keep", "replaces same-day tmin..pres; NaN snow/prcp need an explicit rule"),
 ("old same-day tmin,tmax,prcp,snow,wdir,wspd,wpgt,pres,tsun", "Meteostat daily, day D", "NO - look-ahead", "DROP", "DROP", "DROP", "reviewer R1-2; wdir/wpgt/tsun also >50% missing"),
 ("WEATHER_COMPOSITE", "derived from weather", "depends", "DROP (candidate)", "DROP (candidate)", "DROP (candidate)", "ad-hoc weights, no validation (R2-C); roadmap: drop unless formula+weights can be justified"),
 ("AIRLINE_RELIABILITY (old)", "target mean over ALL rows", "NO - leakage", "RECOMPUTE", "RECOMPUTE", "-", "old code: groupby('AIR')['CANCELLED'].mean() on the full data incl. test years. New: only from rows before the flight date (expanding/lagged) or train-fit"),
 ("ROUTE_POPULARITY(_LOG) (old)", "count over ALL rows", "NO - uses future volume", "RECOMPUTE", "RECOMPUTE", "-", "count of flights per route over the full data; new: train-only or lagged counts"),
 ("historical route performance / previous delays", "derived", "only if lagged", "optional", "optional", "optional", "must use strictly earlier flight dates"),
 ("imputer / scaler / encoders", "preprocessing", "-", "fit on TRAIN only", "fit on TRAIN only", "fit on TRAIN only", "old delay_model.py fitted imputer+scaler on all rows and scaled the target"),
]
ft = pd.DataFrame(F, columns=["feature", "source", "known_at_prediction_time", "cancel_model", "cancel_code_model", "delay_model", "note"])
ft.to_csv(OUT / "09_feature_leakage_table.csv", index=False)
print("\nfeature/leakage table ->", OUT / "09_feature_leakage_table.csv", f"({len(ft)} rows)")
