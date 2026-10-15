"""Steps 3 and 5: target/class distributions per task + feasibility of D-1 weather from existing weather columns."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "csv"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
W = ["tmin","tmax","prcp","snow","wdir","wspd","wpgt","pres","tsun"]

d = pd.read_csv(D / "flights_Delay_final.csv", parse_dates=["FL_DATE"]); d["Y"] = d.FL_DATE.dt.year
print("== DELAY file: label consistency")
print(pd.crosstab(d.CANCELLED, d.CANCELLATION_CODE.notna(), rownames=["CANCELLED"], colnames=["has_code"]))
print(pd.crosstab(d.Y, [d.CANCELLED, d.CANCELLATION_CODE.notna()]))
nc = d[d.CANCELLED == 0]
print("\nDEP_DELAY missing among CANCELLED==0:", nc.DEP_DELAY.isna().sum(), "| DEP_TIME missing among CANCELLED==0:", nc.DEP_TIME.isna().sum())
m = nc.DEP_DELAY.dropna()
true_cls = pd.cut(m, [-np.inf, 0, 15, 30, np.inf], labels=[0,1,2,3])
print("\nDelay classes, intended (minutes):"); print(true_cls.value_counts(normalize=True).sort_index().round(4).to_dict(), true_cls.value_counts().sort_index().to_dict())
z = (m - m.mean()) / m.std(ddof=0)   # what delay_model.py actually does (after median imputation ~ same)
bug_cls = pd.cut(z, [-np.inf, 0, 15, 30, np.inf], labels=[0,1,2,3])
print("Delay classes as delay_model.py computes them (thresholds applied to z-scores):"); print(bug_cls.value_counts().sort_index().to_dict())

print("\n== Weather: is it one value per (date, ORIGIN city)?")
w = d.groupby(["FL_DATE","ORIGIN_CITY"])[W].nunique(dropna=False).max()
print("max distinct values per (date, origin) group:", w.to_dict())
print("share missing:", d[W].isna().mean().round(3).to_dict())
print("\n== Weather: does DEST get its own weather? (compare weather of rows where city appears as origin vs dest -> only origin columns exist)")
c = pd.read_csv(D / "flights_Cancel_final.csv", parse_dates=["FL_DATE"])
for name, df, oc in [("delay", d, "ORIGIN_CITY"), ("cancel", c, "ORIGIN_CITY_NAME")]:
    key = df[[ "FL_DATE", oc ]].drop_duplicates()
    prev = key.assign(FL_DATE=key.FL_DATE + pd.Timedelta(days=1))
    have = key.merge(prev.rename(columns={"FL_DATE":"nxt"}).assign(FL_DATE=lambda x: x.nxt)[[ "FL_DATE", oc ]].assign(has_prev=True),
                     how="left", on=["FL_DATE", oc])
    # simpler/robust: coverage of (date-1, city) among (date, city) pairs
    s = set(map(tuple, key.values)); cov = np.mean([ (dt - pd.Timedelta(days=1), ct) in s for dt, ct in key.values ])
    print(f"{name}: (date,origin) pairs={len(key):,}; share whose D-1 observation exists in the same file={cov:.3f}")
