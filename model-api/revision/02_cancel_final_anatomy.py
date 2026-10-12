"""Where do the rows of flights_Cancel_final / flights_Delay_final come from? Year x cancel x code, missingness, test-split rate."""
import pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "csv"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
c = pd.read_csv(D / "flights_Cancel_final.csv", parse_dates=["FL_DATE"])
c["Y"] = c.FL_DATE.dt.year
print("Cancel_final: cancellation code by year (cancelled rows only)")
print(pd.crosstab(c.Y[c.CANCELLED == 1], c.CANCELLATION_CODE[c.CANCELLED == 1].fillna("NaN")))
print("\ncode on NON-cancelled rows:", c.CANCELLATION_CODE[c.CANCELLED == 0].fillna("NaN").value_counts().to_dict())
print("\nmissing share by year:"); print(c.groupby("Y")[["DEP_TIME","wpgt","tsun","pres","wdir","AIRLINE_CODE","CRS_ARR_TIME"]].apply(lambda g: g.isna().mean().round(3)))
print("\nmonths present per year:"); print(c.groupby("Y").FL_DATE.agg(lambda s: s.dt.month.nunique()))
tr, te = c[c.Y <= 2022], c[c.Y >= 2023]
print(f"\nCurrent split: train={len(tr):,} (cancel {tr.CANCELLED.mean():.4f}) test={len(te):,} (cancel {te.CANCELLED.mean():.4f})")
te23 = c[c.Y == 2023]; print(f"2023 only (natural): {len(te23):,}, cancel {te23.CANCELLED.mean():.4f}")
print("\nrows per day (natural years) vs 2016/2024:"); print(c.groupby("Y").FL_DATE.agg(["nunique","size"]).assign(per_day=lambda d: (d["size"]/d["nunique"]).round(0)))
print("\ncancel_code N: rows with code 'N' =", (c.CANCELLATION_CODE == "N").sum())
d = pd.read_csv(D / "flights_Delay_final.csv", parse_dates=["FL_DATE"])
print("\nDelay_final non-cancelled DEP_DELAY describe:"); print(d.loc[d.CANCELLED == 0, "DEP_DELAY"].describe().round(2))
print("DEP_DELAY missing among non-cancelled:", d.loc[d.CANCELLED == 0, "DEP_DELAY"].isna().sum())
print("code values in Delay_final:", d.CANCELLATION_CODE.fillna("NaN").value_counts().to_dict())
