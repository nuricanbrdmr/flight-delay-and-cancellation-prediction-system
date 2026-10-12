"""Full 3M Zelaya file: natural rates, duplicates on the roadmap key, label consistency, overlap with Singh (2024)."""
import pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "csv"
OUT = Path(__file__).resolve().parent / "outputs"
pd.set_option("display.width", 220)
s = pd.read_csv(D / "flights_sample_3m_origin.csv", low_memory=False, parse_dates=["FL_DATE"])
print("rows", len(s), s.FL_DATE.min().date(), s.FL_DATE.max().date())
k = ["FL_DATE","DOT_CODE","FL_NUMBER","ORIGIN","DEST","CRS_DEP_TIME"]
print("duplicates on roadmap key:", int(s.duplicated(k).sum()), "| fully identical rows:", int(s.duplicated().sum()))
print("CANCELLED x has-code:"); print(pd.crosstab(s.CANCELLED, s.CANCELLATION_CODE.notna()))
print("cancel codes:", s.CANCELLATION_CODE.value_counts().to_dict(), "| diverted:", int(s.DIVERTED.sum()))
y = s.groupby(s.FL_DATE.dt.year).agg(rows=("CANCELLED","size"), cancelled=("CANCELLED","sum")); y["rate"] = (y.cancelled/y.rows).round(5)
print(y); y.to_csv(OUT/"06_zelaya_full_year_rates.csv")
nc = s[(s.CANCELLED==0)&(s.DIVERTED==0)]
print("delay-task rows (not cancelled, not diverted):", len(nc), "| DEP_DELAY missing:", int(nc.DEP_DELAY.isna().sum()))
print("airports:", s.ORIGIN.nunique(), "origin cities:", s.ORIGIN_CITY.nunique())
# does the earlier truncated 1,048,575 sample lie inside the full file?
t = pd.read_csv(D/"flights_sample_3m.csv", sep=";", usecols=["FL_DATE","DOT_CODE","FL_NUMBER","ORIGIN","DEST","CRS_DEP_TIME"])
t["FL_DATE"] = pd.to_datetime(t.FL_DATE, format="%d.%m.%Y")
print("truncated sample rows found in full file:", t.merge(s[k], on=k).shape[0], "of", len(t))
x = pd.read_csv(D/"flight_data_2018_2024.csv", usecols=["FlightDate","Cancelled","Diverted","DepDelay"])
print("Singh Jan-2024: rows", len(x), "cancelled", int(x.Cancelled.sum()), "diverted", int(x.Diverted.sum()))
