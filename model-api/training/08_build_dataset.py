"""Rebuild the base table from the raw sources (no model-specific encoding, no balancing).
 - Zelaya full 3M (2019-01..2023-08) + Singh Jan-2024 (natural: cancelled AND non-cancelled). The 2016 / 2024 cancelled-only rows of the old file are NOT used.
 - Dedup on the roadmap key. Weather: Meteostat daily at the origin airport, joined many-to-one on (ORIGIN, FL_DATE - 1 day) = D-1.
 - Only pre-departure fields + targets are kept. Post-event fields (DEP_TIME, TAXI_*, WHEELS_*, ARR_*, ELAPSED/AIR_TIME, DELAY_DUE_*) are dropped here.
Output: csv/clean_flights.csv.gz (git-ignored) + outputs/08_build_report.txt"""
import pandas as pd, numpy as np
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "csv"
OUT = Path(__file__).resolve().parent / "outputs"
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

KEY = ["FL_DATE", "DOT_CODE", "FL_NUMBER", "ORIGIN", "DEST", "CRS_DEP_TIME"]
KEEP = ["FL_DATE","AIRLINE_CODE","DOT_CODE","FL_NUMBER","ORIGIN","DEST","ORIGIN_CITY","DEST_CITY","CRS_DEP_TIME","CRS_ARR_TIME","DISTANCE",
        "CANCELLED","CANCELLATION_CODE","DIVERTED","DEP_DELAY"]

z = pd.read_csv(D / "flights_sample_3m_origin.csv", usecols=KEEP, parse_dates=["FL_DATE"], low_memory=False); z["SOURCE"] = "zelaya_2019_2023"
x = pd.read_csv(D / "flight_data_2018_2024.csv", low_memory=False,
                usecols=["FlightDate","IATA_Code_Operating_Airline","DOT_ID_Operating_Airline","Flight_Number_Operating_Airline","Origin","Dest",
                         "OriginCityName","DestCityName","CRSDepTime","CRSArrTime","Distance","Cancelled","CancellationCode","Diverted","DepDelay"])
x = x.rename(columns={"FlightDate":"FL_DATE","IATA_Code_Operating_Airline":"AIRLINE_CODE","DOT_ID_Operating_Airline":"DOT_CODE","Flight_Number_Operating_Airline":"FL_NUMBER",
                      "Origin":"ORIGIN","Dest":"DEST","OriginCityName":"ORIGIN_CITY","DestCityName":"DEST_CITY","CRSDepTime":"CRS_DEP_TIME","CRSArrTime":"CRS_ARR_TIME",
                      "Distance":"DISTANCE","Cancelled":"CANCELLED","CancellationCode":"CANCELLATION_CODE","Diverted":"DIVERTED","DepDelay":"DEP_DELAY"})
x["FL_DATE"] = pd.to_datetime(x.FL_DATE)
for c in ["CRS_DEP_TIME","CRS_ARR_TIME","FL_NUMBER"]: x[c] = pd.to_numeric(x[c], errors="coerce")
x["SOURCE"] = "singh_2024_01"
df = pd.concat([z, x[KEEP + ["SOURCE"]]], ignore_index=True)
say("rows: zelaya", len(z), "| singh", len(x), "| total", len(df))
n = df.duplicated(KEY).sum(); say("duplicates on roadmap key (dropped):", int(n)); df = df.drop_duplicates(KEY)
df["CANCELLED"] = df.CANCELLED.astype(int); df["DIVERTED"] = df.DIVERTED.astype(int)
say("label consistency CANCELLED vs code:\n", pd.crosstab(df.CANCELLED, df.CANCELLATION_CODE.notna()).to_string())

w = pd.read_csv(D / "weather_daily_by_airport.csv.gz", parse_dates=["WEATHER_DATE"])
wcols = ["tmin","tmax","tavg","prcp","snow","wspd","pres"]          # wdir/wpgt/tsun are >50% missing in Meteostat daily
w1 = w[["ORIGIN","WEATHER_DATE"] + wcols].copy()
w1["FL_DATE"] = w1.WEATHER_DATE + pd.Timedelta(days=1)             # observation of day D-1 is attached to flights of day D
w1 = w1.drop(columns="WEATHER_DATE").rename(columns={c: f"D1_{c}" for c in wcols})
n0 = len(df); df = df.merge(w1, on=["ORIGIN","FL_DATE"], how="left", validate="many_to_one"); assert len(df) == n0
say("rows after weather join (must equal before):", len(df))
say("D-1 weather non-missing share:\n", df[[f"D1_{c}" for c in wcols]].notna().mean().round(4).to_string())
no_wx = df[df.D1_tmax.isna()].ORIGIN.value_counts().head(10)
say("top origins with missing D-1 tmax:\n", no_wx.to_string())
df = df.sort_values(["FL_DATE","CRS_DEP_TIME","ORIGIN","FL_NUMBER"]).reset_index(drop=True)
df.to_csv(D / "clean_flights.csv.gz", index=False)
t = df.groupby(["SOURCE", df.FL_DATE.dt.year]).agg(rows=("CANCELLED","size"), cancelled=("CANCELLED","sum"), diverted=("DIVERTED","sum"))
t["cancel_rate"] = (t.cancelled / t.rows).round(5); say("\n" + t.to_string())
(OUT / "08_build_report.txt").write_text("\n".join(log), encoding="utf-8")
