"""Meteostat DAILY weather per origin airport, 2018-12-31 .. 2024-01-31 (observed values, labelled with the day they describe; D-1 shift is in 08).
Point() interpolation is broken in meteostat 1.7 for many sites, so we use stations directly: nearest stations within MAX_KM, take the
nearest as primary and fill its missing days from the next ones (max 3 stations). Station ids / distances / coverage are logged.
Outputs (git-ignored): csv/weather_daily_by_airport.csv.gz, csv/weather_station_map.csv"""
import datetime as dt, time
import airportsdata, pandas as pd
from meteostat import Stations, Daily
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "csv"
OUT, MAP = D / "weather_daily_by_airport.csv.gz", D / "weather_station_map.csv"
START, END, MAX_KM, MAX_ST = dt.datetime(2018, 12, 31), dt.datetime(2024, 1, 31), 50, 3
VARS = ["tavg","tmin","tmax","prcp","snow","wdir","wspd","wpgt","pres","tsun"]
iata = airportsdata.load("IATA")
airports = sorted(set(pd.read_csv(D / "flights_sample_3m_origin.csv", usecols=["ORIGIN"]).ORIGIN) | set(pd.read_csv(D / "flight_data_2018_2024.csv", usecols=["Origin"]).Origin))
missing = [a for a in airports if a not in iata]
print(f"{len(airports)} origin airports; no coordinates: {missing}", flush=True)
days = pd.date_range(START, END)
frames, rows = [], []
for i, a in enumerate(airports):
    if a in missing: continue
    r = iata[a]
    st = Stations().nearby(r["lat"], r["lon"]).fetch(8)
    st = st[st.distance <= MAX_KM * 1000]
    comb, used = None, []
    for sid, srow in st.iterrows():
        if comb is not None and len(used) >= MAX_ST: break
        try: d = Daily(sid, START, END).fetch()
        except Exception: continue
        if d is None or d.empty or d.tmax.notna().sum() < 100: continue
        d = d.reindex(days)
        comb = d if comb is None else comb.combine_first(d)
        used.append((sid, round(srow.distance / 1000, 1)))
        if comb.tmax.notna().mean() >= 0.97: break
    if comb is None:
        rows.append(dict(ORIGIN=a, stations="", tmax_cov=0.0)); print("NO DATA", a, flush=True); continue
    comb = comb.reindex(columns=VARS).rename_axis("WEATHER_DATE").reset_index(); comb.insert(0, "ORIGIN", a); frames.append(comb)
    rows.append(dict(ORIGIN=a, stations=";".join(f"{s}@{k}km" for s, k in used), tmax_cov=round(comb.tmax.notna().mean(), 3)))
    if i % 20 == 0:
        pd.concat(frames).to_csv(OUT, index=False); pd.DataFrame(rows).to_csv(MAP, index=False); print(f"{i}/{len(airports)} {a}", flush=True)
pd.concat(frames).to_csv(OUT, index=False); pd.DataFrame(rows).to_csv(MAP, index=False); print("done", flush=True)
