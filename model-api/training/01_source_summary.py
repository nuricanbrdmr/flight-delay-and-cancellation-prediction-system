"""Step 1: per-source summary (date range, rows, cancelled/non-cancelled, rate) + per-year cancel rate."""
import pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "csv"
OUT = Path(__file__).resolve().parent / "outputs"; OUT.mkdir(exist_ok=True)

SPEC = {  # file: (sep, date col, cancelled col, date format)
    "flights_sample_3m.csv":     (";", "FL_DATE", "CANCELLED", "%d.%m.%Y"),
    "flight_data_2018_2024.csv": (",", "FlightDate", "Cancelled", None),
    "flights_Cancel_final.csv":  (",", "FL_DATE", "CANCELLED", None),
    "flights_Delay_final.csv":   (",", "FL_DATE", "CANCELLED", None),
}
rows, years = [], []
for fn, (sep, dc, cc, fmt) in SPEC.items():
    df = pd.read_csv(D / fn, sep=sep, usecols=[dc, cc], low_memory=False)
    df[dc] = pd.to_datetime(df[dc], format=fmt, errors="coerce")
    bad = df[dc].isna().sum()
    c = df[cc].fillna(0).astype(float)
    rows.append(dict(file=fn, rows=len(df), date_min=df[dc].min().date(), date_max=df[dc].max().date(),
                     unparsable_dates=int(bad), cancelled=int((c == 1).sum()), not_cancelled=int((c == 0).sum()),
                     cancel_rate=round((c == 1).mean(), 5)))
    g = df.assign(year=df[dc].dt.year, c=c).groupby("year").c.agg(rows="size", cancelled="sum")
    g["cancel_rate"] = (g.cancelled / g.rows).round(5); g.insert(0, "file", fn); years.append(g.reset_index())
s = pd.DataFrame(rows); y = pd.concat(years)
s.to_csv(OUT / "01_source_summary.csv", index=False); y.to_csv(OUT / "01_year_cancel_rates.csv", index=False)
pd.set_option("display.width", 200)
print(s.to_string(index=False)); print(); print(y.to_string(index=False))

# full_data_flightdelay has no date/cancel column
f = pd.read_csv(D / "full_data_flightdelay.csv", usecols=["MONTH", "DEP_DEL15"])
print("\nfull_data_flightdelay.csv rows=%d, DEP_DEL15 rate=%.4f, cols: no date/year, no CANCELLED" % (len(f), f.DEP_DEL15.mean()))
