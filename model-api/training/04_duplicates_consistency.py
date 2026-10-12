"""Roadmap step 2: duplicate report; plus label consistency in the sources and final-vs-source row matching."""
import pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "csv"
OUT = Path(__file__).resolve().parent / "outputs"
pd.set_option("display.width", 220)

s = pd.read_csv(D / "flights_sample_3m.csv", sep=";", usecols=["FL_DATE","AIRLINE_CODE","DOT_CODE","FL_NUMBER","ORIGIN","ORIGIN_CITY","DEST","DEST_CITY","CRS_DEP_TIME","CANCELLED","CANCELLATION_CODE","DIVERTED","DEP_DELAY"])
s["FL_DATE"] = pd.to_datetime(s.FL_DATE, format="%d.%m.%Y")
print("== SOURCE sample_3m label consistency (CANCELLED x has code):")
print(pd.crosstab(s.CANCELLED, s.CANCELLATION_CODE.notna()))
print("diverted:", int((s.DIVERTED == 1).sum()), "| cancelled&diverted:", int(((s.DIVERTED==1)&(s.CANCELLED==1)).sum()))
k_full = ["FL_DATE","DOT_CODE","FL_NUMBER","ORIGIN","DEST","CRS_DEP_TIME"]
dup_full = s.duplicated(k_full, keep=False)
print(f"\n== sample_3m duplicates on roadmap key {k_full}: rows in dup groups={int(dup_full.sum())}, extra copies={int(s.duplicated(k_full).sum())}")

k_red = ["FL_DATE","AIRLINE_CODE","ORIGIN_CITY","DEST_CITY","CRS_DEP_TIME"]
rep = []
rep.append(("sample_3m", "full roadmap key", len(s), int(s.duplicated(k_full).sum())))
rep.append(("sample_3m", "reduced key (no FL_NUMBER/airport code)", len(s), int(s.duplicated(k_red).sum())))

x = pd.read_csv(D / "flight_data_2018_2024.csv", usecols=["FlightDate","IATA_Code_Operating_Airline","DOT_ID_Operating_Airline","Flight_Number_Operating_Airline","Origin","Dest","CRSDepTime","Cancelled","CancellationCode","Duplicate"])
kx = ["FlightDate","DOT_ID_Operating_Airline","Flight_Number_Operating_Airline","Origin","Dest","CRSDepTime"]
rep.append(("flight_data_2018_2024", "full roadmap key", len(x), int(x.duplicated(kx).sum())))
print("\n== flight_data_2018_2024: own 'Duplicate' column ->", x.Duplicate.value_counts(dropna=False).to_dict())
print("   cancelled x code:"); print(pd.crosstab(x.Cancelled, x.CancellationCode.fillna("NaN")))

for fn, cityc, destc, name in [("flights_Cancel_final.csv","ORIGIN_CITY_NAME","DEST_CITY_NAME","Cancel_final"), ("flights_Delay_final.csv","ORIGIN_CITY","DEST_CITY","Delay_final")]:
    f = pd.read_csv(D / fn, parse_dates=["FL_DATE"])
    kk = ["FL_DATE","AIRLINE_CODE",cityc,destc,"CRS_DEP_TIME"]
    rep.append((name, "reduced key (no FL_NUMBER available)", len(f), int(f.duplicated(kk).sum())))
    rep.append((name, "ALL columns identical", len(f), int(f.duplicated().sum())))
    # cross-dataset overlap: how many final rows are found in the (truncated) 3m sample, and do CANCELLED flags agree?
    a = f.rename(columns={cityc:"ORIGIN_CITY", destc:"DEST_CITY"})
    a = a[a.FL_DATE.between("2019-01-01","2023-08-31")]
    sm = s.drop_duplicates(k_red)[k_red+["CANCELLED","CANCELLATION_CODE"]]
    aa = a.drop_duplicates(k_red)[k_red+["CANCELLED","CANCELLATION_CODE"]]
    j = aa.merge(sm, on=k_red, how="left", suffixes=("","_src"), indicator=True)
    hit = j[j._merge == "both"]
    print(f"\n== {name}: 2019-2023 rows (unique by reduced key)={len(aa):,}; found in provided sample_3m={len(hit):,} ({len(hit)/len(aa):.1%})")
    print("   CANCELLED agrees with source:", float((hit.CANCELLED == hit.CANCELLED_src).mean()).__round__(4),
          "| code agrees (NaN==NaN):", float(((hit.CANCELLATION_CODE == hit.CANCELLATION_CODE_src) | (hit.CANCELLATION_CODE.isna() & hit.CANCELLATION_CODE_src.isna())).mean()).__round__(4))
r = pd.DataFrame(rep, columns=["dataset","key","rows","duplicate_rows"]); r["share"] = (r.duplicate_rows/r.rows).round(5)
r.to_csv(OUT/"04_duplicate_report.csv", index=False); print("\n", r.to_string(index=False))
