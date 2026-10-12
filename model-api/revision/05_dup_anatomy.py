"""Why does Delay_final contain each flight ~2x? Which columns differ inside duplicate groups?"""
import pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[2] / "csv"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for fn, oc, dc in [("flights_Delay_final.csv","ORIGIN_CITY","DEST_CITY"), ("flights_Cancel_final.csv","ORIGIN_CITY_NAME","DEST_CITY_NAME")]:
    f = pd.read_csv(D / fn, parse_dates=["FL_DATE"]); f = f[f.FL_DATE.between("2019-01-01","2023-08-31")].copy()
    k = ["FL_DATE","AIRLINE_CODE",oc,dc,"CRS_DEP_TIME"]
    f["n"] = f.groupby(k)[k[0]].transform("size")
    print(f"\n=== {fn} (2019-2023 part): rows={len(f):,}; group-size distribution:", f.drop_duplicates(k).n.value_counts().sort_index().to_dict())
    g = f[f.n > 1]
    cols = [c for c in f.columns if c not in k+["n"]]
    diff = g.groupby(k)[cols].nunique(dropna=False).gt(1).mean().round(3)
    print("share of dup-groups in which column differs:", diff[diff > 0].to_dict())
    print(g.sort_values(k).head(6).to_string())
    # is the 1st/2nd copy different in cancel status?
    print("groups with conflicting CANCELLED:", int((g.groupby(k).CANCELLED.nunique() > 1).sum()), "of", g.groupby(k).ngroups)
