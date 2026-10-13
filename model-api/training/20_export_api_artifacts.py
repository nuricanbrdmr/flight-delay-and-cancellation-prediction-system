"""Export what the SmartTicket model API (model-api/cancel_delay_api.py) needs, from the validated experiments:
 - Mode A models: cancellation (cancel), cause (code), four-class delay without class weights (delayacc)  -> model-api/models_v2/*.ubj
 - config.json (feature lists, category lists, route counts, priors, class names, thresholds chosen on VALIDATION, headline metrics)
 - history lookup tables for 2023-01-01..2024-01-31 and the weather station map
 - a parity sample: raw inputs + the probabilities the TRAINING pipeline gives, so tests can prove that the API builds identical features."""
import sys, json, shutil, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C, fe
from xgboost import XGBClassifier

OUT = Path(__file__).resolve().parents[1] / "models_v2"
OUT.mkdir(exist_ok=True)
(OUT.parent / "tests").mkdir(exist_ok=True)
O = C.OUT
res = json.load(open(O / "14_test_results.json"))

df = fe.load_v2("A")
cats = fe.train_cats(df)
tr = df[df.SPLIT == "train"]
ok = (tr.CANCELLED == 0) & (tr.DIVERTED == 0)
priors = dict(cancel=float(tr.CANCELLED.mean()), delay15=float((ok & (tr.DEP_DELAY > 15)).sum() / ok.sum()), mean_delay=float(tr.DEP_DELAY[ok].clip(-30, 180).mean()))
route_counts = tr.groupby(["ORIGIN", "DEST"]).size()

models = {}
for tag, name in (("cancel", "cancel"), ("code", "code"), ("delayacc", "delay")):
    m = XGBClassifier(); m.load_model(C.D / "models_v2" / f"{tag}.json")
    m.get_booster().save_model(str(OUT / f"{name}.ubj"))
    best = json.load(open(O / f"13_{tag}_best.json"))
    models[name] = dict(features=best["features"], val_metrics=best["val_metrics"], params=best["params"], weight_mode=best["weight_mode"])

cost = pd.read_csv(O / "14_cancel_cost_threshold.csv")
cfg = dict(
    version="2.0-revision", mode="A (information up to the previous completed day, D-1)",
    split=dict(train="2019-01-01..2022-06-30", validation="2022-07-01..2022-12-31",
              test="2023-01-01..2023-08-31 and 2024-01-01..2024-01-31"),   # the data contain no flights for 2023-09..2023-12
    cats={k: v for k, v in cats.items()},
    route_counts={f"{o}-{d}": int(n) for (o, d), n in route_counts.items()},
    priors=priors,
    classes=dict(cause=["A", "B", "C"], delay=["on_time_or_early", "delay_1_15", "delay_16_30", "delay_over_30"]),
    models=models,
    thresholds=dict(cancel_alert_f1_validation=res["cancel"]["at_val_bestF1_threshold"]["threshold"],
                    cancel_cost_ratio={str(int(r.cost_ratio_FN_to_FP)): float(r.threshold_from_validation) for r in pd.read_csv(O / "14_cancel_cost_threshold.csv").itertuples()}),
    test_metrics=dict(cancel=res["cancel"]["test"], cause=res["code"]["test"], delay=res["delayacc"]["test"], delay_gt15=res["delayacc"]["delay_gt15_binary_test"], prevalence_test=res["cancel"]["prevalence_test"],
                      cause_baseline=res["code"]["baseline_majority_accuracy_test"], delay_baseline=res["delayacc"]["baseline_majority_accuracy_test"]),
    history_range=["2023-01-01", "2024-01-31"],
)
json.dump(cfg, open(OUT / "config.json", "w"), indent=1)

# history lookups (rates are computed from strictly earlier days, see 12_features_v2.py); restricted to the test period to keep the files small
v2 = df[df.FL_DATE >= "2023-01-01"]
for tag, keys, cols in (("nat", ["FL_DATE"], [c for c in v2.columns if c.startswith("H_NAT_")]), ("org", ["ORIGIN", "FL_DATE"], [c for c in v2.columns if c.startswith("H_ORG_")]), ("air", ["AIRLINE_CODE", "FL_DATE"], [c for c in v2.columns if c.startswith("H_AIR_")])):
    t = v2[keys + cols].drop_duplicates(keys).copy()
    t["FL_DATE"] = t.FL_DATE.dt.strftime("%Y-%m-%d")
    t.round(5).to_csv(OUT / f"history_{tag}.csv.gz", index=False)
shutil.copy(C.D / "weather_station_map.csv", OUT / "weather_station_map.csv")

# parity sample: raw inputs + expected probabilities from the training pipeline (test period, Mode A models)
rng = np.random.RandomState(5)
test = df[df.SPLIT == "test"].sample(300, random_state=5)
cols = fe.SETS[fe.FULL]
out = []
Xs = {n: fe.matrix(test, models[n]["features"], cats) for n in models}
P = {}
for n, name in (("cancel", "cancel"), ("code", "code"), ("delay", "delay")):
    m = XGBClassifier(); m.load_model(str(OUT / f"{name}.ubj"))
    P[n] = m.predict_proba(Xs[n])
for i, (idx, r) in enumerate(test.iterrows()):
    out.append(dict(
        raw=dict(date=r.FL_DATE.strftime("%Y-%m-%d"), airline=r.AIRLINE_CODE, origin=r.ORIGIN, destination=r.DEST, crs_dep=int(r.CRS_DEP_TIME), crs_arr=int(r.CRS_ARR_TIME), distance_miles=float(r.DISTANCE)),
        weather={c: (None if pd.isna(r[c]) else float(r[c])) for c in C.WEATHER + fe.WX_D},
        history={c: (None if pd.isna(r[c]) else float(r[c])) for c in fe.HIST},
        expected=dict(cancel=float(P["cancel"][i][1] if P["cancel"].ndim > 1 else P["cancel"][i]), cause=[float(x) for x in P["code"][i]], delay=[float(x) for x in P["delay"][i]])))
json.dump(out, open(OUT.parent / "tests" / "parity_sample.json", "w"))
print("exported to", OUT, [f.name for f in OUT.iterdir()])
