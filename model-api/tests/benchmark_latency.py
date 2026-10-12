"""Response-time measurement of the model API (in-process Flask test client, i.e. without HTTP transport; real Meteostat calls on a cold cache).
usage: python tests/benchmark_latency.py   -> training/outputs/21_api_latency.json"""
import json, sys, time, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
t0 = time.time()
import cancel_delay_api as api
api.load_artifacts()
load_s = time.time() - t0
c = api.app.test_client()
routes = [("ORD", "LGA", "UA"), ("ATL", "MCO", "DL"), ("DEN", "LAX", "UA"), ("DFW", "PHX", "AA"), ("SEA", "SFO", "AS"), ("JFK", "BOS", "B6"), ("MSP", "DTW", "DL"), ("LAS", "SLC", "WN"), ("MIA", "ATL", "AA"), ("BOS", "DCA", "AA")]
days = ["2023-02-03", "2023-06-14", "2023-08-20"]
cold, warm, noweather = [], [], []
for d in days:
    for o, de, al in routes:
        body = dict(date=d, airline=al, origin_airport=o, destination_airport=de, departure_time="08:15", arrival_time="11:00")
        a = c.post("/predict", json=body); assert a.status_code == 200, a.get_json()
        cold.append(a.get_json()["latency_ms"])
        b = c.post("/predict", json=body); warm.append(b.get_json()["latency_ms"])
for o, de, al in routes:
    r = c.post("/predict", json=dict(date="2031-05-05", airline=al, origin_airport=o, destination_airport=de, departure_time="08:15", arrival_time="11:00")); noweather.append(r.get_json()["latency_ms"])
q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
out = dict(model_load_seconds=round(load_s, 1), n_cold=len(cold), cold_ms=dict(median=st.median(cold), p95=q(cold, .95), max=max(cold)), warm_ms=dict(median=st.median(warm), p95=q(warm, .95), max=max(warm)), future_date_no_weather_ms=dict(median=st.median(noweather), p95=q(noweather, .95)),
           note="in-process; excludes HTTP/network transport and the flight-search call; cold = first request for an (airport, day) pair incl. Meteostat download")
json.dump(out, open(ROOT / "training" / "outputs" / "21_api_latency.json", "w"), indent=1)
print(json.dumps(out, indent=1))
