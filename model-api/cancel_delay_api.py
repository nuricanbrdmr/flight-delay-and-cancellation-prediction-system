"""SmartTicket model API, version 2 (revision).

Serves the models validated in model-api/training (chronological protocol, no resampling, no post-hoc probability adjustments):
  * cancellation probability               (binary XGBoost)
  * cancellation cause A/B/C, conditional on a cancellation  (XGBoost)
  * departure-delay class probabilities    (four classes, XGBoost without class weights)

Mode A only: information up to the previous completed day (D-1):
  calendar/holiday + schedule + observed D-1 weather (origin, destination; Meteostat station data)
  + recent operational history (cancellation / delay rates of D-1 and D-7..D-1; only available for 2023-01-01..2024-01-31 here).
Mode B (same-day operational state three hours before departure) needs a live data feed and is NOT implemented in this API.

What the output is: a risk indicator from recent observations - not a weather forecast and not a guarantee.
Missing inputs are reported in `warnings`; neutral training-period averages are used for unavailable history.
"""
import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from pandas.tseries.holiday import USFederalHolidayCalendar
from xgboost import Booster, DMatrix

HERE = Path(__file__).resolve().parent
ART = Path(os.environ.get("MODEL_ARTIFACTS", HERE / "models_v2"))
WEATHER_VARS = ["tmin", "tmax", "tavg", "prcp", "snow", "wspd", "pres"]
DEST_WEATHER_VARS = ["tmin", "tmax", "prcp", "snow", "wspd"]
MI_PER_KM = 0.621371

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024            # requests are tiny; reject anything larger
CORS(app, origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(","))
API_KEY = os.environ.get("MODEL_API_KEY")                 # optional shared secret between backend and model API

_state = {}
_lock = threading.Lock()
_weather_cache = {}
_iata = None


# ------------------------------------------------------------------------------------------------ artifacts
def load_artifacts():
    with _lock:
        if _state:
            return _state
        cfg = json.load(open(ART / "config.json"))
        models = {}
        for name in ("cancel", "code", "delay"):
            b = Booster()
            b.load_model(str(ART / f"{name}.ubj"))
            models[name] = b
        hist = {
            "nat": pd.read_csv(ART / "history_nat.csv.gz").set_index("FL_DATE"),
            "org": pd.read_csv(ART / "history_org.csv.gz").set_index(["ORIGIN", "FL_DATE"]),
            "air": pd.read_csv(ART / "history_air.csv.gz").set_index(["AIRLINE_CODE", "FL_DATE"]),
        }
        stations = pd.read_csv(ART / "weather_station_map.csv").set_index("ORIGIN").stations.fillna("").to_dict()
        _state.update(cfg=cfg, models=models, hist=hist, stations=stations)
        return _state


def airports():
    global _iata
    if _iata is None:
        _iata = airportsdata.load("IATA")
    return _iata


# ------------------------------------------------------------------------------------------------ helpers
def hhmm_to_min(hhmm):
    hhmm = int(hhmm)
    return (hhmm // 100) * 60 + hhmm % 100


def parse_time(value, field):
    try:
        h, m = str(value).strip().split(":")[:2]
        h, m = int(h), int(m)
        if not (0 <= h < 24 and 0 <= m < 60):
            raise ValueError
        return h * 100 + m
    except Exception:
        raise ValueError(f"{field} must be 'HH:MM'")


def great_circle_miles(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 3958.8 * np.arcsin(np.sqrt(h)))


def holiday_features(day):
    hol = USFederalHolidayCalendar().holidays(day - timedelta(days=400), day + timedelta(days=400)).values
    d = np.datetime64(day.date())
    pos = np.searchsorted(hol, d)
    nxt, prv = hol[min(pos, len(hol) - 1)], hol[max(pos - 1, 0)]
    return min(float((nxt - d) / np.timedelta64(1, "D")), 14.0), min(float((d - prv) / np.timedelta64(1, "D")), 14.0)


def fetch_weather(iata, day):
    """Observed daily weather of calendar day `day` at the airport (same station logic as in training); None if not (yet) available."""
    if day.date() >= datetime.utcnow().date():
        return None                                           # the day is not completed yet
    key = (iata, day.date().isoformat())
    if key in _weather_cache:
        return _weather_cache[key]
    st = load_artifacts()["stations"].get(iata, "")
    out = None
    if st:
        from meteostat import Daily                           # imported lazily: the API also works offline for history-only use
        merged = {}
        for sid in [s.split("@")[0] for s in st.split(";") if s]:
            try:
                d = Daily(sid, day, day).fetch()
            except Exception:
                continue
            if d is None or d.empty:
                continue
            row = d.iloc[0]
            for v in WEATHER_VARS:
                if v not in merged and pd.notna(row.get(v)):
                    merged[v] = float(row[v])
        out = merged or None
    _weather_cache[key] = out
    return out


def history_for(day, origin, airline):
    """History feature dict (H_*) from the lookup tables; neutral training-period averages where unavailable."""
    S = load_artifacts()
    pri = S["cfg"]["priors"]
    key = day.strftime("%Y-%m-%d")
    vals, found = {}, {"nat": False, "org": False, "air": False}
    for lvl, tab, idx in (("nat", S["hist"]["nat"], key), ("org", S["hist"]["org"], (origin, key)), ("air", S["hist"]["air"], (airline, key))):
        try:
            row = tab.loc[idx]
            vals.update(row.to_dict())
            found[lvl] = True
        except KeyError:
            pass
    for lvl in ("NAT", "ORG", "AIR"):
        for w in ("D1", "D7"):
            for m, p in (("CANCEL", pri["cancel"]), ("DELAY15", pri["delay15"]), ("MEANDELAY", pri["mean_delay"])):
                vals.setdefault(f"H_{lvl}_{w}_{m}", p)
    return vals, all(found.values())


def build_features(raw, weather, history, feature_names, cfg):
    """One-row DataFrame with exactly the training feature layout.
    raw: date (datetime), airline, origin, destination (IATA), crs_dep/crs_arr (hhmm ints), distance_miles
    weather: dict with D1_* / DEST_D1_* feature names (None values = missing) ; history: dict with H_* feature names"""
    day = raw["date"]
    doy = day.timetuple().tm_yday
    to_h, since_h = holiday_features(day)
    route = cfg["route_counts"].get(f"{raw['origin']}-{raw['destination']}", 0)
    f = {
        "MONTH_SIN": np.sin(2 * np.pi * day.month / 12), "MONTH_COS": np.cos(2 * np.pi * day.month / 12), "DAY_OF_WEEK": float(day.weekday()),
        "DOY_SIN": np.sin(2 * np.pi * doy / 366), "DOY_COS": np.cos(2 * np.pi * doy / 366),
        "DAYS_TO_HOLIDAY": to_h, "DAYS_SINCE_HOLIDAY": since_h,
        "CRS_DEP_MIN": float(hhmm_to_min(raw["crs_dep"])), "CRS_ARR_MIN": float(hhmm_to_min(raw["crs_arr"])),
        "DISTANCE": float(raw["distance_miles"]), "ROUTE_COUNT_LOG_TRAIN": float(np.log1p(route)),
    }
    f.update({k: (np.nan if v is None else v) for k, v in weather.items()})
    f.update(history)
    row = {n: f.get(n, np.nan) for n in feature_names if n not in ("AIRLINE_CODE", "ORIGIN", "DEST")}
    X = pd.DataFrame([row])
    for col, val in (("AIRLINE_CODE", raw["airline"]), ("ORIGIN", raw["origin"]), ("DEST", raw["destination"])):
        X[col] = pd.Categorical([val], categories=cfg["cats"][col])
    return X[feature_names]


def predict_all(X_by_model):
    S = load_artifacts()
    out = {}
    for name, X in X_by_model.items():
        booster = S["models"][name]
        best = booster.attr("best_iteration")                 # early stopping: only the first best_iteration+1 trees are used (as in the evaluation)
        rng = (0, int(best) + 1) if best is not None else (0, 0)
        out[name] = booster.predict(DMatrix(X, enable_categorical=True), iteration_range=rng)
    return out


# ------------------------------------------------------------------------------------------------ routes
def check_key():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    return None


@app.route("/predict", methods=["POST"])
def predict():
    denied = check_key()
    if denied:
        return denied
    t0 = time.time()
    try:
        S = load_artifacts()
        cfg = S["cfg"]
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        try:
            day = datetime.strptime(str(data["date"]), "%Y-%m-%d")
        except Exception:
            return jsonify({"error": "date must be 'YYYY-MM-DD'"}), 400
        if day.year < 2019:
            return jsonify({"error": "date must be 2019-01-01 or later"}), 400
        airline = str(data.get("airline", "")).upper().strip()
        o_code = str(data.get("origin_airport", "")).upper().strip()
        d_code = str(data.get("destination_airport", "")).upper().strip()
        ap = airports()
        if o_code not in ap or d_code not in ap or o_code == d_code:
            return jsonify({"error": "origin_airport and destination_airport must be two different valid IATA airport codes"}), 400
        try:
            crs_dep, crs_arr = parse_time(data.get("departure_time"), "departure_time"), parse_time(data.get("arrival_time"), "arrival_time")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        miles = great_circle_miles(ap[o_code], ap[d_code])            # computed here: the front end sends kilometres, the model was trained on statute miles

        warnings = []
        for label, val, key in (("airline", airline, "AIRLINE_CODE"), ("origin airport", o_code, "ORIGIN"), ("destination airport", d_code, "DEST")):
            if val not in cfg["cats"][key]:
                warnings.append(f"{label} '{val}' was not seen in the training period; it is treated as missing")

        prev_day = day - timedelta(days=1)
        w_o, w_d = fetch_weather(o_code, prev_day), fetch_weather(d_code, prev_day)
        weather = {f"D1_{v}": (w_o or {}).get(v) for v in WEATHER_VARS}
        weather.update({f"DEST_D1_{v}": (w_d or {}).get(v) for v in DEST_WEATHER_VARS})
        weather_ok = bool(w_o)
        if prev_day.date() >= datetime.utcnow().date():
            warnings.append("weather of the day before the flight is not yet observed (future flight date); the models treat it as missing")
        elif not weather_ok:
            warnings.append("no station weather found for the origin airport on the previous day; treated as missing")
        hist, hist_ok = history_for(day, o_code, airline)
        if not hist_ok:
            warnings.append("recent operational history (D-1 / D-7) is only available for 2023-01-01..2024-01-31 in this prototype; neutral training-period averages are used")

        raw = dict(date=day, airline=airline, origin=o_code, destination=d_code, crs_dep=crs_dep, crs_arr=crs_arr, distance_miles=miles)
        X = {n: build_features(raw, weather, hist, cfg["models"]["delay" if n == "delay" else n]["features"], cfg) for n in ("cancel", "code", "delay")}
        P = predict_all(X)
        p_cancel = float(P["cancel"][0])
        cause = P["code"][0]
        dly = P["delay"][0]
        thr = cfg["thresholds"]["cancel_alert_f1_validation"]
        base = cfg["priors"]["cancel"]
        names = cfg["classes"]["delay"]
        result = {
            "predictions": {
                "cancelled_probability": {"cancelled": p_cancel, "not_cancelled": 1 - p_cancel},
                "cancel_alert": bool(p_cancel >= thr),
                "alert_threshold": thr,
                "base_rate": base,
                "relative_risk": p_cancel / base,
                "cancellation_cause_if_cancelled": {c: float(p) for c, p in zip(cfg["classes"]["cause"], cause)},
                "delay": {
                    "delay_class": names[int(np.argmax(dly))],
                    "delay_probabilities": {n: float(p) for n, p in zip(names, dly)},
                    "probability_delay_over_15_min": float(dly[2] + dly[3]),
                },
            },
            "inputs": {"mode": "A", "weather_available": weather_ok, "history_available": hist_ok, "distance_miles": round(miles, 1), "weather_day": prev_day.strftime("%Y-%m-%d")},
            "warnings": warnings,
            "model_info": {
                "version": cfg["version"], "evaluation_period": cfg["split"]["test"],
                "cancel_test_pr_auc": cfg["test_metrics"]["cancel"]["pr_auc"], "cancel_test_roc_auc": cfg["test_metrics"]["cancel"]["roc_auc"], "no_skill_pr_auc": cfg["test_metrics"]["prevalence_test"],
                "delay_test_accuracy": cfg["test_metrics"]["delay"]["accuracy"], "delay_majority_baseline": cfg["test_metrics"]["delay_baseline"],
                "note": "Risk indicator based on recent observations; not a weather forecast and not a guarantee. Probabilities are not post-processed.",
            },
            "latency_ms": round(1000 * (time.time() - t0), 1),
        }
        return jsonify(result)
    except Exception as e:                                         # never leak stack traces to clients
        app.logger.exception("prediction failed")
        return jsonify({"error": "prediction failed", "detail": type(e).__name__}), 500


@app.route("/airlines", methods=["GET"])
def get_airlines():
    return jsonify({"airlines": load_artifacts()["cfg"]["cats"]["AIRLINE_CODE"]})


@app.route("/airports", methods=["GET"])
def get_airports():
    return jsonify({"airports": load_artifacts()["cfg"]["cats"]["ORIGIN"]})


@app.route("/cities", methods=["GET"])
def get_cities():
    known = airports()
    names = {known[c]["city"] for c in load_artifacts()["cfg"]["cats"]["ORIGIN"] if c in known and known[c].get("city")}
    return jsonify({"cities": sorted(names)})


@app.route("/model-info", methods=["GET"])
def model_info():
    cfg = load_artifacts()["cfg"]
    return jsonify({k: cfg[k] for k in ("version", "mode", "split", "priors", "thresholds", "test_metrics", "history_range")})


@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "version": load_artifacts()["cfg"]["version"]})


if __name__ == "__main__":
    load_artifacts()
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5050)), debug=False, threaded=True)
