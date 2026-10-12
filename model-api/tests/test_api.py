"""Tests of the SmartTicket model API. Run from model-api/:  python -m unittest discover -s tests -v
No network is needed: weather retrieval is patched. The parity test proves that the API builds exactly the training features."""
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cancel_delay_api as api  # noqa: E402

SAMPLE = json.load(open(ROOT / "tests" / "parity_sample.json"))
WX = {"tmin": 2.0, "tmax": 9.0, "tavg": 5.0, "prcp": 1.2, "snow": 0.0, "wspd": 14.0, "pres": 1012.0}
BODY = dict(date="2023-02-03", airline="UA", origin_airport="ORD", destination_airport="LGA", departure_time="07:30", arrival_time="10:45")


class Parity(unittest.TestCase):
    def test_api_features_reproduce_training_probabilities(self):
        S = api.load_artifacts(); cfg = S["cfg"]
        worst = 0.0
        for s in SAMPLE:
            r = s["raw"]
            raw = dict(date=datetime.strptime(r["date"], "%Y-%m-%d"), airline=r["airline"], origin=r["origin"], destination=r["destination"], crs_dep=r["crs_dep"], crs_arr=r["crs_arr"], distance_miles=r["distance_miles"])
            X = {n: api.build_features(raw, s["weather"], s["history"], cfg["models"][n]["features"], cfg) for n in ("cancel", "code", "delay")}
            P = api.predict_all(X)
            exp = s["expected"]
            worst = max(worst, abs(float(P["cancel"][0]) - exp["cancel"]), float(np.abs(P["code"][0] - np.array(exp["cause"])).max()), float(np.abs(P["delay"][0] - np.array(exp["delay"])).max()))
        self.assertLess(worst, 1e-5, f"max abs difference {worst}")


class Endpoints(unittest.TestCase):
    def setUp(self):
        api.app.config["TESTING"] = True
        self.c = api.app.test_client()
        self.patch = mock.patch.object(api, "fetch_weather", side_effect=lambda iata, day: dict(WX) if day.date() < datetime.utcnow().date() else None)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def post(self, **over):
        return self.c.post("/predict", json={**BODY, **over})

    def test_schema_and_probabilities(self):
        r = self.post(); self.assertEqual(r.status_code, 200)
        p = r.get_json()["predictions"]
        self.assertAlmostEqual(sum(p["cancelled_probability"].values()), 1.0, places=6)
        self.assertAlmostEqual(sum(p["delay"]["delay_probabilities"].values()), 1.0, places=5)
        self.assertAlmostEqual(sum(p["cancellation_cause_if_cancelled"].values()), 1.0, places=5)
        self.assertAlmostEqual(p["delay"]["probability_delay_over_15_min"], p["delay"]["delay_probabilities"]["delay_16_30"] + p["delay"]["delay_probabilities"]["delay_over_30"], places=6)
        j = r.get_json(); self.assertTrue(j["inputs"]["weather_available"]); self.assertTrue(j["inputs"]["history_available"]); self.assertEqual(j["inputs"]["mode"], "A")

    def test_no_post_hoc_adjustment_and_deterministic(self):
        a, b = self.post().get_json()["predictions"], self.post().get_json()["predictions"]
        self.assertEqual(a["cancelled_probability"], b["cancelled_probability"])
        self.assertNotIn("model_adjustments", a); self.assertNotIn("corrections_applied", json.dumps(a))
        # a summer, daytime, short-haul flight of a large carrier must not be rescaled by any rule: compare with the raw model output
        r = self.post(date="2023-07-12", airline="DL", departure_time="12:00", arrival_time="13:10", origin_airport="ATL", destination_airport="MCO").get_json()
        S = api.load_artifacts(); cfg = S["cfg"]
        day = datetime(2023, 7, 12); hist, _ = api.history_for(day, "ATL", "DL")
        weather = {**{f"D1_{k}": v for k, v in WX.items()}, **{f"DEST_D1_{k}": WX[k] for k in api.DEST_WEATHER_VARS}}
        ap = api.airports(); miles = api.great_circle_miles(ap["ATL"], ap["MCO"])
        raw = dict(date=day, airline="DL", origin="ATL", destination="MCO", crs_dep=1200, crs_arr=1310, distance_miles=miles)
        direct = float(api.predict_all({"cancel": api.build_features(raw, weather, hist, cfg["models"]["cancel"]["features"], cfg)})["cancel"][0])
        self.assertAlmostEqual(r["predictions"]["cancelled_probability"]["cancelled"], direct, places=6)

    def test_future_date_warns_and_uses_missing_weather(self):
        j = self.post(date="2031-05-05").get_json()
        self.assertFalse(j["inputs"]["weather_available"]); self.assertFalse(j["inputs"]["history_available"])
        self.assertTrue(any("not yet observed" in w for w in j["warnings"])); self.assertTrue(any("history" in w for w in j["warnings"]))

    def test_unseen_airline_is_reported(self):
        j = self.post(airline="ZZ").get_json()
        self.assertTrue(any("not seen" in w for w in j["warnings"]))

    def test_validation_errors(self):
        for bad in (dict(date="03/02/2023"), dict(origin_airport="XXX"), dict(destination_airport="ORD"), dict(departure_time="7h30"), dict(date="2010-01-01")):
            self.assertEqual(self.post(**bad).status_code, 400, bad)
        self.assertEqual(self.c.post("/predict", data="not json", content_type="application/json").status_code, 400)

    def test_distance_is_computed_in_miles(self):
        j = self.post().get_json()
        self.assertTrue(600 < j["inputs"]["distance_miles"] < 800)          # ORD-LGA is about 733 miles

    def test_helpers_endpoints(self):
        self.assertEqual(self.c.get("/test").status_code, 200)
        self.assertIn("UA", self.c.get("/airlines").get_json()["airlines"])
        self.assertIn("ORD", self.c.get("/airports").get_json()["airports"])
        self.assertIn("test_metrics", self.c.get("/model-info").get_json())


if __name__ == "__main__":
    unittest.main()
