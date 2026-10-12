# model-api

| Path | Purpose |
|---|---|
| `cancel_delay_api.py` | SmartTicket model API v2 (Flask). Serves the validated Mode-A models. |
| `models_v2/` | Exported models (`*.ubj`), `config.json`, history lookup tables, weather-station map. Created by `revision/20_export_api_artifacts.py`. |
| `tests/` | `python -m unittest discover -s tests -v` (feature parity with training, schema, validation, no post-hoc adjustment); `benchmark_latency.py` measures response times. |
| `revision/` | Reproducible experiments for the manuscript revision (data rebuild, features, models, test report, tables). See `revision/OZET.md`. |
| `cancel_model.py`, `cancel_code_model.py`, `delay_model.py`, `models/` | LEGACY training code and models, superseded by `revision/` (kept for the record; do not use). |

## Run the API
```
pip install -r requirements.txt
python cancel_delay_api.py            # http://127.0.0.1:5050
```
Environment variables: `PORT`, `HOST`, `CORS_ORIGINS`, `MODEL_API_KEY` (if set, requests must send header `X-API-Key`), `MODEL_ARTIFACTS`.

Example request:
```
POST /predict
{"date":"2023-02-03","airline":"UA","origin_airport":"ORD","destination_airport":"LGA","departure_time":"07:30","arrival_time":"10:45"}
```
The response contains the cancellation probability, an alert flag (threshold fixed on the validation period), cause probabilities (conditional on a cancellation), the four delay-class probabilities, the inputs that were actually available (`weather_available`, `history_available`) and `warnings`.

## What the API does and does not do
* Mode A only: calendar/holiday, schedule, observed weather of the previous day (Meteostat stations), recent operational history (available for 2023-01-01..2024-01-31; otherwise neutral training averages, reported in `warnings`).
* The output is a risk indicator, not a weather forecast; probabilities are not post-processed (the earlier ad-hoc 30-70% reductions were removed).
* Mode B (same-day state three hours before departure) needs a live data feed and is not implemented.
* The distance is computed from the airport coordinates in statute miles (the model was trained on miles).
