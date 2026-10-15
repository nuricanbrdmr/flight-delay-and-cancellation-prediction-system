# SmartTicket: flight cancellation, cancellation-cause and departure-delay risk

Code, trained models and prototype that accompany the article *Predicting Flight Cancellations, Cancellation Causes and Departure Delays from Historical, Meteorological and Same-Day Operational Data, with a Smart Ticketing Prototype* (E. N. Mutlu, N. C. Birdemir, E. E. Ülkü; Marmara University).

The repository contains three parts:

| Folder | Content |
|---|---|
| `model-api/training/` | The experiments: data rebuild, leakage-controlled features, model training and validation-based tuning, one-shot test evaluation, tables and figures of the article. |
| `model-api/trained_models/` | The eight final XGBoost models behind the results of the article (Mode A and B, gzip-compressed JSON, about 60 MB in total). |
| `model-api/` | The model API of the prototype (`cancel_delay_api.py`, Flask), the exported Mode-A models (`models_v2/`) and tests (`tests/`). See [`model-api/README.md`](model-api/README.md). |
| `bitirme-backend/`, `bitirme-frontend/` | The SmartTicket web prototype (Node.js/Express back end, React/Vite front end). The article describes it as a technical-feasibility prototype that has not been evaluated with users. |
| `Proje raporu/` | The article manuscripts (English and Turkish) and earlier project reports. |

## What the models predict

Three related tasks on U.S. domestic flights (Zelaya sample 2019-01 to 2023-08 plus the complete month 2024-01 of Singh; both from the DOT/BTS on-time data):

1. flight cancellation (binary, all flights);
2. cancellation cause (DOT classes A carrier, B weather, C national air system) for cancelled flights;
3. departure-delay class (<=0, 1-15, 16-30, >30 minutes) for operated flights.

Inputs are calendar and schedule data, the weather of the previous completed day at the origin and destination airport (Meteostat station observations, *not* forecasts), recent cancellation and delay rates (previous day / previous seven days) and, in the optional day-of-departure **Mode B**, the operational state observed three hours before the scheduled departure (requires a live data feed). **Mode A** uses information up to the previous day only and is the only mode that is valid for tickets bought in advance.

The protocol is chronological: training 2019-01 to 2022-06, validation 2022-07 to 2022-12 (hyper-parameter search and decision thresholds), test 2023-01 to 2023-08 and 2024-01 (used once). The data keep their natural class distribution. The outputs are risk indicators, not precise forecasts; results are reported in the article.

## Reproducing the experiments

Requirements: Python 3.10+, `pip install -r model-api/training/requirements.txt`. The data are not part of the repository (`csv/` is git-ignored). Download

* `flights_sample_3m_origin.csv`: Zelaya (2023), <https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023>
* `flight_data_2018_2024.csv`: Singh (2024), <https://www.kaggle.com/datasets/shubhamsingh42/flight-delay-dataset-2018-2024>

into `csv/`. Weather is downloaded from Meteostat by the first script. Run from the repository root:

| Step | Script | Result |
|---|---|---|
| 1 | `python -I model-api/training/07_fetch_weather.py` | daily station observations per origin airport |
| 2 | `python -I model-api/training/08_build_dataset.py` | `csv/clean_flights.csv.gz` (de-duplicated, pre-departure fields only, D-1 weather) |
| 3 | `python -I model-api/training/12_features_v2.py` | history features (self-tested for leakage) |
| 4 | `python -I model-api/training/15_live_features.py` | Mode-B same-day features on an absolute (UTC) time axis (self-tested for leakage) |
| 5 | `python -I model-api/training/13_train_task.py {cancel\|code\|delay\|delayacc} {A\|B}` | ablation, random search on the validation period, final fit (`csv/models_v2/`, `outputs/13_*`) |
| 5b | `python -I model-api/training/21_unpack_trained_models.py` | optional shortcut: unpacks the final models of the article from `model-api/trained_models/` (the long training of step 5 can then be skipped) |
| 6 | `python -I model-api/training/14_test_report.py` | one-shot test evaluation, bootstrap intervals, SHAP, VIF (`outputs/14_*`) |
| 7 | `python -I model-api/training/19_test_breakdown_calibration.py` (2023 vs 2024-01, Brier score, calibration), `18_baselines.py`, `17_extra_figures.py`, `18b_descriptive_figures.py`, `16_make_paper_tables.py` | comparison algorithms, figures, tables (`docs/PAPER_TABLES.md`) |
| 8 | `python -I model-api/training/20_export_api_artifacts.py` | exports the Mode-A models and look-up tables used by the API |

Scripts `00`-`06`, `09`, `11` document the data checks (sources, duplicates, class distributions, leakage table, balancing experiment). Random seeds are fixed (42 for XGBoost and SMOTE-NC, 7 for the random search, 0 for subsamples, 123 for the bootstrap).

## Running the prototype

```bash
# 1. model API (http://127.0.0.1:5050)
cd model-api && pip install -r requirements.txt && python cancel_delay_api.py

# 2. back end: copy bitirme-backend/.env.example to .env and fill it in (never commit .env)
cd bitirme-backend && npm install && npm start

# 3. front end (http://localhost:5173)
cd bitirme-frontend && npm install && npm run dev
```

The flight search uses an external flight-search service (RapidAPI key in `.env`); the model API works on its own (see `model-api/README.md`). Tests of the model API: `cd model-api && python -m unittest discover -s tests -v`.

![Search form](model-api/training/outputs/figures/smartticket_home.png)

## Limitations (see the article)

The prototype shows model-based risk indications only for dates covered by its look-up tables; for other dates, missing inputs are reported as warnings and neutral training averages are used. There has been no user study, and no load, scalability or security testing beyond what the article reports.

## License

MIT, see `LICENSE`.
