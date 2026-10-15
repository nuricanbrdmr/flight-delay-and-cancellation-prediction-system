#!/bin/bash
# Full pipeline of the experiments, in the order used for the article. Run from the repository root; the raw data must be in ./csv (see README.md).
# Long steps: 13_train_task.py (about 3 hours in total for all tasks and both modes on a 6-core machine).
set -e
cd "$(dirname "$0")/../.."
PY="python -I"
T=model-api/training

$PY $T/07_fetch_weather.py            # Meteostat daily station observations per origin airport
$PY $T/08_build_dataset.py            # csv/clean_flights.csv.gz
$PY $T/12_features_v2.py              # history features + calendar + destination weather (self-test inside)
$PY $T/15_live_features.py            # Mode-B same-day features on an absolute (UTC) time axis (self-test inside)

for spec in "cancel A" "code A" "delay A" "delayacc A" "cancel B" "code B" "delay B" "delayacc B"; do
  set -- $spec
  $PY $T/13_train_task.py $1 $2 > csv/train_$1_$2.log 2>&1
done

$PY $T/14_test_report.py              # one-shot test evaluation, bootstrap intervals, SHAP, VIF
$PY $T/19_test_breakdown_calibration.py   # 2023 vs January 2024, Brier score and calibration curve
$PY $T/18_baselines.py                # comparison algorithms under the same protocol
$PY $T/17_extra_figures.py
$PY $T/18b_descriptive_figures.py
$PY $T/16_make_paper_tables.py        # docs/PAPER_TABLES.md
$PY $T/20_export_api_artifacts.py     # Mode-A models and look-up tables for the API
