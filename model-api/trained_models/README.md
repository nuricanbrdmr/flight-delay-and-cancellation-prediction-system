# Trained models of the experiments

The final XGBoost models behind the results of the article (fitted on the training period with the hyper-parameters selected on the validation period, number of trees set by early stopping), stored as gzip-compressed XGBoost JSON files so that every file stays far below the 100 MB limit of GitHub. They let you reproduce the test report and the additional analyses (`training/14_test_report.py`, `19_test_breakdown_calibration.py`) without retraining.

| File | Task | Mode | Trees | Class weights | Size (MB, uncompressed) | SHA-256 of the uncompressed JSON |
|---|---|---|---|---|---|---|
| `cancel.json.gz` | cancellation (binary) | A | 217 | none | 2.2 | `0c47751ba616b7724299098914cdeabe75134d8335eb5e5045d777779290eef4` |
| `cancel_B.json.gz` | cancellation (binary) | B | 78 | scale_pos_weight 33.3 | 1.1 | `bde62d238ea2be97fd6215ebc95a5dadb6910053693e68fc489c9e55fa46043e` |
| `code.json.gz` | cancellation cause (A/B/C) | A | 101 | none | 3.1 | `747c3d8587aef807e2e518995e662de44edf6913c41f0b0ac551c656fe710b74` |
| `code_B.json.gz` | cancellation cause (A/B/C) | B | 145 | none | 3.8 | `d56cc4712240aada783959797adece8b70328df0c601806ed30e8963c83ff2fa` |
| `delayacc.json.gz` | departure delay, 4 classes, no class weights | A | 771 | none | 36.9 | `704c96636d74aace714e626803d16532f176321d80a2033ef8c66bf41992c437` |
| `delayacc_B.json.gz` | departure delay, 4 classes, no class weights | B | 786 | none | 39.1 | `7feb8e779426038bc596146ebc36f4d93cb679e9d734f5574a7e8f11f9d14202` |
| `delay.json.gz` | departure delay, 4 classes, class-weighted | A | 798 | balanced | 145.5 | `98502464643c658982a60ce496fe75640d0c2440d16635dd6ea0cdc35216eaec` |
| `delay_B.json.gz` | departure delay, 4 classes, class-weighted | B | 799 | balanced | 143.5 | `90ce8b723a5089858a1140179e4c478aabd648eacd0f55ce361dbd09d5b147dd` |

Mode A = information up to the previous day, Mode B = additionally the same-day state three hours before departure (computed on an absolute UTC time axis). Hyper-parameters and validation metrics of each model are in `../training/outputs/13_<tag>_best.json`. The cancellation model of Mode B uses class weighting, so its raw scores are not calibrated probabilities (see the article, Section 4.1).

## Use

```python
import gzip, xgboost as xgb
m = xgb.XGBClassifier()
m.load_model(bytearray(gzip.open('trained_models/cancel.json.gz').read()))
```

To put all models where the training scripts expect them (`csv/models_v2/<tag>.json`):

```
python -I model-api/training/21_unpack_trained_models.py
```

The feature lists and category lists that the models need are in `../training/outputs/13_<tag>_best.json` (feature order) and are rebuilt from the data by `../training/fe.py`. The API uses the Mode A models exported to `../models_v2/` (`.ubj`).
