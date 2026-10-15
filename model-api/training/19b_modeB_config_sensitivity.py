"""Sensitivity check for the Mode B cancellation model (reported in the article, Section 4.4.3): the configuration selected for the Mode A model
(unweighted, depth 4) is fitted on the Mode B inputs and compared with the validation-selected Mode B configuration. This is NOT used for model selection.
Output: outputs/19b_modeB_config_sensitivity.json"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C, fe
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

DF = fe.load_v2("B"); cats = fe.train_cats(DF)
d, _ = fe.task_frame(DF, "cancel")
tr, va, te = d[d.SPLIT == "train"], d[d.SPLIT == "validation"], d[d.SPLIT == "test"]
out = {}
for label, tag in [("config_selected_for_ModeA_applied_to_ModeB_inputs", "cancel"), ("config_selected_for_ModeB", "cancel_B")]:
    best = json.load(open(C.OUT / f"13_{tag}_best.json")); cols = fe.SETS_B[fe.FULL_B] + fe.CATS
    p = dict(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1, random_state=42, early_stopping_rounds=50, n_estimators=800,
             objective="binary:logistic", eval_metric="aucpr", scale_pos_weight=best["scale_pos_weight"], **best["params"])
    m = XGBClassifier(**p).fit(fe.matrix(tr, cols, cats), tr.y.values, eval_set=[(fe.matrix(va, cols, cats), va.y.values)], verbose=False)
    pv, pt = m.predict_proba(fe.matrix(va, cols, cats))[:, 1], m.predict_proba(fe.matrix(te, cols, cats))[:, 1]
    out[label] = dict(params=best["params"], scale_pos_weight=best["scale_pos_weight"], trees=int(m.best_iteration), val_pr_auc=float(average_precision_score(va.y, pv)),
                      test_pr_auc=float(average_precision_score(te.y, pt)), test_roc_auc=float(roc_auc_score(te.y, pt)))
    print(label, out[label], flush=True)
json.dump(out, open(C.OUT / "19b_modeB_config_sensitivity.json", "w"), indent=1)
print("done", flush=True)
