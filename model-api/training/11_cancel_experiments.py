"""Cancellation model: DOT-only natural data, fit on TRAIN, compare balancing strategies on VALIDATION PR-AUC. TEST IS NOT TOUCHED."""
import sys, time, json, numpy as np, pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix,
                             precision_recall_fscore_support, balanced_accuracy_score, f1_score)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as C

df = C.load()
tr, va = df[df.SPLIT == "train"], df[df.SPLIT == "validation"]
ytr, yva = tr.CANCELLED.values, va.CANCELLED.values
print(f"train {len(tr):,} (pos {ytr.mean():.4f}) | validation {len(va):,} (pos {yva.mean():.4f}) | test rows held out: {(df.SPLIT=='test').sum():,}", flush=True)

PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=2.0, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1, random_state=42,
              eval_metric="aucpr", early_stopping_rounds=40)

def evaluate(y, p):
    ap, auc = average_precision_score(y, p), roc_auc_score(y, p)
    pr, rc, th = precision_recall_curve(y, p); f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    i = int(np.nanargmax(f[:-1])); t = th[i]
    out = dict(pr_auc=ap, roc_auc=auc, thr_bestF1=float(t))
    for name, thr in [("bestF1", t), ("0.5", 0.5)]:
        yh = (p >= thr).astype(int); P, R, F, _ = precision_recall_fscore_support(y, yh, labels=[0, 1], zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        out.update({f"{name}_prec0": P[0], f"{name}_rec0": R[0], f"{name}_f1_0": F[0], f"{name}_prec1": P[1], f"{name}_rec1": R[1], f"{name}_f1_1": F[1],
                    f"{name}_macroF1": f1_score(y, yh, average="macro"), f"{name}_balAcc": balanced_accuracy_score(y, yh),
                    f"{name}_TN": tn, f"{name}_FP": fp, f"{name}_FN": fn, f"{name}_TP": tp})
    return out

results = []
for weather in [True, False]:
    st = C.fit_stats(tr)
    Xtr0, num = C.make_features(tr, st, weather); Xva0, _ = C.make_features(va, st, weather)
    Xtr, Xva = C.finalize(Xtr0, num, st, fit=True), C.finalize(Xva0, num, st)
    for strategy in ["none", "class_weight", "smotenc_train_only"]:
        t0 = time.time(); X, y, p = Xtr, ytr, dict(PARAMS)
        if strategy == "class_weight":
            p["scale_pos_weight"] = float((ytr == 0).sum() / (ytr == 1).sum())
        if strategy == "smotenc_train_only":
            cat_idx = [Xtr0.columns.get_loc(c + "_C") for c in C.CATS]
            Xi = Xtr0.copy(); Xi[num] = Xi[num].fillna(st["medians"])
            sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=0.10, k_neighbors=5, random_state=42)
            Xr, yr = sm.fit_resample(Xi.values, ytr)
            Xr = pd.DataFrame(Xr, columns=Xi.columns)
            for c in Xr.columns:
                Xr[c] = Xr[c].astype(int) if c.endswith("_C") else Xr[c].astype(float)
            X = C.finalize(Xr, num, st); y = yr
            print(f"   SMOTENC: {len(Xtr):,} -> {len(X):,} rows (synthetic {len(X)-len(Xtr):,}), pos rate {yr.mean():.3f}", flush=True)
        m = XGBClassifier(**p).fit(X, y, eval_set=[(Xva, yva)], verbose=False)
        pv = m.predict_proba(Xva)[:, 1]
        r = dict(weather=weather, strategy=strategy, best_iter=int(m.best_iteration), seconds=round(time.time() - t0), **evaluate(yva, pv)); results.append(r)
        print(f"weather={weather!s:5} {strategy:20} PR-AUC={r['pr_auc']:.4f} ROC-AUC={r['roc_auc']:.4f} | bestF1 thr={r['thr_bestF1']:.3f} "
              f"P1={r['bestF1_prec1']:.3f} R1={r['bestF1_rec1']:.3f} F1_1={r['bestF1_f1_1']:.3f} macroF1={r['bestF1_macroF1']:.3f} balAcc={r['bestF1_balAcc']:.3f} ({r['seconds']}s)", flush=True)
        pd.DataFrame(results).to_csv(C.OUT / "11_cancel_validation_results.csv", index=False)
print("done", flush=True)
