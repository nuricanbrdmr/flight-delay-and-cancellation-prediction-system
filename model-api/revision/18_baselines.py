"""Baseline algorithms under the SAME protocol as the XGBoost models (same split, same features, test used once, no balancing).
Models: Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting ("Gradient Boosting"). KNN is not run: it is not tractable on 3.5M rows with mixed categorical features.
Baselines use fixed, untuned hyper-parameters and (cancellation/delay) a 1,000,000-row random subsample of the training period; XGBoost was tuned on validation and uses all training rows.
Thresholds (binary task) are fixed on validation. Output: outputs/18_baselines.json"""
import sys, json, time, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C, fe
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve, f1_score, accuracy_score, balanced_accuracy_score)

SUB = 1_000_000
df = fe.load_v2("B")
cats = fe.train_cats(df)
out = {}


def codes(d):
    return np.column_stack([pd.Categorical(d[c], categories=cats[c]).codes for c in fe.CATS]).astype(float)


def build(d, cols, med, onehot_air):
    X = d[cols].copy()
    X = X.fillna(med)
    num = X.values.astype(float)
    if onehot_air:
        air = pd.get_dummies(pd.Categorical(d["AIRLINE_CODE"], categories=cats["AIRLINE_CODE"])).values.astype(float)
        return np.hstack([num, air])
    return np.hstack([num, codes(d)])


def models(K):
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=300, C=1.0),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=100, random_state=0),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=50, max_samples=0.3, n_jobs=-1, random_state=0),
        "Gradient Boosting (HistGB)": lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_depth=6, min_samples_leaf=100, random_state=0),
    }


for task in ["cancel", "code", "delay"]:
    d, names = fe.task_frame(df, task)
    K = len(names)
    tr, va, te = d[d.SPLIT == "train"], d[d.SPLIT == "validation"], d[d.SPLIT == "test"]
    if task != "code" and len(tr) > SUB:
        tr = tr.sample(SUB, random_state=0)
    for mode in ["A", "B"]:
        cols = fe.SETS_B[fe.FULL_B] if mode == "B" else fe.SETS[fe.FULL]
        med = tr[cols].median()
        for name, mk in models(K).items():
            t0 = time.time()
            oh = name == "Logistic Regression"
            Xtr, Xva, Xte = build(tr, cols, med, oh), build(va, cols, med, oh), build(te, cols, med, oh)
            if oh:
                sc = StandardScaler().fit(Xtr); Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
            m = mk().fit(Xtr, tr.y.values)
            pv, pt = m.predict_proba(Xva), m.predict_proba(Xte)
            yv, yt = va.y.values, te.y.values
            if K == 2:
                _, _, th = precision_recall_curve(yv, pv[:, 1])
                prc, rcl, th = precision_recall_curve(yv, pv[:, 1])
                f = 2 * prc * rcl / np.clip(prc + rcl, 1e-12, None)
                thr = float(th[int(np.nanargmax(f[:-1]))]); yh = (pt[:, 1] >= thr).astype(int)
                tp = int(((yh == 1) & (yt == 1)).sum()); fp = int(((yh == 1) & (yt == 0)).sum()); fn = int(((yh == 0) & (yt == 1)).sum())
                r = dict(pr_auc=float(average_precision_score(yt, pt[:, 1])), roc_auc=float(roc_auc_score(yt, pt[:, 1])), val_pr_auc=float(average_precision_score(yv, pv[:, 1])),
                         threshold=thr, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), accuracy=float((yh == yt).mean()))
                r["f1"] = 2 * r["precision"] * r["recall"] / max(r["precision"] + r["recall"], 1e-12)
            else:
                yh = pt.argmax(1)
                r = dict(accuracy=float(accuracy_score(yt, yh)), macro_f1=float(f1_score(yt, yh, average="macro")), balanced_accuracy=float(balanced_accuracy_score(yt, yh)),
                         macro_roc_auc_ovr=float(np.mean([roc_auc_score((yt == k).astype(int), pt[:, k]) for k in range(K)])),
                         val_accuracy=float(accuracy_score(yv, pv.argmax(1))))
                if task == "delay":
                    r["gt15_roc_auc"] = float(roc_auc_score(yt >= 2, pt[:, 2] + pt[:, 3]))
            r["seconds"] = round(time.time() - t0)
            r["n_train"] = int(len(tr))
            out.setdefault(task, {}).setdefault(mode, {})[name] = r
            print(task, mode, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            json.dump(out, open(C.OUT / "18_baselines.json", "w"), indent=1)
print("done", flush=True)
