"""usage: python 13_train_task.py <cancel|code|delay>
1) feature-set ablation (default params)  2) limited random search (validation-based)  3) refit best on TRAIN (early stop on VALIDATION).
THE TEST SET IS NOT USED HERE.
Artifacts: outputs/13_<task>_ablation.csv, 13_<task>_search.csv, 13_<task>_best.json ; csv/models_v2/<task>.json ; <task>_valpred.npy"""
import sys, json, time, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C, fe
from xgboost import XGBClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, log_loss)

task_arg = sys.argv[1]                        # cancel | code | delay | delayacc  (delayacc = same data, NO class weighting, selected by accuracy)
mode = sys.argv[2] if len(sys.argv) > 2 else "A"   # A = information up to D-1 ; B = A + same-day state observed T-3h
task = "delay" if task_arg == "delayacc" else task_arg
tag = task_arg + ("_B" if mode == "B" else "")
SUB = {"cancel": None, "code": None, "delay": 700_000}[task]
MODELS = C.D / "models_v2"
MODELS.mkdir(exist_ok=True)
df = fe.load_v2(mode)
cats = fe.train_cats(df)
d, names = fe.task_frame(df, task)
K = len(names)
tr, va = d[d.SPLIT == "train"], d[d.SPLIT == "validation"]
print(f"[{tag}] train {len(tr):,} validation {len(va):,} classes={names}; train dist={np.bincount(tr.y).tolist()} val dist={np.bincount(va.y, minlength=K).tolist()}", flush=True)
tr_s = tr if SUB is None else tr.sample(SUB, random_state=0).sort_values("FL_DATE")
if SUB:
    print(f"   (ablation + search use a {SUB:,}-row random train subsample; final refit uses all train rows)", flush=True)


def metrics(y, P):
    if K == 2:
        p = P[:, 1]
        return dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p))
    yh = P.argmax(1)
    ovr = np.mean([average_precision_score((y == k).astype(int), P[:, k]) for k in range(K)])
    return dict(macro_f1=f1_score(y, yh, average="macro"), accuracy=accuracy_score(y, yh), bal_acc=balanced_accuracy_score(y, yh),
                macro_pr_auc=ovr, logloss=log_loss(y, P, labels=list(range(K))))


SELECT = "pr_auc" if K == 2 else ("accuracy" if task_arg == "delayacc" else "macro_f1")


def weights(y, mode):
    if K == 2 or mode == "none":
        return None
    cnt = np.bincount(y, minlength=K).astype(float)
    w = len(y) / (K * cnt)
    return (w if mode == "balanced" else np.sqrt(w))[y]


def fit(trd, cols, params, wmode, spw=1.0, es=40):
    Xtr, Xva = fe.matrix(trd, cols + fe.CATS, cats), fe.matrix(va, cols + fe.CATS, cats)
    p = dict(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1, random_state=42,
             early_stopping_rounds=es, n_estimators=800, **params)
    if K == 2:
        p.update(objective="binary:logistic", eval_metric="aucpr", scale_pos_weight=spw)
    else:
        p.update(objective="multi:softprob", eval_metric="mlogloss", num_class=K)
    m = XGBClassifier(**p).fit(Xtr, trd.y.values, sample_weight=weights(trd.y.values, wmode),
                               eval_set=[(Xva, va.y.values)], verbose=False)
    P = m.predict_proba(Xva)
    if K == 2 and P.ndim == 1:
        P = np.c_[1 - P, P]
    return m, P


DEFAULT = dict(learning_rate=0.08, max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0)
neg_pos = float((tr.y == 0).sum() / max((tr.y == 1).sum(), 1))
dwmode = "none" if (K == 2 or task_arg == "delayacc") else "balanced"
dspw = float(np.sqrt(neg_pos)) if K == 2 else 1.0

# ---- 1) ablation ------------------------------------------------------------------------------------------------
rows = []
SETS = fe.SETS_B if mode == "B" else fe.SETS
for name, cols in SETS.items():
    t0 = time.time()
    m, P = fit(tr_s, cols, DEFAULT, dwmode, dspw)
    r = dict(feature_set=name, n_features=len(cols) + 3, best_iter=int(m.best_iteration), **metrics(va.y.values, P))
    rows.append(r)
    print(f"  ablation {name:40} " + " ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float)) + f" ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(C.OUT / f"13_{tag}_ablation.csv", index=False)

# ---- 2) limited random search on the full feature set ------------------------------------------------------------
rng = np.random.RandomState(7)
NIT = {"cancel": 12, "code": 20, "delay": 8}[task]
space = dict(learning_rate=lambda: float(np.exp(rng.uniform(np.log(0.03), np.log(0.15)))),
             max_depth=lambda: int(rng.randint(4, 11)),
             min_child_weight=lambda: int(np.exp(rng.uniform(0, np.log(60)))),
             subsample=lambda: float(rng.uniform(0.6, 1.0)),
             colsample_bytree=lambda: float(rng.uniform(0.5, 1.0)),
             reg_lambda=lambda: float(np.exp(rng.uniform(np.log(0.5), np.log(20)))),
             reg_alpha=lambda: float(rng.choice([0, 0.1, 0.5, 1, 3])),
             gamma=lambda: float(rng.choice([0, 0.5, 1, 3])))
cols = SETS[fe.FULL_B] if mode == "B" else SETS[fe.FULL]
srows = []
wchoices = ([("none", 1.0), ("none", float(np.sqrt(neg_pos))), ("none", neg_pos)] if K == 2
            else ([("none", 1.0)] if task_arg == "delayacc" else [("none", 1.0), ("sqrt", 1.0), ("balanced", 1.0)]))
for i in range(NIT):
    prm = {k: f() for k, f in space.items()}
    wm, spw = wchoices[rng.randint(len(wchoices))]
    t0 = time.time()
    m, P = fit(tr_s, cols, prm, wm, spw)
    r = dict(it=i, weight_mode=wm, scale_pos_weight=round(spw, 2), best_iter=int(m.best_iteration),
             **{k: round(v, 4) if isinstance(v, float) else v for k, v in prm.items()}, **metrics(va.y.values, P))
    srows.append(r)
    print(f"  search {i:2d} {SELECT}={r[SELECT]:.4f} depth={prm['max_depth']} lr={prm['learning_rate']:.3f} w={wm}/{spw:.1f} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(srows).to_csv(C.OUT / f"13_{tag}_search.csv", index=False)
s = pd.DataFrame(srows)
best = s.loc[s[SELECT].idxmax()]
print(f"best search config: it={int(best.it)} {SELECT}={best[SELECT]:.4f}", flush=True)
bp = {k: (int(best[k]) if k in ("max_depth", "min_child_weight") else float(best[k])) for k in space}
bw, bspw = str(best.weight_mode), float(best.scale_pos_weight)

# ---- 3) refit on ALL train rows, early stopping on validation ---------------------------------------------------------
t0 = time.time()
m, P = fit(tr, cols, bp, bw, bspw, es=50)
vm = metrics(va.y.values, P)
print("FINAL (train fit, validation metrics) " + " ".join(f"{k}={v:.4f}" for k, v in vm.items()) + f" best_iter={m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
m.save_model(MODELS / f"{tag}.json")
np.save(MODELS / f"{tag}_valpred.npy", P)
json.dump(dict(task=task, tag=tag, mode=mode, params=bp, weight_mode=bw, scale_pos_weight=bspw, best_iteration=int(m.best_iteration),
               features=cols + fe.CATS, val_metrics=vm, search_subsample=SUB, n_search=NIT),
          open(C.OUT / f"13_{tag}_best.json", "w"), indent=1)
print("done", flush=True)
