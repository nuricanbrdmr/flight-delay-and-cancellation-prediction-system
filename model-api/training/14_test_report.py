"""FINAL evaluation: the models fitted in 13_train_task.py are applied ONCE to the held-out TEST period (2023-01..2024-01).
Thresholds / class decisions were fixed on VALIDATION. Also: bootstrap CIs, monthly breakdown, cost-threshold table, SHAP (xgboost pred_contribs), VIF/correlation, figures.
Outputs: outputs/14_test_results.json, 14_*.csv, figures/*.png"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C, fe
from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve, precision_recall_fscore_support, confusion_matrix,
                             f1_score, accuracy_score, balanced_accuracy_score)
from statsmodels.stats.outliers_influence import variance_inflation_factor

FIG = C.OUT / "figures"
FIG.mkdir(exist_ok=True)
MODELS = C.D / "models_v2"
DFS = {"A": fe.load_v2("A"), "B": fe.load_v2("B")}
df = DFS["A"]
cats = fe.train_cats(df)
res = {}
rng = np.random.RandomState(123)
NB = 100


def boot(fn, y, P, n=NB):
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        try: vals.append(fn(y[b], P[b]))
        except Exception: pass
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def load(tag):
    best = json.load(open(C.OUT / f"13_{tag}_best.json"))
    m = XGBClassifier(); m.load_model(MODELS / f"{tag}.json")
    d, names = fe.task_frame(DFS[best.get("mode", "A")], best["task"])
    va, te = d[d.SPLIT == "validation"], d[d.SPLIT == "test"]
    cols = best["features"]
    Xva, Xte = fe.matrix(va, cols, cats), fe.matrix(te, cols, cats)
    pv, pt = m.predict_proba(Xva), m.predict_proba(Xte)
    if pv.ndim == 1: pv, pt = np.c_[1 - pv, pv], np.c_[1 - pt, pt]
    return m, best, names, va, te, Xva, Xte, pv, pt


def shap_bar(m, X, task, K, title):
    s = X.sample(min(30000, len(X)), random_state=0)
    c = m.get_booster().predict(DMatrix(s, enable_categorical=True), pred_contribs=True)
    a = np.abs(c[..., :-1]).mean(axis=(0, 1)) if c.ndim == 3 else np.abs(c[:, :-1]).mean(0)
    imp = pd.Series(a, index=X.columns).sort_values(ascending=False)
    imp.to_csv(C.OUT / f"14_{task}_shap_mean_abs.csv", header=["mean_abs_shap"])
    top = imp.head(15)[::-1]
    plt.figure(figsize=(7, 5)); plt.barh(top.index, top.values, color="#3b6ea5"); plt.xlabel("mean |SHAP value| (test sample, n=%d)" % len(s)); plt.title(title); plt.tight_layout()
    plt.savefig(FIG / f"shap_{task}.png", dpi=200); plt.close()
    return imp


# ======================================= CANCELLATION (binary) ======================================================
def cm_plot(cm, labels, title, fn):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)): ax.text(j, i, f"{cm[i, j]:,}" + chr(10) + f"({cmn[i, j]:.0%})", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > .5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title); plt.tight_layout(); plt.savefig(FIG / fn, dpi=200); plt.close()


def binary(tag):
    m, best, names, va, te, Xva, Xte, pv, pt = load(tag)
    yv, yt = va.y.values, te.y.values
    pr_v, rc_v, th_v = precision_recall_curve(yv, pv[:, 1])
    f_v = 2 * pr_v * rc_v / np.clip(pr_v + rc_v, 1e-12, None)
    thr_f1 = float(th_v[int(np.nanargmax(f_v[:-1]))])
    def bin_report(y, p, thr):
        yh = (p >= thr).astype(int)
        P, R, F, S = precision_recall_fscore_support(y, yh, labels=[0, 1], zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        return dict(threshold=float(thr), precision=P.tolist(), recall=R.tolist(), f1=F.tolist(), macro_f1=float(f1_score(y, yh, average="macro")),
                    balanced_accuracy=float(balanced_accuracy_score(y, yh)), accuracy=float(accuracy_score(y, yh)), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    ap_t, auc_t = float(average_precision_score(yt, pt[:, 1])), float(roc_auc_score(yt, pt[:, 1]))
    res[tag] = dict(n_test=len(yt), prevalence_test=float(yt.mean()), prevalence_val=float(yv.mean()),
                         val=dict(pr_auc=float(average_precision_score(yv, pv[:, 1])), roc_auc=float(roc_auc_score(yv, pv[:, 1]))),
                         test=dict(pr_auc=ap_t, pr_auc_ci95=boot(lambda y, p: average_precision_score(y, p), yt, pt[:, 1]),
                                   roc_auc=auc_t, roc_auc_ci95=boot(lambda y, p: roc_auc_score(y, p), yt, pt[:, 1])),
                         at_val_bestF1_threshold=bin_report(yt, pt[:, 1], thr_f1),
                         at_0p5=bin_report(yt, pt[:, 1], 0.5))
    print("cancel test:", {k: res[tag]["test"][k] for k in ["pr_auc", "pr_auc_ci95", "roc_auc", "roc_auc_ci95"]}, "prevalence", yt.mean(), flush=True)

    # cost-sensitive threshold (relative units: cost of a false alarm = 1, cost of a missed cancellation = r). Chosen on VALIDATION, evaluated on TEST.
    rows = []
    for r in [2, 5, 10, 20, 50]:
        ths = np.unique(np.quantile(pv[:, 1], np.linspace(0.5, 0.9995, 400)))
        costs = [((pv[:, 1] >= t) & (yv == 0)).sum() + r * ((pv[:, 1] < t) & (yv == 1)).sum() for t in ths]
        t = float(ths[int(np.argmin(costs))]); yh = (pt[:, 1] >= t)
        fp, fn, tp = int((yh & (yt == 0)).sum()), int((~yh & (yt == 1)).sum()), int((yh & (yt == 1)).sum())
        cost, no_alert, all_alert = fp + r * fn, r * int(yt.sum()), int((yt == 0).sum())
        rows.append(dict(cost_ratio_FN_to_FP=r, threshold_from_validation=round(t, 4), alert_rate=round(float(yh.mean()), 4), recall=round(tp / yt.sum(), 4),
                         precision=round(tp / max(tp + fp, 1), 4), FP=fp, FN=fn, cost_model=cost, cost_no_alert=no_alert, cost_alert_all=all_alert,
                         saving_vs_no_alert_pct=round(100 * (no_alert - cost) / no_alert, 2), cost_per_1000_flights=round(1000 * cost / len(yt), 2)))
    pd.DataFrame(rows).to_csv(C.OUT / f"14_{tag}_cost_threshold.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    # monthly
    mon = []
    for p, g in te.assign(p=pt[:, 1]).groupby(te.FL_DATE.dt.to_period("M")):
        if g.y.nunique() > 1: mon.append(dict(month=str(p), n=len(g), prevalence=g.y.mean(), pr_auc=average_precision_score(g.y, g.p), roc_auc=roc_auc_score(g.y, g.p)))
    pd.DataFrame(mon).to_csv(C.OUT / f"14_{tag}_monthly.csv", index=False)

    pr_t, rc_t, _ = precision_recall_curve(yt, pt[:, 1])
    plt.figure(figsize=(5.5, 4.5)); plt.plot(rc_t, pr_t, color="#3b6ea5", label=f"XGBoost (PR-AUC={ap_t:.3f})"); plt.axhline(yt.mean(), ls="--", color="grey", label=f"no-skill ({yt.mean():.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Cancellation prediction ({tag}) - test period"); plt.legend(); plt.tight_layout(); plt.savefig(FIG / f"pr_curve_{tag}.png", dpi=200); plt.close()
    shap_bar(m, Xte, tag, 2, f"Cancellation model ({tag}) - global SHAP importance")
    cm = confusion_matrix(yt, (pt[:, 1] >= thr_f1).astype(int))


    def cm_plot(cm, labels, title, fn):
        cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(5.2, 4.4)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)): ax.text(j, i, f"{cm[i, j]:,}\n({cmn[i, j]:.0%})", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > .5 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title); plt.tight_layout(); plt.savefig(FIG / fn, dpi=200); plt.close()
    cm_plot(cm, ["Not cancelled", "Cancelled"], f"Cancellation {tag} - test (val-tuned thr.)", f"cm_{tag}.png")



for _t in ['cancel', 'cancel_B']:
    binary(_t)

# ======================================= multi-class tasks ========================================================
def multi(tag, title, labels):
    m, best, names, va, te, Xva, Xte, pv, pt = load(tag)
    task = best["task"]
    yv, yt = va.y.values, te.y.values
    yh = pt.argmax(1); K = len(names)
    P, R, F, S = precision_recall_fscore_support(yt, yh, labels=list(range(K)), zero_division=0)
    maj = int(np.bincount(va.y).argmax())     # majority class according to VALIDATION (no test peeking)
    out = dict(n_test=len(yt), classes=names, test_class_counts=np.bincount(yt, minlength=K).tolist(),
               val=dict(accuracy=float(accuracy_score(yv, pv.argmax(1))), macro_f1=float(f1_score(yv, pv.argmax(1), average="macro"))),
               test=dict(accuracy=float(accuracy_score(yt, yh)), accuracy_ci95=boot(lambda y, p: accuracy_score(y, p.argmax(1)), yt, pt),
                         macro_f1=float(f1_score(yt, yh, average="macro")), macro_f1_ci95=boot(lambda y, p: f1_score(y, p.argmax(1), average="macro"), yt, pt),
                         balanced_accuracy=float(balanced_accuracy_score(yt, yh)),
                         macro_pr_auc=float(np.mean([average_precision_score((yt == k).astype(int), pt[:, k]) for k in range(K)])),
                         macro_roc_auc_ovr=float(np.mean([roc_auc_score((yt == k).astype(int), pt[:, k]) for k in range(K)]))),
               baseline_majority_class=names[maj], baseline_majority_accuracy_test=float((yt == maj).mean()),
               per_class=[dict(cls=names[k], precision=float(P[k]), recall=float(R[k]), f1=float(F[k]), support=int(S[k])) for k in range(K)],
               confusion_matrix=confusion_matrix(yt, yh, labels=list(range(K))).tolist())
    if task == "delay":   # DOT definition: delayed = departure delay > 15 min = classes 2+3
        s_t, s_v = pt[:, 2] + pt[:, 3], pv[:, 2] + pv[:, 3]
        out["delay_gt15_binary_test"] = dict(pr_auc=float(average_precision_score(yt >= 2, s_t)), roc_auc=float(roc_auc_score(yt >= 2, s_t)), prevalence=float((yt >= 2).mean()))
    mon = []
    for p, g in te.assign(pred=yh).groupby(te.FL_DATE.dt.to_period("M")):
        mon.append(dict(month=str(p), n=len(g), accuracy=accuracy_score(g.y, g.pred), macro_f1=f1_score(g.y, g.pred, average="macro"), majority_baseline=(g.y == maj).mean()))
    pd.DataFrame(mon).to_csv(C.OUT / f"14_{tag}_monthly.csv", index=False)
    cm_plot(np.array(out["confusion_matrix"]), labels, title, f"cm_{tag}.png")
    shap_bar(m, Xte, tag, K, f"{title} ({tag}) - SHAP importance")
    res[tag] = out
    print(tag, "test:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["test"].items()}, "majority-baseline acc:", round(out["baseline_majority_accuracy_test"], 4), flush=True)

for _t in ["code", "code_B"]:
    multi(_t, "Cancellation cause", ["A Carrier", "B Weather", "C NAS"])
for _t in ["delay", "delayacc", "delay_B", "delayacc_B"]:
    multi(_t, "Departure delay class", ["<=0 min", "1-15 min", "16-30 min", ">30 min"])

# ======================================= multicollinearity (train, final feature set) ================================
cols = fe.SETS_B[fe.FULL_B]
tr = DFS["B"][DFS["B"].SPLIT == "train"]
Z = tr[cols].sample(150000, random_state=0)
Z = Z.fillna(Z.median())
Z = (Z - Z.mean()) / Z.std().replace(0, 1)
vif = pd.DataFrame({"feature": cols, "VIF": [variance_inflation_factor(Z.values, i) for i in range(len(cols))]}).sort_values("VIF", ascending=False)
vif.to_csv(C.OUT / "14_vif_train.csv", index=False)
corr = Z.corr()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if abs(corr.loc[a, b]) >= 0.8]
pd.DataFrame(pairs, columns=["feature_a", "feature_b", "pearson_r"]).sort_values("pearson_r", key=abs, ascending=False).to_csv(C.OUT / "14_high_correlation_pairs.csv", index=False)
fig, ax = plt.subplots(figsize=(11, 9)); im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=6); ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=6)
plt.colorbar(im, fraction=0.03); plt.title("Pearson correlation of model features (train sample)"); plt.tight_layout(); plt.savefig(FIG / "correlation_matrix.png", dpi=200); plt.close()
res["multicollinearity"] = dict(n_features=len(cols), vif_gt10=int((vif.VIF > 10).sum()), vif_gt5=int((vif.VIF > 5).sum()), max_vif=float(vif.VIF.max()), n_pairs_abs_r_ge_0p8=len(pairs))
print("VIF>10:", res["multicollinearity"]["vif_gt10"], "max", round(vif.VIF.max(), 1), "pairs |r|>=.8:", len(pairs), flush=True)

json.dump(res, open(C.OUT / "14_test_results.json", "w"), indent=1)
print("done", flush=True)
