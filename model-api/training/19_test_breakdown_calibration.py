"""Additional test-period analyses on the SAVED models (no retraining; the test period is not used for any selection):
 (1) all three tasks evaluated separately for the 2023 part (2023-01..2023-08, sampled data) and the January-2024 part (complete month, ~10x denser) of the test period;
 (2) probability quality of the cancellation model: Brier score (vs. a constant prediction), calibration curve, calibration intercept/slope, expected calibration error;
     optional recalibration check (isotonic regression fitted on the VALIDATION period, evaluated on the test period).
Outputs: outputs/19_test_by_period.csv, 19_cancel_calibration.json, 19_cancel_calibration_bins.csv, figures/calibration_cancel.png"""
import sys, json, gc, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C, fe
from xgboost import XGBClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve, precision_recall_fscore_support, accuracy_score,
                             f1_score, balanced_accuracy_score, brier_score_loss)
from sklearn.isotonic import IsotonicRegression
import statsmodels.api as sm

MODELS = C.D / "models_v2"
PERIODS = {"2023-01..2023-08": lambda d: d.FL_DATE < pd.Timestamp("2024-01-01"), "2024-01": lambda d: d.FL_DATE >= pd.Timestamp("2024-01-01"), "all test": lambda d: d.FL_DATE >= pd.Timestamp("2023-01-01"),
           "all test, scheduled dep. >= 06:00 local": lambda d: (d.FL_DATE >= pd.Timestamp("2023-01-01")) & (d.CRS_DEP_MIN >= 360)}
rows, cal = [], {}


def predict(tag, DF, cats):
    best = json.load(open(C.OUT / f"13_{tag}_best.json"))
    m = XGBClassifier(); m.load_model(MODELS / f"{tag}.json")
    d, names = fe.task_frame(DF, best["task"])
    va, te = d[d.SPLIT == "validation"], d[d.SPLIT == "test"]
    cols = best["features"]
    pv, pt = m.predict_proba(fe.matrix(va, cols, cats)), m.predict_proba(fe.matrix(te, cols, cats))
    if pv.ndim == 1: pv, pt = np.c_[1 - pv, pv], np.c_[1 - pt, pt]
    return best, names, va, te, pv, pt


def ece_quantile(y, p, nb=10):
    q = pd.qcut(p, nb, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    return float((g.n * (g.obs - g.pred).abs()).sum() / g.n.sum()), g


def cancel_block(tag, mode, va, te, pv, pt):
    yv, yt = va.y.values, te.y.values
    pr, rc, th = precision_recall_curve(yv, pv[:, 1]); f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    thr = float(th[int(np.nanargmax(f[:-1]))])                       # same validation-chosen threshold as in the main test report
    for pname, sel in PERIODS.items():
        mk = sel(te).values
        y, p = yt[mk], pt[mk, 1]
        yh = (p >= thr).astype(int)
        P, R, F, _ = precision_recall_fscore_support(y, yh, labels=[0, 1], zero_division=0)
        rows.append(dict(task="cancellation", mode=mode, period=pname, n=int(mk.sum()), prevalence=float(y.mean()), pr_auc=float(average_precision_score(y, p)),
                         pr_auc_lift=float(average_precision_score(y, p) / y.mean()), roc_auc=float(roc_auc_score(y, p)), threshold=thr,
                         precision=float(P[1]), recall=float(R[1]), f1=float(F[1]), macro_f1=float(f1_score(y, yh, average="macro")), alert_rate=float(yh.mean())))
    # ---- probability quality (cancellation model)
    out = {}
    for pname, sel in PERIODS.items():
        mk = sel(te).values
        y, p = yt[mk], np.clip(pt[mk, 1], 1e-6, 1 - 1e-6)
        brier = float(brier_score_loss(y, p)); base = float(y.mean() * (1 - y.mean()))     # Brier of the constant prediction "test prevalence"
        lg = np.log(p / (1 - p)); fit = sm.Logit(y, sm.add_constant(lg)).fit(disp=0)
        ece, g = ece_quantile(y, p)
        out[pname] = dict(n=int(mk.sum()), prevalence=float(y.mean()), mean_predicted=float(p.mean()), brier=brier, brier_constant_test_prevalence=base,
                          brier_skill_vs_constant=float(1 - brier / base), calibration_intercept=float(fit.params[0]), calibration_slope=float(fit.params[1]),
                          ece_10_quantile_bins=ece)
        if pname == "all test": out[pname]["bins"] = g.reset_index(drop=True).assign(period=pname).to_dict("records")
    # recalibration check: isotonic regression on VALIDATION predictions, applied to test
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pv[:, 1], yv)
    pc = iso.predict(pt[:, 1])
    out["isotonic_recalibration_(validation-fitted)"] = dict(brier_test_raw=float(brier_score_loss(yt, pt[:, 1])), brier_test_recalibrated=float(brier_score_loss(yt, pc)),
                                                              ece_raw=ece_quantile(yt, np.clip(pt[:, 1], 1e-6, 1))[0], ece_recalibrated=ece_quantile(yt, pc)[0],
                                                              roc_auc_raw=float(roc_auc_score(yt, pt[:, 1])), roc_auc_recalibrated=float(roc_auc_score(yt, pc)))
    out["validation"] = dict(prevalence=float(yv.mean()), mean_predicted=float(pv[:, 1].mean()), brier=float(brier_score_loss(yv, pv[:, 1])))
    cal[tag] = out
    return pt[:, 1], yt, te.FL_DATE.values, pc


def multi_block(tag, mode, task_name, names, va, te, pv, pt):
    yv, yt = va.y.values, te.y.values
    maj = int(np.bincount(yv).argmax())                                # majority class according to VALIDATION
    K = len(names)
    for pname, sel in PERIODS.items():
        mk = sel(te).values
        y, P = yt[mk], pt[mk]
        yh = P.argmax(1)
        r = dict(task=task_name, mode=mode, period=pname, n=int(mk.sum()), accuracy=float(accuracy_score(y, yh)), majority_baseline=float((y == maj).mean()),
                 macro_f1=float(f1_score(y, yh, average="macro")), balanced_accuracy=float(balanced_accuracy_score(y, yh)),
                 macro_roc_auc_ovr=float(np.mean([roc_auc_score((y == k).astype(int), P[:, k]) for k in range(K) if 0 < (y == k).sum() < len(y)])))
        if K == 4:
            s = P[:, 2] + P[:, 3]
            r.update(delay_gt15_prevalence=float((y >= 2).mean()), delay_gt15_roc_auc=float(roc_auc_score(y >= 2, s)), delay_gt15_pr_auc=float(average_precision_score(y >= 2, s)))
        rows.append(r)


curves = {}
for mode in ["A", "B"]:
    DF = fe.load_v2(mode); cats = fe.train_cats(DF)
    sfx = "_B" if mode == "B" else ""
    best, names, va, te, pv, pt = predict("cancel" + sfx, DF, cats)
    curves[mode] = cancel_block("cancel" + sfx, mode, va, te, pv, pt)
    best, names, va, te, pv, pt = predict("code" + sfx, DF, cats)
    multi_block("code" + sfx, mode, "cancellation cause", names, va, te, pv, pt)
    for t, nm in [("delayacc", "delay 4 classes (unweighted)"), ("delay", "delay 4 classes (class-weighted)")]:
        best, names, va, te, pv, pt = predict(t + sfx, DF, cats)
        multi_block(t + sfx, mode, nm, names, va, te, pv, pt)
    del DF; gc.collect()
    print("mode", mode, "done", flush=True)

T = pd.DataFrame(rows)
T.to_csv(C.OUT / "19_test_by_period.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.round(4).to_string(index=False))
bins = []
for tag, o in cal.items():
    for rec in o["all test"].pop("bins"): bins.append(dict(model=tag, **rec))
pd.DataFrame(bins).to_csv(C.OUT / "19_cancel_calibration_bins.csv", index=False)
json.dump(cal, open(C.OUT / "19_cancel_calibration.json", "w"), indent=1)
for tag, o in cal.items():
    print(tag, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in o.items() if isinstance(v, dict)})

# ---- calibration figure (reliability diagram on log-log axes; 15 quantile bins incl. a finer top end)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, (mode, tag, col) in zip(axs, [("A", "cancel", "#3b6ea5"), ("B", "cancel_B", "#e08a2c")]):
    p, y, dates, pc = curves[mode]
    q = np.unique(np.concatenate([np.linspace(0, 0.9, 10), [0.95, 0.98, 0.99, 0.995, 1.0]]))
    b = pd.cut(p, np.unique(np.quantile(p, q)), include_lowest=True, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "b": b}).groupby("b", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    ax.plot([1e-3, 1], [1e-3, 1], "--", color="grey", label="perfect calibration")
    ax.plot(g.pred, g.obs, "o-", color=col, label=f"Mode {mode} (raw model output)")
    if mode == "B":                                     # the validation-selected Mode-B model uses class weighting -> raw outputs are not probabilities
        g2 = pd.DataFrame({"p": pc, "y": y, "b": b}).groupby("b", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))
        ax.plot(g2.pred, g2.obs, "s-", color="#6a9f3b", label="Mode B, isotonic recalibration (fitted on validation)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2e-3, 1); ax.set_ylim(2e-3, 1)
    ax.set_xlabel("Mean predicted cancellation probability"); ax.set_ylabel("Observed cancellation rate")
    o = cal[tag]["all test"]
    ax.set_title(f"Mode {mode}: Brier {o['brier']:.4f}, slope {o['calibration_slope']:.2f}", fontsize=10); ax.legend(loc="upper left" if mode == "A" else "lower right", fontsize=8)
plt.tight_layout(); plt.savefig(C.OUT / "figures" / "calibration_cancel.png", dpi=200); plt.close()
print("done", flush=True)
