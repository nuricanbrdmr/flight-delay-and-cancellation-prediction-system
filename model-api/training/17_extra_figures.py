"""Extra manuscript figures: monthly stability, cost-ratio curve, ROC (cancellation A vs B), multiclass OvR ROC (delay), delay class-weight vs unweighted confusion."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C, fe
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score

FIG = C.OUT / "figures"
O = C.OUT
BLUE, ORANGE = "#3b6ea5", "#d98c3f"

# 1) monthly PR-AUC (cancellation), modes A and B
a, b = pd.read_csv(O / "14_cancel_monthly.csv"), pd.read_csv(O / "14_cancel_B_monthly.csv")
x = np.arange(len(a)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(x - w / 2, a.pr_auc, w, label="Mode A (D-1 information)", color=BLUE)
ax.bar(x + w / 2, b.pr_auc, w, label="Mode B (+ same-day state, T-3h)", color=ORANGE)
ax.plot(x, a.prevalence, "k_", markersize=14, label="no-skill (monthly cancellation rate)")
ax.set_xticks(x); ax.set_xticklabels(a.month, rotation=35, ha="right"); ax.set_ylabel("PR-AUC"); ax.legend(fontsize=7)
ax.set_title("Cancellation prediction - monthly test performance"); plt.tight_layout(); plt.savefig(FIG / "monthly_pr_auc_cancel.png", dpi=200); plt.close()

# 2) cost-ratio curve
c = pd.read_csv(O / "14_cancel_B_cost_threshold.csv"); ca = pd.read_csv(O / "14_cancel_cost_threshold.csv")
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(ca.cost_ratio_FN_to_FP, ca.saving_vs_no_alert_pct, "o-", color=BLUE, label="Mode A")
ax.plot(c.cost_ratio_FN_to_FP, c.saving_vs_no_alert_pct, "s-", color=ORANGE, label="Mode B")
ax.axhline(0, color="grey", lw=0.8); ax.set_xscale("log"); ax.set_xticks(c.cost_ratio_FN_to_FP); ax.set_xticklabels(c.cost_ratio_FN_to_FP)
ax.set_xlabel("cost of a missed cancellation / cost of a false alarm"); ax.set_ylabel("cost saving vs. issuing no alerts (%)"); ax.legend()
ax.set_title("Cost-sensitive alerting (thresholds fixed on validation)"); plt.tight_layout(); plt.savefig(FIG / "cost_ratio_curve.png", dpi=200); plt.close()


def predict(tag):
    best = json.load(open(O / f"13_{tag}_best.json")); mode = best.get("mode", "A")
    df = fe.load_v2(mode); cats = fe.train_cats(df)
    m = XGBClassifier(); m.load_model(C.D / "models_v2" / f"{tag}.json")
    d, names = fe.task_frame(df, best["task"]); te = d[d.SPLIT == "test"]
    P = m.predict_proba(fe.matrix(te, best["features"], cats))
    return te.y.values, (np.c_[1 - P, P] if P.ndim == 1 else P), names


# 3) ROC cancellation
fig, ax = plt.subplots(figsize=(4.8, 4.4))
for tag, lab, col in [("cancel", "Mode A", BLUE), ("cancel_B", "Mode B", ORANGE)]:
    y, P, _ = predict(tag); f, t, _ = roc_curve(y, P[:, 1]); ax.plot(f, t, color=col, label=f"{lab} (AUC = {roc_auc_score(y, P[:, 1]):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend()
ax.set_title("Cancellation prediction - ROC, test period"); plt.tight_layout(); plt.savefig(FIG / "roc_cancel.png", dpi=200); plt.close()

# 4) multiclass OvR ROC for delay (unweighted Mode B)
y, P, names = predict("delayacc_B")
fig, ax = plt.subplots(figsize=(4.8, 4.4))
for k, n in enumerate(names):
    f, t, _ = roc_curve((y == k).astype(int), P[:, k]); ax.plot(f, t, label=f"{n} (AUC = {roc_auc_score((y == k).astype(int), P[:, k]):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=7)
ax.set_title("Departure delay classes - one-vs-rest ROC, test"); plt.tight_layout(); plt.savefig(FIG / "roc_delay_ovr.png", dpi=200); plt.close()
print("figures written")
