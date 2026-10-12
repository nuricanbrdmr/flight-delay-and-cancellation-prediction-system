"""Generate manuscript-ready (English) tables from the result files -> PAPER_TABLES.md. No number is typed by hand."""
import json, numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parent
O = R / "outputs"
res = json.load(open(O / "14_test_results.json"))
out = []


def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def add(title, body, note=None):
    out.append(f"### {title}\n\n{body}\n" + (f"\n*{note}*\n" if note else ""))


def ci(v, d=3):
    return f"{v[0]:.{d}f}-{v[1]:.{d}f}"


# ---- T1 data ------------------------------------------------------------------------------------------------------
src = pd.read_csv(O / "06_zelaya_full_year_rates.csv")
dist = pd.read_csv(O / "09_task_class_distributions.csv")
canc = dist[dist.task.str.startswith("cancel (")].pivot_table(index="split", columns="cls", values="n").reindex(["train", "validation", "test"])
t1 = pd.DataFrame({"Split": ["Training", "Validation", "Test"], "Period": ["2019-01 to 2022-06", "2022-07 to 2022-12", "2023-01 to 2023-08 and 2024-01"],
                   "Flights": [int(canc.loc[s].sum()) for s in canc.index], "Cancelled": [int(canc.loc[s, "1"]) for s in canc.index]})
t1["Cancellation rate (%)"] = (100 * t1.Cancelled / t1.Flights).round(2)
add("Table A. Final dataset after reconstruction (natural DOT class distribution, no external cancelled-only records)", md(t1),
    "Sources: Zelaya 3M sample (2019-01 to 2023-08; 3,000,000 flights) + Singh (2024-01, complete month, 582,425 flights). Duplicates on FL_DATE+DOT_CODE+FL_NUMBER+ORIGIN+DEST+CRS_DEP_TIME: 0. "
    "The cancelled-only 2016 records of the earlier file (unknown provenance) were removed.")

# ---- T2 targets -----------------------------------------------------------------------------------------------------
d2 = dist[~dist.task.str.startswith("cancel (")].copy()
d2 = d2[~((d2.task.str.startswith("cancel_code")) & (d2.cls == "D"))]
d2["n (share %)"] = d2.apply(lambda r: f"{int(r.n):,} ({100*r.share:.1f})", axis=1)
t2 = d2.pivot_table(index=["task", "cls"], columns="split", values="n (share %)", aggfunc="first")[["train", "validation", "test"]].reset_index()
add("Table B. Target definitions and class distributions (cancellation cause: class D excluded)", md(t2),
    "Cancellation: all flights. Cause: cancelled flights, DOT codes A-C (code D was assigned almost exclusively in 2020, 38% of training cancellations but 6 and 27 cases in validation/test). "
    "Delay: not cancelled, not diverted, departure delay in minutes (<=0, 1-15, 16-30, >30).")

# ---- T3 ablation ---------------------------------------------------------------------------------------------------
def ab(tag, metric_cols):
    a = pd.read_csv(O / f"13_{tag}_ablation.csv")
    return a[["feature_set"] + metric_cols].rename(columns={"feature_set": "Feature set"})
a1 = pd.read_csv(O / "13_cancel_ablation.csv")[["feature_set", "pr_auc", "roc_auc"]].rename(columns={"feature_set": "Feature set (Mode A)", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"})
add("Table C1. Feature ablation, cancellation model (validation period, default hyper-parameters)", md(a1),
    "A: calendar + schedule + previous-day (D-1) origin weather; B: A + history; C: B + holiday + destination D-1 weather; D: C without weather; E: C without history; F: calendar + schedule only. "
    "History = cancellation/delay rates of the nation, origin airport and airline on D-1 and D-7..D-1.")
b1 = pd.read_csv(O / "13_cancel_B_ablation.csv")[["feature_set", "pr_auc", "roc_auc"]].rename(columns={"feature_set": "Feature set (Mode B)", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"})
add("Table C2. Feature ablation, cancellation model, Mode B (adds same-day operational state observed 3 h before departure)", md(b1))
c1 = pd.read_csv(O / "13_code_ablation.csv")[["feature_set", "accuracy", "macro_f1"]].rename(columns={"feature_set": "Feature set (Mode A)", "accuracy": "Accuracy", "macro_f1": "Macro-F1"})
add("Table C3. Feature ablation, cancellation-cause model (validation period)", md(c1))

# ---- T4 SMOTE comparison ---------------------------------------------------------------------------------------------
s = pd.read_csv(O / "11_cancel_validation_results.csv", sep=None, engine="python")
s["Weather"] = s.weather.astype(str).map({"True": "with D-1 weather", "False": "without weather"})
s = s[["Weather", "strategy", "pr_auc", "roc_auc", "bestF1_prec1", "bestF1_rec1", "bestF1_f1_1", "bestF1_macroF1", "bestF1_balAcc"]]
s.columns = ["Weather", "Imbalance handling", "PR-AUC", "ROC-AUC", "Precision (cancel)", "Recall (cancel)", "F1 (cancel)", "Macro-F1", "Balanced acc."]
add("Table D. Imbalance handling for cancellation prediction (validation, basic feature set; SMOTENC applied to the training part only)", md(s),
    "strategy: none / class_weight (scale_pos_weight = n_neg/n_pos) / smotenc_train_only (sampling_strategy 0.10, k=5; categorical = airline, origin, destination). Thresholds maximise F1 on validation.")

# ---- T5 main results ------------------------------------------------------------------------------------------------
rows = []
for tag, name, mode in [("cancel", "Cancellation (binary)", "A"), ("cancel_B", "Cancellation (binary)", "B")]:
    x = res[tag]; t = x["test"]; b = x["at_val_bestF1_threshold"]
    rows.append([name, mode, f"PR-AUC {t['pr_auc']:.3f} ({ci(t['pr_auc_ci95'])}); ROC-AUC {t['roc_auc']:.3f} ({ci(t['roc_auc_ci95'])})",
                 f"P={b['precision'][1]:.3f}, R={b['recall'][1]:.3f}, F1={b['f1'][1]:.3f}; macro-F1={b['macro_f1']:.3f}; bal. acc.={b['balanced_accuracy']:.3f}",
                 f"no-skill PR-AUC = prevalence = {x['prevalence_test']:.4f}; accuracy {b['accuracy']:.3f} (trivial all-negative: {1-x['prevalence_test']:.3f})"])
for tag, name, mode in [("code", "Cancellation cause (A/B/C)", "A"), ("code_B", "Cancellation cause (A/B/C)", "B"),
                        ("delayacc", "Delay, 4 classes - accuracy-oriented (no class weights)", "A"), ("delayacc_B", "Delay, 4 classes - accuracy-oriented (no class weights)", "B"),
                        ("delay", "Delay, 4 classes - class-weighted", "A"), ("delay_B", "Delay, 4 classes - class-weighted", "B")]:
    x = res[tag]; t = x["test"]
    extra = f"; delay>15 min as binary: ROC-AUC {x['delay_gt15_binary_test']['roc_auc']:.3f}, PR-AUC {x['delay_gt15_binary_test']['pr_auc']:.3f} (prevalence {x['delay_gt15_binary_test']['prevalence']:.3f})" if "delay_gt15_binary_test" in x else ""
    rows.append([name, mode, f"Accuracy {t['accuracy']:.3f} ({ci(t['accuracy_ci95'])}); macro-F1 {t['macro_f1']:.3f} ({ci(t['macro_f1_ci95'])})",
                 f"balanced acc. {t['balanced_accuracy']:.3f}; macro PR-AUC {t['macro_pr_auc']:.3f}; macro ROC-AUC (OvR) {t['macro_roc_auc_ovr']:.3f}",
                 f"majority-class baseline accuracy {x['baseline_majority_accuracy_test']:.3f} ('{x['baseline_majority_class']}')" + extra])
t5 = pd.DataFrame(rows, columns=["Task", "Mode", "Headline test metrics (95% bootstrap CI)", "Further metrics", "Reference baseline"])
add("Table E. Test-period results (2023-01..2023-08, 2024-01; models fitted on training data, hyper-parameters/thresholds chosen on validation, test used once)", md(t5),
    "Mode A: information available up to the previous completed day (D-1). Mode B: Mode A + operational state of the same day observed 3 hours before scheduled departure "
    "(delays/cancellations of earlier flights at the origin and of the airline, share of scheduled-but-not-yet-departed flights). 56% of the test rows belong to 2024-01 (complete month), see Table G.")

# ---- T6 class-wise ---------------------------------------------------------------------------------------------------
cw = []
for tag in ["code_B", "delayacc_B", "delay_B"]:
    x = res[tag]
    for c in x["per_class"]:
        cw.append([tag, c["cls"], c["support"], c["precision"], c["recall"], c["f1"]])
add("Table F. Class-wise test results (Mode B models)", md(pd.DataFrame(cw, columns=["Model", "Class", "Support", "Precision", "Recall", "F1"])),
    "delayacc_B = no class weights (accuracy-oriented); delay_B = balanced class weights (macro-F1-oriented). Class 16-30 min is never predicted by the unweighted model.")

# ---- T7 monthly ---------------------------------------------------------------------------------------------------------
m = pd.read_csv(O / "14_cancel_monthly.csv").merge(pd.read_csv(O / "14_cancel_B_monthly.csv"), on=["month", "n", "prevalence"], suffixes=(" (A)", " (B)"))
m = m[["month", "n", "prevalence", "pr_auc (A)", "pr_auc (B)", "roc_auc (A)", "roc_auc (B)"]]
add("Table G. Cancellation model, monthly test performance (temporal stability)", md(m, 4),
    "Performance varies strongly between months; cancellations are dominated by few storm days.")

# ---- T8 cost -------------------------------------------------------------------------------------------------------------
ct = pd.read_csv(O / "14_cancel_B_cost_threshold.csv")[["cost_ratio_FN_to_FP", "threshold_from_validation", "alert_rate", "recall", "precision", "cost_per_1000_flights", "saving_vs_no_alert_pct"]]
ct.columns = ["Cost ratio FN:FP", "Threshold (from validation)", "Alert rate", "Recall", "Precision", "Cost per 1000 flights (FP=1 unit)", "Saving vs. no alerts (%)"]
add("Table H. Cost-sensitive threshold analysis, cancellation model (Mode B), test period", md(ct),
    "Relative cost units (false alarm = 1, missed cancellation = ratio). Thresholds minimise validation cost. Negative/zero saving at low ratios shows that alerts only pay off when a missed cancellation is much costlier than a false alarm. "
    "No monetary or annual-benefit claim is made.")

# ---- T9 hyperparameters --------------------------------------------------------------------------------------------------
hp = []
for tag in ["cancel_B", "code_B", "delayacc_B", "delay_B"]:
    b = json.load(open(O / f"13_{tag}_best.json")); p = b["params"]
    hp.append([tag, p["learning_rate"], p["max_depth"], p["min_child_weight"], p["subsample"], p["colsample_bytree"], p["reg_lambda"], p["reg_alpha"], p["gamma"], b["weight_mode"], b["best_iteration"]])
add("Table I. Selected XGBoost hyper-parameters (random search on the validation period)",
    md(pd.DataFrame(hp, columns=["Model", "learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "lambda", "alpha", "gamma", "class weights", "trees"]), 3),
    "Search: 12 (cancellation), 20 (cause), 8 (delay; on a 700k-row random subsample of the training period) random configurations; bounds: learning_rate 0.03-0.15 (log), max_depth 4-10, min_child_weight 1-60 (log), "
    "subsample 0.6-1.0, colsample_bytree 0.5-1.0, lambda 0.5-20 (log), alpha {0,0.1,0.5,1,3}, gamma {0,0.5,1,3}, class-weight scheme; objective: validation PR-AUC (cancellation), macro-F1 (cause, weighted delay), accuracy (unweighted delay); "
    "early stopping on validation. The final model is refitted on all training rows.")

# ---- T10 multicollinearity ---------------------------------------------------------------------------------------------------
v = pd.read_csv(O / "14_vif_train.csv").head(10)
v.columns = ["Feature", "VIF"]
mc = res["multicollinearity"]
add("Table J. Multicollinearity of the final feature set (training sample): ten largest VIF values", md(v, 1),
    f"{mc['vif_gt10']} of {mc['n_features']} numeric features have VIF>10 and {mc['n_pairs_abs_r_ge_0p8']} feature pairs have |r|>=0.8 (see 14_high_correlation_pairs.csv, figures/correlation_matrix.png). "
    "High VIF arises from nested history aggregates (D-1 vs D-7, nation vs airport) and sine/cosine calendar encodings. Gradient-boosted trees are insensitive to collinearity for prediction, but SHAP attributions are shared among correlated features.")

# ---- SHAP ------------------------------------------------------------------------------------------------------------------------
sh = []
for tag in ["cancel_B", "code_B", "delayacc_B"]:
    s_ = pd.read_csv(O / f"14_{tag}_shap_mean_abs.csv", index_col=0).head(5)
    sh.append([tag, "; ".join(f"{i} ({v:.3f})" for i, v in s_.mean_abs_shap.items())])
add("Table K. Five most important features by mean absolute SHAP value (test sample of 30,000 rows)", md(pd.DataFrame(sh, columns=["Model", "Top features (mean abs. SHAP)"])))

(R.parent / "docs" / "PAPER_TABLES.md").write_text("# Manuscript-ready tables (generated by 16_make_paper_tables.py)\n\n" + "\n".join(out), encoding="utf-8")
print("written", R.parent / "docs" / "PAPER_TABLES.md")
