"""Descriptive figures 1-5 for the manuscript, regenerated from the rebuilt dataset + the raw Zelaya file (delay-cause columns)."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import common as C

FIG = C.OUT / "figures"
BLUE, ORANGE, GREY = "#3b6ea5", "#d98c3f", "#8a8a8a"
stats = {}

cf = pd.read_csv(C.D / "clean_flights.csv.gz", usecols=["FL_DATE", "AIRLINE_CODE", "CANCELLED", "DIVERTED", "DEP_DELAY", "SOURCE"], parse_dates=["FL_DATE"])
ok = cf[(cf.CANCELLED == 0) & (cf.DIVERTED == 0)]
late = ok.DEP_DELAY > 15
stats.update(n_total=len(cf), n_cancelled=int(cf.CANCELLED.sum()), n_diverted=int(cf.DIVERTED.sum()), n_ok=len(ok), delayed_gt15=int(late.sum()), delayed_share=float(late.mean()))

# Fig 1 - delay status (15-minute rule)
fig, ax = plt.subplots(figsize=(4.2, 3.2))
v = [int((~late).sum()), int(late.sum())]
ax.bar(["On time / <=15 min", "Delayed > 15 min"], v, color=[BLUE, ORANGE])
for i, n in enumerate(v): ax.text(i, n, f"{n/1e6:.2f} M\n({100*n/sum(v):.1f}%)", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Flights"); ax.set_ylim(0, max(v) * 1.2); ax.set_title("Departure status (non-cancelled, non-diverted flights)", fontsize=9); plt.tight_layout(); plt.savefig(FIG / "desc_fig1_delay_status.png", dpi=200); plt.close()

# Fig 2 - delayed flights by airline (top 12)
raw = pd.read_csv(C.D / "flights_sample_3m_origin.csv", usecols=["AIRLINE", "AIRLINE_CODE", "DEP_DELAY", "CANCELLED", "DIVERTED", "DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS", "DELAY_DUE_SECURITY", "DELAY_DUE_LATE_AIRCRAFT"])
nm = raw.drop_duplicates("AIRLINE_CODE").set_index("AIRLINE_CODE").AIRLINE
d = ok.assign(late=late).groupby("AIRLINE_CODE").late.agg(["sum", "size"]).sort_values("sum", ascending=False).head(12)
fig, ax = plt.subplots(figsize=(5.4, 3.6))
labels = [nm.get(c, c).replace(" Inc.", "").replace(" Co.", "").replace(" Airlines", "").replace(" Air Lines", "") for c in d.index]
ax.barh(labels[::-1], d["sum"].values[::-1], color=BLUE); ax.set_xlabel("Flights delayed > 15 min"); ax.set_title("Delayed departures by airline (12 largest)", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "desc_fig2_delay_by_airline.png", dpi=200); plt.close()

# Fig 3 / 4 - delay causes (reported by DOT for arrival delays >= 15 min)
cc = ["DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS", "DELAY_DUE_SECURITY", "DELAY_DUE_LATE_AIRCRAFT"]
lab = ["Carrier", "Weather", "National Air System", "Security", "Late aircraft"]
c = raw.dropna(subset=cc)
share = c[cc].idxmax(axis=1).value_counts(normalize=True).reindex(cc).fillna(0)
stats["cause_n"] = int(len(c)); stats["cause_share"] = {l: float(share[k]) for l, k in zip(lab, cc)}
fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.bar(lab, 100 * share.values, color=BLUE)
for i, s_ in enumerate(share.values): ax.text(i, 100 * s_, f"{100*s_:.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Share of delayed flights (%)"); ax.set_title("Dominant reported delay cause", fontsize=9); plt.xticks(rotation=20, ha="right", fontsize=8); plt.tight_layout()
plt.savefig(FIG / "desc_fig3_delay_causes.png", dpi=200); plt.close()
cor = c[cc].corr().values
fig, ax = plt.subplots(figsize=(4.4, 3.8)); im = ax.imshow(cor, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=7); ax.set_yticks(range(5)); ax.set_yticklabels(lab, fontsize=7)
for i in range(5):
    for j in range(5): ax.text(j, i, f"{cor[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, fraction=0.04); ax.set_title("Correlation of delay-cause minutes", fontsize=9); plt.tight_layout(); plt.savefig(FIG / "desc_fig4_cause_heatmap.png", dpi=200); plt.close()

# Fig 5 - pipeline (vertical, large text)
steps = ["Raw DOT records: Zelaya (2019-01 to 2023-08), Singh (2024-01)", "De-duplication on date, DOT id, flight number, origin, destination, scheduled time",
         "Features known before departure: calendar, schedule, D-1 weather (origin, destination), D-1 / D-7 history, (Mode B) same-day state at T-3h",
         "Chronological split: train 2019-01..2022-06, validation 2022-07..12, test 2023-01..2024-01", "Fit on train only; hyper-parameters and thresholds chosen on validation (XGBoost)",
         "One-shot test evaluation, ablation, SHAP, cost analysis"]
fig, ax = plt.subplots(figsize=(3.6, 4.4)); ax.set_xlim(0, 1); ax.set_ylim(0, len(steps)); ax.axis("off")
import textwrap
for i, s_ in enumerate(steps):
    y0 = len(steps) - 1 - i
    ax.add_patch(FancyBboxPatch((0.03, y0 + 0.12), 0.94, 0.70, boxstyle="round,pad=0.01,rounding_size=0.05", fc="#e9f0f8", ec=BLUE, lw=1))
    ax.text(0.5, y0 + 0.47, textwrap.fill(s_, 44), ha="center", va="center", fontsize=7.2)
    if i < len(steps) - 1: ax.annotate("", xy=(0.5, y0 + 0.02), xytext=(0.5, y0 + 0.12), arrowprops=dict(arrowstyle="->", color=GREY))
plt.tight_layout(); plt.savefig(FIG / "desc_fig5_pipeline.png", dpi=220); plt.close()
json.dump(stats, open(C.OUT / "18b_descriptive_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
