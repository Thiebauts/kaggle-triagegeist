"""Generate all publication figures as vector PDF + PNG from raw data.

Reads data from the overnight Azure ML results and outputs to report/figures/.
All figures use the Almond Blossoms palette for consistency with the LaTeX report.

Usage:
    python report/latex-figures/generate_all_figures.py
"""

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Raw data directories. These point at the local Azure-overnight outputs that
# are NOT in the repo (heavy .npy / .csv files). Override with
# TRIAGEGEIST_OVERNIGHT_V1 / TRIAGEGEIST_OVERNIGHT_V2 environment variables if
# your data lives elsewhere; otherwise the default assumes the original layout
# under ~/Documents/Kaggle/Triagegeist/.
import os as _os
_DEFAULT_V1 = Path.home() / "Documents/Kaggle/Triagegeist/overnight_results"
_DEFAULT_V2 = Path.home() / "Documents/Kaggle/Triagegeist/overnight_results_v2"
RES_V1 = Path(_os.environ.get("TRIAGEGEIST_OVERNIGHT_V1", _DEFAULT_V1))
RES_MIMIC = RES_V1 / "mimic"
RES_V2 = Path(_os.environ.get("TRIAGEGEIST_OVERNIGHT_V2", _DEFAULT_V2))

# ---------------------------------------------------------------------------
# Almond Blossoms palette
# ---------------------------------------------------------------------------
VG_SKY = "#7BA7BC"
VG_HEADING = "#3D6B7E"
VG_BRANCH = "#5C4033"
VG_LEAF = "#6B8E6B"
VG_BLOSSOM = "#C4A882"
VG_PETAL = "#E8DDD3"
VG_BOX_BG = "#F4F0EB"
VG_BOX_FRAME = "#C4D7DF"

# Per-ESI colours (warm→cool gradient)
ESI_COLORS = {1: VG_BRANCH, 2: VG_HEADING, 3: VG_SKY, 4: VG_LEAF, 5: VG_BLOSSOM}
ESI_SEQ = [ESI_COLORS[i] for i in range(1, 6)]

COLOR_POS = "#E8525A"
COLOR_NEG = "#5B9BD5"

# Style baseline
mpl.rcParams.update({
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.prop_cycle": mpl.cycler(color=ESI_SEQ),
})


def save_fig(fig, name):
    """Save as PDF (vector) for LaTeX inclusion."""
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


def placeholder(name, reason):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f"{name}\nunavailable\n({reason})",
            ha="center", va="center", fontsize=14, color="gray")
    ax.axis("off")
    save_fig(fig, name)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Clinical operating curve
# ═══════════════════════════════════════════════════════════════════════════
def fig_clinical_operating_curve():
    path = RES_MIMIC / "clinical_operating_curve_mimic.csv"
    if not path.exists():
        placeholder("clinical_operating_curve", "missing CSV"); return
    df = pd.read_csv(path)
    ut_pct = df["undertriage_rate"] * 100

    fig, ax = plt.subplots(figsize=(11, 6.5))
    # axvspan fills the full visible y-range automatically, so the safety
    # and risk zones stay flush with the axis edges regardless of y-margin.
    xmax_zone = ut_pct.max() + 1
    ax.axvspan(0, 5, color=VG_LEAF, alpha=0.2,
               label="Safety zone (UT ≤ 5%)", zorder=0)
    ax.axvspan(5, xmax_zone, color=COLOR_POS, alpha=0.1,
               label="Under-triage risk zone", zorder=0)
    ax.plot(ut_pct, df["macro_f1"], "o-", color=VG_HEADING, linewidth=2, markersize=6)
    ax.axvline(5, color=COLOR_POS, linestyle="--", alpha=0.8, linewidth=1.8,
               label="ACS-COT ≤ 5% target")
    f1_max = df["macro_f1"].max()
    ax.axhline(f1_max, color="gray", linestyle=":", alpha=0.6,
               label=f"Performance-optimal F1 = {f1_max:.3f}")

    p25 = df.iloc[(df["multiplier"] - 2.5).abs().idxmin()]
    ax.scatter(p25["undertriage_rate"] * 100, p25["macro_f1"], s=250, c=VG_BLOSSOM,
               edgecolors="black", linewidths=1.5, zorder=5,
               label=f"2.5× point (F1={p25['macro_f1']:.3f}, UT={p25['undertriage_rate']*100:.1f}%)")

    safe = df[df["undertriage_rate"] <= 0.05]
    if len(safe):
        p5 = safe.iloc[0]
        ax.scatter(p5["undertriage_rate"] * 100, p5["macro_f1"], s=250, c=VG_LEAF,
                   marker="^", edgecolors="black", linewidths=1.5, zorder=5,
                   label=f"First ≤5% UT: mult={p5['multiplier']}, F1={p5['macro_f1']:.3f}")

    ax.set_xlabel("Under-triage rate (%)", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("Clinical Operating Curve — MIMIC-IV-ED\nF1 vs Under-triage Tradeoff via Threshold Multiplier")
    ax.legend(loc="lower right", fontsize=9, markerscale=0.45,
              labelspacing=0.6, borderpad=0.6)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, xmax_zone)
    ax.margins(x=0)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_mults = df.iloc[::4]
    ax2.set_xticks(tick_mults["undertriage_rate"] * 100)
    ax2.set_xticklabels([f"{m:.2f}×" for m in tick_mults["multiplier"]], fontsize=9)
    ax2.set_xlabel("Threshold multiplier", fontsize=10)
    save_fig(fig, "clinical_operating_curve")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Confusion matrix + distance heatmap (combined)
# ═══════════════════════════════════════════════════════════════════════════
def fig_confusion_combined():
    # Prefer repo-relative path; fall back to legacy absolute path.
    path = REPO / "results" / "mimic" / "pipeline_results.json"
    if not path.exists():
        path = RES_MIMIC / "mimic_pipeline_results.json"
    if not path.exists():
        placeholder("confusion_matrix_combined", "missing results"); return
    with open(path) as f:
        data = json.load(f)
    cm = np.array(data.get("confusion_matrix_oof") or [])
    if cm.size == 0:
        placeholder("confusion_matrix_combined", "no confusion matrix"); return
    row_sum = cm.sum(axis=1, keepdims=True)
    row_pct = 100 * cm / np.maximum(row_sum, 1)

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(15, 6.5))
    # Two equal heatmap panels; colorbar placed manually.
    # top=0.80 leaves vertical room for the legend (above each subplot title)
    # and the suptitle, so they don't collide.
    gs = gridspec.GridSpec(1, 2, wspace=0.20,
                           left=0.06, right=0.87, top=0.80, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # --- Left: standard confusion matrix ---
    im = ax1.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt_color = "white" if row_pct[i, j] > 50 else "black"
            ax1.text(j, i, f"{cm[i,j]:,}\n{row_pct[i,j]:.1f}%",
                     ha="center", va="center", color=txt_color, fontsize=10)
    ax1.set_xticks(range(5)); ax1.set_yticks(range(5))
    ax1.set_xticklabels([f"ESI-{i+1}" for i in range(5)], fontsize=12)
    ax1.set_yticklabels([f"ESI-{i+1}" for i in range(5)], fontsize=12)
    ax1.set_xlabel("Predicted", fontsize=12); ax1.set_ylabel("Actual", fontsize=12)
    ax1.set_title(f"Confusion Matrix (N={cm.sum():,})", fontsize=14)
    ax1.set_aspect("equal")

    # --- Right: clinical distance heatmap ---
    colors_by_dist = {0: "white", 1: VG_PETAL, 2: VG_BLOSSOM, 3: COLOR_POS, 4: "#8b0000"}
    for i in range(5):
        for j in range(5):
            dist = abs(i - j)
            alpha = 1.0 if j > i else (0.5 if j < i else 1.0)
            face = colors_by_dist.get(dist, "#8b0000")
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                              edgecolor="white", linewidth=2, alpha=alpha)
            ax2.add_patch(rect)
            ax2.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                     color="black", fontsize=10, fontweight="bold")
    ax2.set_xlim(-0.5, 4.5); ax2.set_ylim(4.5, -0.5)
    ax2.set_xticks(range(5)); ax2.set_yticks(range(5))
    ax2.set_xticklabels([f"ESI-{i+1}" for i in range(5)], fontsize=12)
    ax2.set_yticklabels([f"ESI-{i+1}" for i in range(5)], fontsize=12)
    ax2.set_xlabel("Predicted", fontsize=12); ax2.set_ylabel("Actual", fontsize=12, labelpad=2)
    ax2.set_title("Clinical Severity of Errors", fontsize=14)
    ax2.set_aspect("equal")
    ax2.tick_params(axis="y", pad=2)

    # Legend above right subplot: 4 entries, single row.
    # Under-/over-triage shading is explained in the caption.
    legend_elems = [
        Patch(facecolor="white", edgecolor="black", label="Correct"),
        Patch(facecolor=VG_PETAL, edgecolor="black", label="Off by 1"),
        Patch(facecolor=VG_BLOSSOM, edgecolor="black", label="Off by 2"),
        Patch(facecolor=COLOR_POS, edgecolor="black", label=r"Off by $\geq$3"),
    ]
    ax2.legend(handles=legend_elems, loc="lower center",
               bbox_to_anchor=(0.5, 1.10), ncol=4,
               fontsize=11, frameon=False)

    fig.suptitle("Confusion Analysis — MIMIC-IV-ED, Standard Threshold",
                 fontsize=15, fontweight="bold", color=VG_HEADING, y=0.97)

    # Draw first to finalise axes positions, then place colorbar manually
    fig.canvas.draw()
    bbox1 = ax1.get_position()
    cax = fig.add_axes([bbox1.x1 + 0.008, bbox1.y0, 0.012, bbox1.height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Row %", fontsize=12)

    save_fig(fig, "confusion_matrix_combined")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Keyword leakage heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig_keyword_leakage():
    path = RES_V1 / "keyword_distribution_by_esi.csv"
    if not path.exists():
        placeholder("keyword_leakage_heatmap", "missing csv"); return
    df = pd.read_csv(path)
    pivot = df.pivot(index="keyword", columns="esi_level", values="pct_of_esi")
    pivot = pivot.loc[pivot.max(axis=1) > 0]
    pivot = pivot.reindex(sorted(pivot.index, key=lambda k: -pivot.loc[k].max()))

    fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="Reds", aspect="auto", vmin=0, vmax=25)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            color = "white" if v > 15 else "black"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"ESI-{c}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Severity Keyword Distribution by ESI Level\nEvidence of Synthetic Label Leakage")
    plt.colorbar(im, ax=ax, label="% of ESI level")
    save_fig(fig, "keyword_leakage_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Feature importance (Kaggle honest tabular)
# ═══════════════════════════════════════════════════════════════════════════
def fig_feature_importance():
    path = RES_V1 / "tabular_feature_importance.csv"
    if not path.exists():
        placeholder("feature_importance", "missing csv"); return
    df = pd.read_csv(path).head(15).iloc[::-1]

    vital_raw = {"systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
                 "temperature_c", "spo2", "gcs_total", "pain_score"}
    engineered = {"shock_index", "mean_arterial_pressure", "pulse_pressure",
                  "news2_score", "n_abnormal_vitals", "bmi", "weight_kg", "height_cm",
                  "arrival_hour_sin", "arrival_hour_cos"}
    colors = []
    for f in df["feature"]:
        if f in vital_raw: colors.append(VG_HEADING)
        elif f in engineered: colors.append(VG_BLOSSOM)
        else: colors.append(VG_LEAF)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.arange(len(df)), df["gain"], color=colors)
    ax.set_yticks(np.arange(len(df))); ax.set_yticklabels(df["feature"])
    ax.set_xlabel("LightGBM gain importance")
    ax.set_title("Top 15 Features — Honest Tabular Model, Kaggle Data")
    ax.legend(handles=[
        Patch(color=VG_HEADING, label="Raw vitals"),
        Patch(color=VG_BLOSSOM, label="Engineered vitals"),
        Patch(color=VG_LEAF, label="Demographics/history/other"),
    ], loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, "feature_importance")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Ordinal penalty comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig_ordinal_penalty():
    path = RES_MIMIC / "ordinal_penalty_mimic_results.json"
    if not path.exists():
        placeholder("ordinal_penalty_comparison", "missing json"); return
    with open(path) as f:
        data = json.load(f)
    variants = data.get("variants", {})
    order = ["a", "a_thresh", "b", "b_thresh", "c", "c_thresh"]
    labels = ["A\nStandard", "A+thresh", "B\nAsym weights", "B+thresh", "C\nDirectional", "C+thresh"]
    f1s = [variants.get(k, {}).get("macro_f1", 0) for k in order]
    uts = [variants.get(k, {}).get("under_triage_pct", 0) for k in order]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = np.arange(len(order)); w = 0.4
    bars1 = ax1.bar(x - w/2, f1s, w, label="Macro F1", color=VG_HEADING)
    ax1.set_ylabel("Macro F1", color=VG_HEADING)
    ax1.tick_params(axis="y", labelcolor=VG_HEADING)
    ax1.set_ylim(0, max(f1s) * 1.15)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, uts, w, label="Under-triage %", color=COLOR_POS)
    ax2.set_ylabel("Under-triage rate (%)", color=COLOR_POS)
    ax2.tick_params(axis="y", labelcolor=COLOR_POS)
    ax2.axhline(5, color="red", linestyle="--", alpha=0.5, label="ACS-COT 5% target")
    ax2.set_ylim(0, max(uts) * 1.15)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_title("Directional Ordinal Penalty — F1 vs Under-triage Tradeoff (MIMIC-IV-ED)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    for bar, v in zip(bars1, f1s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", fontsize=9, color=VG_HEADING)
    for bar, v in zip(bars2, uts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", fontsize=9, color=COLOR_POS)
    save_fig(fig, "ordinal_penalty_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Bias audit disparities — age-band only (current Figure 9.1 source)
# ═══════════════════════════════════════════════════════════════════════════
def fig_bias_audit_age():
    """Per-age-band under-triage rates: standard vs asymmetric 2.5x threshold.

    Replaces the legacy fig_bias_audit() which ranked named ethnic
    intersectional subgroups. The race-to-five-category mapping used in the
    audit is incomplete (33 MIMIC race strings collapsed to 5 by substring
    matching), so ranked race-level claims are not defensible. Chapter 9
    explicitly avoids them; this figure makes the same restraint visual.

    Data lives at results/mimic/age_band_ut_by_threshold.json (provenance
    documented in the JSON's _about field).
    """
    data_path = REPO / "results" / "mimic" / "age_band_ut_by_threshold.json"
    if not data_path.exists():
        placeholder("bias_audit_disparities", "missing age-band json"); return
    with open(data_path) as f:
        ab = json.load(f)
    bands = [r["age_band"] for r in ab["bands"]]
    std = [r["ut_standard"] * 100 for r in ab["bands"]]
    asym = [r["ut_asymmetric_2_5x"] * 100 for r in ab["bands"]]
    overall_std = ab["overall_ut_standard"] * 100
    target = ab["acs_cot_target"] * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(bands))
    h = 0.38
    bars_std = ax.barh(y - h / 2, std, height=h, color=VG_HEADING, alpha=0.9,
                       edgecolor="white", linewidth=0.5,
                       label="Standard threshold")
    bars_asym = ax.barh(y + h / 2, asym, height=h, color=VG_BLOSSOM, alpha=0.95,
                        edgecolor="white", linewidth=0.5, hatch="//",
                        label=r"Asymmetric 2.5$\times$ threshold")

    ax.axvline(overall_std, color="black", linestyle=":", alpha=0.7,
               linewidth=1.4, label=f"Overall UT = {overall_std:.2f}%")
    ax.axvline(target, color=COLOR_POS, linestyle="--", alpha=0.85,
               linewidth=1.4, label=f"ACS-COT {target:.0f}% target")

    for bar, v in zip(bars_std, std):
        ax.text(v + 0.25, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", ha="left", fontsize=10, color=VG_HEADING)
    for bar, v in zip(bars_asym, asym):
        ax.text(v + 0.25, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", ha="left", fontsize=10, color="#7a6644")

    ax.set_yticks(y)
    ax.set_yticklabels(bands, fontsize=11)
    ax.invert_yaxis()  # 18-30 at the top
    ax.set_xlim(0, 22)
    ax.set_xlabel("Under-triage rate (%)", fontsize=12)
    ax.set_ylabel("Age band", fontsize=12)
    ax.set_title("Under-triage by age band (MIMIC-IV-ED)", fontsize=13,
                 fontweight="bold", color=VG_HEADING)
    ax.legend(loc="lower right", fontsize=10, frameon=True, fancybox=False,
              edgecolor="lightgray")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    save_fig(fig, "bias_audit_disparities")


# ═══════════════════════════════════════════════════════════════════════════
# 7b. Legacy intersectional bias audit — superseded by fig_bias_audit_age().
# Kept for provenance; do not register in main(). Race aggregation is
# acknowledged as incomplete (chapter 9), so ranked named-subgroup output
# is no longer used in the report.
# ═══════════════════════════════════════════════════════════════════════════
def fig_bias_audit():
    std_path = RES_MIMIC / "bias_audit_top20_undertriage.csv"
    asym_path = RES_MIMIC / "bias_audit_with_asymmetric.csv"
    if not std_path.exists():
        placeholder("bias_audit_disparities", "missing csv"); return
    std = pd.read_csv(std_path).head(15).iloc[::-1]
    labels_list = std.apply(lambda r: f"{r['axis']}: {r['subgroup']}", axis=1)

    asym_ut = pd.Series([np.nan] * len(std), index=std.index)
    if asym_path.exists():
        asym = pd.read_csv(asym_path)
        for i, r in std.iterrows():
            m = asym[(asym["axis"] == r["axis"]) & (asym["subgroup"] == r["subgroup"])]
            if len(m):
                asym_ut[i] = m.iloc[0]["undertriage_rate"]

    def color_axis(axis):
        if "age" in axis and "sex" not in axis and "race" not in axis: return VG_HEADING
        if axis == "sex": return VG_LEAF
        if axis == "race": return VG_BLOSSOM
        return VG_SKY

    colors = [color_axis(r["axis"]) for _, r in std.iterrows()]

    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(std))
    ax.barh(y, std["undertriage_rate"] * 100, color=colors, alpha=0.85,
            label="Standard threshold")
    ax.barh(y, asym_ut * 100, color=colors, alpha=0.4, hatch="//",
            label="After asymmetric 2.5× threshold")
    ax.set_yticks(y); ax.set_yticklabels(labels_list, fontsize=9)
    ax.axvline(15.2, color="black", linestyle=":", alpha=0.6, label="Overall UT = 15.2%")
    ax.axvline(5, color="red", linestyle="--", alpha=0.6, label="ACS-COT 5% target")
    ax.set_xlabel("Under-triage rate (%)")
    ax.set_title("Intersectional Under-triage Disparities — MIMIC-IV-ED (Top 15 Subgroups)")
    ax.legend(loc="lower right", fontsize=9)
    save_fig(fig, "bias_audit_disparities")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Per-class F1 comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig_per_class_f1():
    path = RES_MIMIC / "ordinal_penalty_mimic_results.json"
    if not path.exists():
        placeholder("per_class_f1_comparison", "missing json"); return
    with open(path) as f:
        data = json.load(f)
    variants = data.get("variants", {})

    def get_pc(var_key):
        v = variants.get(var_key, {})
        pc = v.get("per_class_f1", {})
        return [pc.get(f"esi{i}", 0) for i in range(1, 6)]

    a = get_pc("a"); b = get_pc("b"); c = get_pc("c")

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(5); w = 0.27
    ax.bar(x - w, a, w, label="A: Standard", color=VG_HEADING)
    ax.bar(x, b, w, label="B: Asym weights", color=VG_BLOSSOM)
    ax.bar(x + w, c, w, label="C: Directional", color=COLOR_POS)
    for i, (av, bv, cv) in enumerate(zip(a, b, c)):
        ax.text(i - w, av + 0.01, f"{av:.2f}", ha="center", fontsize=8)
        ax.text(i, bv + 0.01, f"{bv:.2f}", ha="center", fontsize=8)
        ax.text(i + w, cv + 0.01, f"{cv:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"ESI-{i}" for i in range(1, 6)])
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 by Training Strategy — MIMIC-IV-ED")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "per_class_f1_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# 9. Class distribution comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig_class_distribution():
    kag_path = RES_V1 / "kaggle_class_distribution.csv"
    if not kag_path.exists():
        placeholder("class_distribution_comparison", "missing kaggle csv"); return
    kag = pd.read_csv(kag_path).sort_values("triage_acuity")
    mimic_pcts = [5.7, 33.3, 53.8, 6.8, 0.3]
    classes = list(range(1, 6))
    colors = [ESI_COLORS[c] for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    kag_pcts = kag["pct"].tolist()
    ax1.bar(classes, kag_pcts, color=colors)
    for c, p in zip(classes, kag_pcts):
        ax1.text(c, p + 1, f"{p:.1f}%", ha="center", fontsize=10)
    ax1.set_title("Kaggle (Synthetic)"); ax1.set_xlabel("ESI level"); ax1.set_ylabel("% of cases")
    ax1.set_xticks(classes)

    ax2.bar(classes, mimic_pcts, color=colors)
    for c, p in zip(classes, mimic_pcts):
        ax2.text(c, p + 1, f"{p:.1f}%", ha="center", fontsize=10)
    ax2.set_title("MIMIC-IV-ED (Real)"); ax2.set_xlabel("ESI level"); ax2.set_xticks(classes)

    fig.suptitle("ESI Distribution: Kaggle Synthetic vs MIMIC-IV-ED Real Data", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "class_distribution_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# 10. Pareto training vs threshold
# ═══════════════════════════════════════════════════════════════════════════
def fig_pareto():
    sweep_path = RES_V2 / "asymmetric_weight_sweep.csv"
    thresh_path = RES_MIMIC / "clinical_operating_curve_mimic.csv"
    if not sweep_path.exists() or not thresh_path.exists():
        placeholder("pareto_training_vs_threshold", "missing data"); return
    sweep = pd.read_csv(sweep_path)
    thresh = pd.read_csv(thresh_path)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(thresh["undertriage_rate"] * 100, thresh["macro_f1"],
            "o-", color=VG_HEADING, label="Post-hoc threshold shifting", markersize=5)
    ax.scatter(sweep["undertriage_rate"] * 100, sweep["macro_f1"],
               s=55, c=VG_BLOSSOM, alpha=0.4, label="Asymmetric training (all configs)")
    pareto = sweep[sweep["is_pareto_optimal"]] if "is_pareto_optimal" in sweep.columns else sweep
    ax.plot(pareto.sort_values("undertriage_rate")["undertriage_rate"] * 100,
            pareto.sort_values("undertriage_rate")["macro_f1"],
            "s-", color=COLOR_POS, markersize=10, linewidth=2,
            label="Asymmetric training — Pareto frontier")
    ax.axvline(5, color="red", linestyle="--", alpha=0.6, label="ACS-COT 5% target")
    ax.set_xlabel("Under-triage rate (%)"); ax.set_ylabel("Macro F1")
    ax.set_title("Two Paths to Clinical Safety — Training Weights vs Post-hoc Thresholds")
    ax.legend(loc="lower right", fontsize=10); ax.grid(True, alpha=0.3)
    save_fig(fig, "pareto_training_vs_threshold")


# ═══════════════════════════════════════════════════════════════════════════
# 11. Feature importance augmented vs baseline
# ═══════════════════════════════════════════════════════════════════════════
def fig_feature_importance_augmented():
    aug_path = RES_V2 / "augmented_feature_importance.csv"
    baseline_path = RES_V1 / "tabular_feature_importance.csv"
    if not aug_path.exists():
        placeholder("feature_importance_augmented_vs_baseline", "missing csv"); return
    aug = pd.read_csv(aug_path).head(20).iloc[::-1]
    new_feats = {"hr_age_zscore", "sbp_age_zscore", "dbp_age_zscore", "rr_age_zscore",
                 "o2sat_age_zscore", "temp_age_zscore", "n_age_abnormal_vitals", "rems_score",
                 "charlson_x_abnormal", "comorbid_tachycardia", "comorbid_hypotension",
                 "comorbid_hypoxia"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    if baseline_path.exists():
        base = pd.read_csv(baseline_path).head(20).iloc[::-1]
        axes[0].barh(np.arange(len(base)), base["gain"], color=VG_HEADING)
        axes[0].set_yticks(np.arange(len(base))); axes[0].set_yticklabels(base["feature"])
        axes[0].set_title("Baseline (65 features)"); axes[0].set_xlabel("Gain")
    else:
        axes[0].text(0.5, 0.5, "baseline unavailable", ha="center", va="center"); axes[0].axis("off")

    colors = [COLOR_POS if f in new_feats else VG_HEADING for f in aug["feature"]]
    axes[1].barh(np.arange(len(aug)), aug["gain"], color=colors)
    axes[1].set_yticks(np.arange(len(aug))); axes[1].set_yticklabels(aug["feature"])
    axes[1].set_title("Augmented (77 features — new in red)"); axes[1].set_xlabel("Gain")
    fig.suptitle("Feature Importance — Baseline vs Augmented Feature Set", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "feature_importance_augmented_vs_baseline")


# ═══════════════════════════════════════════════════════════════════════════
# 12. Calibration curves
# ═══════════════════════════════════════════════════════════════════════════
def fig_calibration_curves():
    """Calibration reliability diagrams — requires raw OOF probabilities."""
    # Check for pre-computed calibration curve data
    oof_path = RES_V2 / "mimic_oof_prob.npy"
    y_path = RES_V2 / "mimic_y_true.npy"
    if not oof_path.exists() or not y_path.exists():
        placeholder("calibration_curves", "OOF probabilities not available locally")
        return

    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    p_raw = np.load(oof_path)
    y = np.load(y_path)
    classes = np.sort(np.unique(y))
    K = len(classes)

    p_cal = np.zeros_like(p_raw)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, va in skf.split(p_raw, y):
        for k in range(K):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_raw[tr, k], (y[tr] == classes[k]).astype(int))
            p_cal[va, k] = iso.predict(p_raw[va, k])
        row_sum = p_cal[va].sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        p_cal[va] /= row_sum

    fig, axes = plt.subplots(1, K, figsize=(5 * K, 5))
    for k in range(K):
        ax = axes[k]
        target = (y == classes[k]).astype(int)
        frac_raw, mean_raw = calibration_curve(target, p_raw[:, k], n_bins=10)
        frac_cal, mean_cal = calibration_curve(target, p_cal[:, k], n_bins=10)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(mean_raw, frac_raw, "o-", label="Before", color=ESI_SEQ[k], alpha=0.7)
        ax.plot(mean_cal, frac_cal, "s-", label="After isotonic", color=ESI_SEQ[k])
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
        ax.set_title(f"ESI-{classes[k]}"); ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Reliability Diagrams — Before vs After Isotonic Calibration", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "calibration_curves")


# ═══════════════════════════════════════════════════════════════════════════
# 13. Operating curve — calibrated vs original
# ═══════════════════════════════════════════════════════════════════════════
def fig_operating_curve_calibrated():
    orig_path = RES_MIMIC / "clinical_operating_curve_mimic.csv"
    cal_path = RES_V2 / "calibrated_operating_curve.csv"
    if not orig_path.exists() or not cal_path.exists():
        placeholder("operating_curve_calibrated_vs_original", "missing data"); return
    orig = pd.read_csv(orig_path)
    cal = pd.read_csv(cal_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(orig["undertriage_rate"] * 100, orig["macro_f1"], "o-",
            color=VG_BLOSSOM, label="Original", markersize=5)
    ax.plot(cal["undertriage_rate"] * 100, cal["macro_f1"], "s-",
            color=VG_HEADING, label="Calibrated (isotonic)", markersize=5)
    ax.axvline(5, color="red", linestyle="--", alpha=0.5, label="ACS-COT 5% target")
    ax.set_xlabel("Under-triage rate (%)"); ax.set_ylabel("Macro F1")
    ax.set_title("Operating Curve — Original vs Calibrated Probabilities")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_fig(fig, "operating_curve_calibrated_vs_original")


# ═══════════════════════════════════════════════════════════════════════════
# 14. Age undertriage vitals comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig_age_undertriage_vitals():
    path = RES_V2 / "age_stratified_undertriage.json"
    if not path.exists():
        placeholder("age_undertriage_vitals_comparison", "missing json"); return
    with open(path) as f:
        data = json.load(f)

    age_rows = data["by_age_group"]
    comp_rows = data["compensated_shock_test"]
    groups = [r["age_group"] for r in age_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: compensated shock test
    x = np.arange(len(comp_rows))
    w = 0.25
    ax1.bar(x - w, [r["pct_normal_hr_and_sbp"] for r in comp_rows], w,
            label="Normal HR + SBP", color=VG_HEADING)
    ax1.bar(x, [r["pct_normal_hr"] for r in comp_rows], w,
            label="Normal HR only", color=VG_SKY)
    ax1.bar(x + w, [r["pct_normal_sbp"] for r in comp_rows], w,
            label="Normal SBP only", color=VG_LEAF)
    ax1.set_xticks(x); ax1.set_xticklabels([r["age_group"] for r in comp_rows])
    ax1.set_ylabel("% of high-acuity patients (ESI 1–2)")
    ax1.set_title("Compensated Shock — Normal Vitals Despite High Acuity")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # Right: mean vitals of under-triaged
    vitals = ["heartrate", "sbp", "resprate", "o2sat"]
    x = np.arange(len(vitals))
    w = 0.25
    for i, r in enumerate(age_rows):
        vals = [r[f"mean_{v}"] for v in vitals]
        ax2.bar(x + (i - 1) * w, vals, w, label=r["age_group"], color=ESI_SEQ[i])
    ax2.set_xticks(x); ax2.set_xticklabels(["HR", "SBP", "Resp rate", "SpO2"])
    ax2.set_ylabel("Mean value")
    ax2.set_title("Mean Vitals of Under-triaged Patients by Age")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Age-stratified Under-triage and Compensated Shock Analysis", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "age_undertriage_vitals_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# 15. SHAP bar — global
# ═══════════════════════════════════════════════════════════════════════════
def fig_shap_bar_global():
    sv_path = RES_V2 / "shap_values.npy"
    fn_path = RES_V2 / "shap_feature_names.json"
    if not sv_path.exists():
        placeholder("shap_bar_global", "missing shap_values.npy"); return
    sv = np.load(sv_path)
    with open(fn_path) as f:
        feature_names = json.load(f)
    # Global mean |SHAP| across all classes
    mean_abs = np.abs(sv).mean(axis=(0, 2))  # (115,)
    top_k = 20
    order = np.argsort(mean_abs)[::-1][:top_k]
    names = [feature_names[i] for i in order]
    vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(top_k)
    ax.barh(y, vals, color=VG_HEADING, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (global)")
    ax.set_title("Top 20 Features — Global SHAP Importance (MIMIC-IV-ED)")
    save_fig(fig, "shap_bar_global")


# ═══════════════════════════════════════════════════════════════════════════
# 16. SHAP beeswarm
# ═══════════════════════════════════════════════════════════════════════════
def fig_shap_beeswarm():
    sv_path = RES_V2 / "shap_values.npy"
    fn_path = RES_V2 / "shap_feature_names.json"
    x_path = RES_V2 / "shap_X_subsample.npy"
    if not sv_path.exists():
        placeholder("shap_beeswarm", "missing shap_values.npy"); return
    sv = np.load(sv_path)      # (20000, 115, 5)
    X = np.load(x_path)        # (20000, 115)
    with open(fn_path) as f:
        feature_names = json.load(f)

    # Aggregate across classes (sum) for beeswarm
    sv_agg = sv.sum(axis=2)    # (20000, 115)
    mean_abs = np.abs(sv_agg).mean(axis=0)
    top_k = 20
    order = np.argsort(mean_abs)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, 8))
    # Subsample for plotting speed
    n_plot = min(2000, sv_agg.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(sv_agg.shape[0], n_plot, replace=False)

    for rank, feat_i in enumerate(order[::-1]):
        vals = sv_agg[idx, feat_i]
        feat_data = X[idx, feat_i]
        # Normalize feature values for color mapping
        fmin, fmax = np.nanpercentile(feat_data, [2, 98])
        if fmax - fmin < 1e-8:
            norm_data = np.zeros_like(feat_data)
        else:
            norm_data = np.clip((feat_data - fmin) / (fmax - fmin), 0, 1)
        # Jitter y
        jitter = rng.uniform(-0.3, 0.3, size=len(vals))
        colors = plt.cm.coolwarm(norm_data)
        ax.scatter(vals, rank + jitter, c=colors, s=3, alpha=0.5, rasterized=True)

    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in order[::-1]])
    ax.set_xlabel("SHAP value (sum across classes)")
    ax.set_title("SHAP Beeswarm — Top 20 Features (MIMIC-IV-ED)")
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=mpl.colors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=10)

    save_fig(fig, "shap_beeswarm")


# ═══════════════════════════════════════════════════════════════════════════
# 17. SHAP bar — combined per-class (already exists, re-export)
# ═══════════════════════════════════════════════════════════════════════════
def fig_shap_bar_combined():
    sv_path = RES_V2 / "shap_values.npy"
    fn_path = RES_V2 / "shap_feature_names.json"
    if not sv_path.exists():
        placeholder("shap_bar_combined", "missing shap_values.npy"); return
    sv = np.load(sv_path)
    with open(fn_path) as f:
        feature_names = json.load(f)
    mean_abs = np.abs(sv).mean(axis=0)  # (115, 5)

    import matplotlib.gridspec as gridspec
    TOP_K = 10
    ESI_BAR_COLORS = [VG_BRANCH, VG_HEADING, VG_SKY, VG_LEAF, VG_BLOSSOM]
    esi_names = ["ESI-1 (Resuscitation)", "ESI-2 (Emergent)", "ESI-3 (Urgent)",
                 "ESI-4 (Less Urgent)", "ESI-5 (Non-Urgent)"]

    fig = plt.figure(figsize=(12, 14), facecolor="white")
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.40, left=0.10, right=0.95, top=0.94, bottom=0.04)

    for i in range(5):
        if i < 4:
            ax = fig.add_subplot(gs[i // 2, i % 2])
        else:
            gs_b = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2, :])
            ax = fig.add_subplot(gs_b[0, 1:3])
        vals = mean_abs[:, i]
        top_idx = np.argsort(vals)[::-1][:TOP_K]
        y = np.arange(TOP_K)
        ax.barh(y, vals[top_idx], color=ESI_BAR_COLORS[i], edgecolor="white", linewidth=0.3)
        ax.set_yticks(y); ax.set_yticklabels([feature_names[j] for j in top_idx], fontsize=10)
        ax.invert_yaxis(); ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.set_title(esi_names[i], fontsize=12, fontweight="bold", color=VG_HEADING)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Top 10 Features Driving Per-Class ESI Predictions (SHAP)",
                 fontsize=15, fontweight="bold", color=VG_HEADING, y=0.98)
    save_fig(fig, "shap_bar_combined")


# ═══════════════════════════════════════════════════════════════════════════
# 18. SHAP waterfall — per ESI
# ═══════════════════════════════════════════════════════════════════════════
def fig_shap_waterfalls():
    sv_path = RES_V2 / "shap_values.npy"
    fn_path = RES_V2 / "shap_feature_names.json"
    x_path = RES_V2 / "shap_X_subsample.npy"
    pt_path = RES_V2 / "shap_example_patients.csv"
    if not sv_path.exists():
        placeholder("shap_waterfall_esi1", "missing shap data"); return

    sv = np.load(sv_path)
    X = np.load(x_path)
    with open(fn_path) as f:
        feature_names = json.load(f)
    patients = pd.read_csv(pt_path)
    BASE_VALUES = np.array([-4.690, -1.380, -0.688, -4.825, -9.617])
    ESI_TITLES = {1: "ESI-1 (Resuscitation)", 2: "ESI-2 (Emergent)", 3: "ESI-3 (Urgent)",
                  4: "ESI-4 (Less Urgent)", 5: "ESI-5 (Non-Urgent)"}
    MAX_DISPLAY = 11
    COLOR_BASE = "#888888"

    def waterfall_panel(ax, sample_idx, class_idx):
        shap_v = sv[sample_idx, :, class_idx]
        xv = X[sample_idx]
        base = BASE_VALUES[class_idx]
        fx = base + shap_v.sum()

        order = np.argsort(np.abs(shap_v))[::-1]
        top_idx = order[:MAX_DISPLAY]
        rest_idx = order[MAX_DISPLAY:]
        rest_sum = shap_v[rest_idx].sum()

        names = [feature_names[i] for i in top_idx] + [f"{len(rest_idx)} other features"]
        values = [shap_v[i] for i in top_idx] + [rest_sum]
        feat_vals = [xv[i] for i in top_idx] + [None]
        n = len(names)

        cumulative = np.zeros(n + 1)
        cumulative[0] = fx
        for i in range(n):
            cumulative[i + 1] = cumulative[i] - values[i]

        y_pos = np.arange(n)
        bar_h = 0.6
        all_edges = np.concatenate([cumulative])
        x_span = all_edges.max() - all_edges.min()
        min_bar = x_span * 0.06

        for i in range(n):
            val = values[i]
            left = min(cumulative[i], cumulative[i + 1])
            width = abs(val)
            color = COLOR_POS if val > 0 else COLOR_NEG
            ax.barh(y_pos[i], width, left=left, height=bar_h, color=color,
                    edgecolor="white", linewidth=0.4)
            label = f"+{val:.2f}" if val > 0 else f"{val:.2f}"
            if width >= min_bar:
                ax.text(left + width / 2, y_pos[i], label, ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
            else:
                ax.text(left + width + x_span * 0.01, y_pos[i], label, ha="left",
                        va="center", fontsize=8, fontweight="bold", color=color)
            if i < n - 1:
                ax.plot([cumulative[i + 1], cumulative[i + 1]],
                        [y_pos[i] + bar_h / 2, y_pos[i + 1] - bar_h / 2],
                        color="#bbbbbb", linewidth=0.5, linestyle="--")

        y_labels = []
        for i in range(n):
            fv = feat_vals[i]
            nm = names[i]
            if fv is not None:
                fv_str = f"{fv:g}" if abs(fv) < 1000 else f"{fv:.0f}"
                y_labels.append(f"{fv_str} = {nm}")
            else:
                y_labels.append(nm)
        ax.set_yticks(y_pos); ax.set_yticklabels(y_labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel(f"$E[f(X)] = {base:.2f}$          $f(x) = {fx:.2f}$",
                      fontsize=9, color=COLOR_BASE, style="italic")
        ax.tick_params(axis="x", labelsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    for esi in range(1, 6):
        cls = esi - 1
        hc = patients[(patients["esi"] == esi) & (patients["type"] == "high_conf_correct")]
        mc = patients[(patients["esi"] == esi) & (patients["type"] == "misclassified")]
        if len(hc) == 0 or len(mc) == 0:
            continue
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="white")
        fig.subplots_adjust(wspace=0.45, left=0.12, right=0.97, top=0.88, bottom=0.08)
        waterfall_panel(ax_l, int(hc.iloc[0]["subsample_idx"]), cls)
        ax_l.set_title("High-Confidence Correct", fontsize=12, fontweight="bold", color=VG_HEADING, pad=6)
        waterfall_panel(ax_r, int(mc.iloc[0]["subsample_idx"]), cls)
        ax_r.set_title("Misclassified", fontsize=12, fontweight="bold", color=VG_HEADING, pad=6)
        fig.suptitle(f"SHAP Waterfall — {ESI_TITLES[esi]}", fontsize=14,
                     fontweight="bold", color=VG_HEADING, y=0.97)
        save_fig(fig, f"shap_waterfall_esi{esi}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print(f"[Figures] Output → {FIG_DIR}")
    figures = [
        ("clinical_operating_curve", fig_clinical_operating_curve),
        ("confusion_matrix_combined", fig_confusion_combined),
        ("keyword_leakage_heatmap", fig_keyword_leakage),
        ("feature_importance", fig_feature_importance),
        ("ordinal_penalty_comparison", fig_ordinal_penalty),
        ("bias_audit_disparities", fig_bias_audit_age),
        ("per_class_f1_comparison", fig_per_class_f1),
        ("class_distribution_comparison", fig_class_distribution),
        ("pareto_training_vs_threshold", fig_pareto),
        ("feature_importance_augmented", fig_feature_importance_augmented),
        ("calibration_curves", fig_calibration_curves),
        ("operating_curve_calibrated", fig_operating_curve_calibrated),
        ("age_undertriage_vitals", fig_age_undertriage_vitals),
        ("shap_bar_global", fig_shap_bar_global),
        ("shap_beeswarm", fig_shap_beeswarm),
        ("shap_bar_combined", fig_shap_bar_combined),
        ("shap_waterfalls", fig_shap_waterfalls),
    ]
    succ = fail = 0
    for name, fn in figures:
        try:
            fn()
            succ += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            fail += 1
    print(f"\n[Figures] {succ} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
