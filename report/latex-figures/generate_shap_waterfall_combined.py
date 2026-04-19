"""Generate per-ESI SHAP waterfall figures from raw data (vector graphics).

One figure per ESI level: high-conf correct (left) vs misclassified (right).
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Palette (Almond Blossoms)
# ---------------------------------------------------------------------------
VG_HEADING = "#3D6B7E"
COLOR_POS = "#E8525A"   # warm red for positive SHAP
COLOR_NEG = "#5B9BD5"   # blue for negative SHAP
COLOR_BASE = "#888888"  # grey for annotations

# ---------------------------------------------------------------------------
# Data paths
# Input: heavy .npy files NOT in the repo. Override with the
# TRIAGEGEIST_OVERNIGHT_V2 environment variable if your data lives elsewhere.
# Output: always written to <repo>/report/figures.
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get(
    "TRIAGEGEIST_OVERNIGHT_V2",
    str(Path.home() / "Documents/Kaggle/Triagegeist/overnight_results_v2"),
)
OUT_DIR = str(Path(__file__).resolve().parent.parent.parent / "report" / "figures")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
shap_values = np.load(f"{DATA_DIR}/shap_values.npy")          # (20000, 115, 5)
X_sub = np.load(f"{DATA_DIR}/shap_X_subsample.npy")           # (20000, 115)
with open(f"{DATA_DIR}/shap_feature_names.json") as f:
    feature_names = json.load(f)
patients = pd.read_csv(f"{DATA_DIR}/shap_example_patients.csv")

# Base values (verified against original waterfall images)
BASE_VALUES = np.array([-4.690, -1.380, -0.688, -4.825, -9.617])

ESI_TITLES = {
    1: "ESI-1 (Resuscitation)",
    2: "ESI-2 (Emergent)",
    3: "ESI-3 (Urgent)",
    4: "ESI-4 (Less Urgent)",
    5: "ESI-5 (Non-Urgent)",
}

MAX_DISPLAY = 11


def waterfall_panel(ax, sample_idx: int, class_idx: int, max_display: int = MAX_DISPLAY):
    """Draw a single waterfall plot on the given axes."""
    sv = shap_values[sample_idx, :, class_idx]
    xv = X_sub[sample_idx]
    base = BASE_VALUES[class_idx]
    fx = base + sv.sum()

    # Sort by absolute value, keep top features
    order = np.argsort(np.abs(sv))[::-1]
    top_idx = order[:max_display]
    rest_idx = order[max_display:]
    rest_sum = sv[rest_idx].sum()

    names = [feature_names[i] for i in top_idx] + [f"{len(rest_idx)} other features"]
    values = [sv[i] for i in top_idx] + [rest_sum]
    feat_vals = [xv[i] for i in top_idx] + [None]

    n = len(names)

    # Waterfall: start from f(x) at top, subtract contributions going down
    cumulative = np.zeros(n + 1)
    cumulative[0] = fx
    for i in range(n):
        cumulative[i + 1] = cumulative[i] - values[i]

    y_pos = np.arange(n)
    bar_h = 0.6

    # Need x-range to decide label placement — compute it first
    all_edges = np.concatenate([cumulative, [cumulative.min(), cumulative.max()]])
    x_span = all_edges.max() - all_edges.min()
    min_bar_for_inside = x_span * 0.06  # bars narrower than this get outside labels

    for i in range(n):
        val = values[i]
        left = min(cumulative[i], cumulative[i + 1])
        width = abs(val)
        color = COLOR_POS if val > 0 else COLOR_NEG

        ax.barh(y_pos[i], width, left=left, height=bar_h, color=color,
                edgecolor="white", linewidth=0.4)

        # Value label — inside if bar is wide enough, else outside
        label = f"+{val:.2f}" if val > 0 else f"{val:.2f}"
        if width >= min_bar_for_inside:
            ax.text(left + width / 2, y_pos[i], label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white")
        else:
            # Place outside, to the right of the bar
            offset = x_span * 0.01
            ax.text(left + width + offset, y_pos[i], label, ha="left", va="center",
                    fontsize=6, fontweight="bold", color=color)

        # Connector line
        if i < n - 1:
            ax.plot([cumulative[i + 1], cumulative[i + 1]],
                    [y_pos[i] + bar_h / 2, y_pos[i + 1] - bar_h / 2],
                    color="#bbbbbb", linewidth=0.5, linestyle="--")

    # Feature labels
    y_labels = []
    for i in range(n):
        fv = feat_vals[i]
        nm = names[i]
        if fv is not None:
            fv_str = f"{fv:g}" if abs(fv) < 1000 else f"{fv:.0f}"
            y_labels.append(f"{fv_str} = {nm}")
        else:
            y_labels.append(nm)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()

    # f(x) and E[f(X)] annotations — placed as xlabel-style below the axis
    ax.set_xlabel(f"$E[f(X)] = {base:.2f}$          $f(x) = {fx:.2f}$",
                  fontsize=7, color=COLOR_BASE, style="italic")

    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)


# ---------------------------------------------------------------------------
# Generate one figure per ESI level
# ---------------------------------------------------------------------------
for esi in range(1, 6):
    class_idx = esi - 1

    hc = patients[(patients["esi"] == esi) & (patients["type"] == "high_conf_correct")]
    mc = patients[(patients["esi"] == esi) & (patients["type"] == "misclassified")]
    if len(hc) == 0 or len(mc) == 0:
        continue

    hc_idx = int(hc.iloc[0]["subsample_idx"])
    mc_idx = int(mc.iloc[0]["subsample_idx"])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="white")
    fig.subplots_adjust(wspace=0.45, left=0.12, right=0.97, top=0.88, bottom=0.08)

    waterfall_panel(ax_l, hc_idx, class_idx)
    ax_l.set_title("High-Confidence Correct", fontsize=10,
                    fontweight="bold", color=VG_HEADING, pad=6)

    waterfall_panel(ax_r, mc_idx, class_idx)
    ax_r.set_title("Misclassified", fontsize=10,
                    fontweight="bold", color=VG_HEADING, pad=6)

    fig.suptitle(
        f"SHAP Waterfall — {ESI_TITLES[esi]}",
        fontsize=12, fontweight="bold", color=VG_HEADING, y=0.97,
    )

    stem = f"shap_waterfall_esi{esi}"
    fig.savefig(f"{OUT_DIR}/{stem}.pdf", bbox_inches="tight")
    print(f"Saved: {stem}.pdf")
    plt.close(fig)

# Clean up the old combined file
import os
for ext in (".pdf",):
    path = f"{OUT_DIR}/shap_waterfall_combined{ext}"
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed: shap_waterfall_combined{ext}")
