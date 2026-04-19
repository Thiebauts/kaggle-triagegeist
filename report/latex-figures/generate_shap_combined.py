"""Generate combined SHAP per-class bar chart (2-2-1 layout).

Reads the raw SHAP tensor from the overnight results and produces
a single multi-panel figure for the LaTeX report.
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Almond Blossoms palette (from thiebaut-report.sty)
# ---------------------------------------------------------------------------
VG_SKY = "#7BA7BC"
VG_HEADING = "#3D6B7E"
VG_BRANCH = "#5C4033"
VG_LEAF = "#6B8E6B"
VG_BLOSSOM = "#C4A882"

# One colour per ESI level
ESI_COLORS = [VG_BRANCH, VG_HEADING, VG_SKY, VG_LEAF, VG_BLOSSOM]

TOP_K = 10  # features per panel

# ---------------------------------------------------------------------------
# Load raw SHAP data — heavy .npy files NOT in the repo. Override path with
# TRIAGEGEIST_OVERNIGHT_V2 environment variable if your data lives elsewhere.
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get(
    "TRIAGEGEIST_OVERNIGHT_V2",
    str(Path.home() / "Documents/Kaggle/Triagegeist/overnight_results_v2"),
)

shap_values = np.load(f"{DATA_DIR}/shap_values.npy")  # (20000, 115, 5)
with open(f"{DATA_DIR}/shap_feature_names.json") as f:
    feature_names = json.load(f)

# Per-class mean |SHAP| → shape (115, 5)
mean_abs = np.abs(shap_values).mean(axis=0)


def plot_panel(ax, title, class_idx, color):
    """Draw a single horizontal bar chart for one ESI class."""
    vals = mean_abs[:, class_idx]
    top_idx = np.argsort(vals)[::-1][:TOP_K]
    top_features = [feature_names[i] for i in top_idx]
    top_values = vals[top_idx]

    y = np.arange(TOP_K)
    ax.barh(y, top_values, color=color, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(top_features, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", color=VG_HEADING)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Build figure: 3 rows × 2 columns, last row centred
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 14), facecolor="white")
gs = gridspec.GridSpec(
    3, 2,
    hspace=0.35,
    wspace=0.40,
    left=0.10,
    right=0.95,
    top=0.94,
    bottom=0.04,
)

# Row 1: ESI-1 (left), ESI-2 (right)
ax1 = fig.add_subplot(gs[0, 0])
plot_panel(ax1, "ESI-1 (Resuscitation)", class_idx=0, color=ESI_COLORS[0])

ax2 = fig.add_subplot(gs[0, 1])
plot_panel(ax2, "ESI-2 (Emergent)", class_idx=1, color=ESI_COLORS[1])

# Row 2: ESI-3 (left), ESI-4 (right)
ax3 = fig.add_subplot(gs[1, 0])
plot_panel(ax3, "ESI-3 (Urgent)", class_idx=2, color=ESI_COLORS[2])

ax4 = fig.add_subplot(gs[1, 1])
plot_panel(ax4, "ESI-4 (Less Urgent)", class_idx=3, color=ESI_COLORS[3])

# Row 3: ESI-5 centred
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2, :])
ax5 = fig.add_subplot(gs_bottom[0, 1:3])
plot_panel(ax5, "ESI-5 (Non-Urgent)", class_idx=4, color=ESI_COLORS[4])

fig.suptitle(
    "Top 10 Features Driving Per-Class ESI Predictions (SHAP)",
    fontsize=13,
    fontweight="bold",
    color=VG_HEADING,
    y=0.98,
)

# Save
out_dir = str(Path(__file__).resolve().parent.parent.parent / "report" / "figures")
fig.savefig(f"{out_dir}/shap_bar_combined.pdf", bbox_inches="tight")
print("Saved: shap_bar_combined.pdf")
plt.close(fig)
