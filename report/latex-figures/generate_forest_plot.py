"""Generate forest plot: inter-rater reliability benchmark for Chapter 7.

Plots published Cohen's kappa values from ESI and RETTS literature
alongside our model's result on MIMIC-IV-ED.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Almond Blossoms palette
# ---------------------------------------------------------------------------
VG_SKY = "#7BA7BC"
VG_HEADING = "#3D6B7E"
VG_BRANCH = "#5C4033"
VG_LEAF = "#6B8E6B"
VG_BLOSSOM = "#C4A882"
VG_PETAL = "#E8DDD3"

# Output directory derived from this script's location: <repo>/report/figures
REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = str(REPO / "report" / "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
studies = [
    # ESI block (top)
    {"label": "Tanabe et al. 2004 — ESI v3 validation",
     "kappa": 0.89, "ci_low": None, "ci_high": None,
     "system": "ESI", "n": None},
    {"label": "Mirhaghi 2015 — expert-to-expert",
     "kappa": 0.900, "ci_low": None, "ci_high": None,
     "system": "ESI", "n": None},
    {"label": "Mirhaghi 2015 — paper scenarios",
     "kappa": 0.824, "ci_low": None, "ci_high": None,
     "system": "ESI", "n": None},
    {"label": "Mirhaghi 2015 — pooled meta-analysis",
     "kappa": 0.791, "ci_low": 0.752, "ci_high": 0.825,
     "system": "ESI", "n": 40579, "diamond": True},
    {"label": "Mirhaghi 2015 — nurse-to-expert",
     "kappa": 0.732, "ci_low": None, "ci_high": None,
     "system": "ESI", "n": None},
    {"label": "Mirhaghi 2015 — live patients",
     "kappa": 0.694, "ci_low": None, "ci_high": None,
     "system": "ESI", "n": None},
    {"label": "Travers et al. 2009 — pediatric ESI, live",
     "kappa": 0.57, "ci_low": None, "ci_high": None,
     "system": "ESI", "n": None},
    # RETTS block (middle)
    {"label": "Widgren & Jourak 2011 — METTS validation",
     "kappa": 0.83, "ci_low": 0.761, "ci_high": 0.903,
     "system": "RETTS", "n": 132},
    {"label": "Nissen et al. 2014 — RETTS-HEV Denmark",
     "kappa": 0.60, "ci_low": None, "ci_high": None,
     "system": "RETTS", "n": 146},
    {"label": "Wireklint et al. 2018 — Swedish nurses",
     "kappa": 0.455, "ci_low": None, "ci_high": None,
     "system": "RETTS", "n": 1281},
    {"label": "Olsson et al. 2022 — student nurses",
     "kappa": 0.411, "ci_low": None, "ci_high": None,
     "system": "RETTS", "n": 2590},
    # Our result (bottom)
    {"label": "This work — MIMIC-IV-ED, 5-fold CV",
     "kappa": 0.701, "ci_low": None, "ci_high": None,
     "system": "ours", "n": 418100, "highlight": True},
]

# Reverse so first entry is at the top of the plot
studies = studies[::-1]
n = len(studies)

# ---------------------------------------------------------------------------
# Figure — wider for readability
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

y_positions = np.arange(n)

# Group separator positions (after reversing: ours=0, RETTS=1-4, ESI=5-11)
sep_after_ours = 0.5
sep_after_retts = 4.5

for i, s in enumerate(studies):
    y = y_positions[i]
    kappa = s["kappa"]
    ci_lo = s.get("ci_low")
    ci_hi = s.get("ci_high")
    is_diamond = s.get("diamond", False)
    is_highlight = s.get("highlight", False)
    system = s["system"]

    if is_highlight:
        color = VG_BLOSSOM
        marker = "s"
        ms = 10
        zorder = 10
    elif is_diamond:
        color = VG_BRANCH
        marker = "D"
        ms = 9
        zorder = 5
    elif system == "RETTS":
        color = VG_LEAF
        marker = "o"
        ms = 7
        zorder = 3
    else:
        color = VG_SKY
        marker = "o"
        ms = 7
        zorder = 3

    # Error bar
    if ci_lo is not None and ci_hi is not None:
        ax.errorbar(kappa, y, xerr=[[kappa - ci_lo], [ci_hi - kappa]],
                    fmt="none", ecolor=color, elinewidth=1.5, capsize=4,
                    capthick=1.0, zorder=zorder - 1)

    # Marker
    ax.plot(kappa, y, marker=marker, color=color, markersize=ms,
            markeredgecolor="white", markeredgewidth=0.6, zorder=zorder)

    # Kappa value to the right of each marker
    offset = 0.015
    if ci_hi is not None:
        offset = ci_hi - kappa + 0.015
    ax.text(kappa + offset, y, f"{kappa:.3f}",
            va="center", ha="left", fontsize=8, color=color)

    # N annotation on the far right
    n_val = s.get("n")
    if n_val is not None:
        n_str = f"N={n_val:,}"
    else:
        n_str = ""
    ax.text(1.04, y, n_str, va="center", ha="left",
            fontsize=7, color="#999999", transform=ax.get_yaxis_transform())

# Reference line at kappa = 0.694 (live patient benchmark)
ax.axvline(0.694, color=VG_BRANCH, linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
ax.text(0.694, n + 0.8, "Live patient\nhuman benchmark",
        ha="center", va="bottom", fontsize=8, color=VG_BRANCH, style="italic")

# Group separators
ax.axhline(sep_after_ours, color=VG_PETAL, linewidth=1.0, zorder=0)
ax.axhline(sep_after_retts, color=VG_PETAL, linewidth=1.0, zorder=0)

# Landis-Koch bands
ax.axvspan(0.61, 0.80, alpha=0.05, color=VG_SKY, zorder=0)
ax.axvspan(0.81, 1.02, alpha=0.05, color=VG_LEAF, zorder=0)
ax.text(0.705, n + 0.15, "Substantial", ha="center", fontsize=7,
        color="#aaaaaa", style="italic")
ax.text(0.91, n + 0.15, "Almost perfect", ha="center", fontsize=7,
        color="#aaaaaa", style="italic")

# Y-axis labels
ax.set_yticks(y_positions)
ax.set_yticklabels([s["label"] for s in studies], fontsize=9)
# Keep the bold weight on our entry so the reader's eye can find it, but
# render in the default text colour to avoid self-highlighting. The gold
# square marker continues to flag the row visually.
labels = ax.get_yticklabels()
labels[0].set_fontweight("bold")

# Legend
import matplotlib.lines as mlines
esi_handle = mlines.Line2D([], [], color=VG_SKY, marker="o", linestyle="None",
                           markersize=7, markeredgecolor="white", markeredgewidth=0.6,
                           label="ESI studies")
retts_handle = mlines.Line2D([], [], color=VG_LEAF, marker="o", linestyle="None",
                             markersize=7, markeredgecolor="white", markeredgewidth=0.6,
                             label="RETTS studies")
meta_handle = mlines.Line2D([], [], color=VG_BRANCH, marker="D", linestyle="None",
                            markersize=8, markeredgecolor="white", markeredgewidth=0.6,
                            label="Meta-analysis (pooled)")
ours_handle = mlines.Line2D([], [], color=VG_BLOSSOM, marker="s", linestyle="None",
                            markersize=9, markeredgecolor="white", markeredgewidth=0.6,
                            label="This work")
ax.legend(handles=[esi_handle, retts_handle, meta_handle, ours_handle],
          loc="upper left", fontsize=8, frameon=True, framealpha=0.9,
          edgecolor=VG_PETAL, fancybox=False)

ax.set_xlabel("Cohen's $\\kappa$", fontsize=11, color=VG_HEADING)
ax.set_xlim(0.35, 1.02)
ax.set_ylim(-0.6, n + 1.8)
ax.grid(axis="x", alpha=0.2, color=VG_PETAL, linewidth=0.5)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)

plt.tight_layout()

fig.savefig(f"{OUT_DIR}/reliability_benchmark_forest.pdf", bbox_inches="tight")
print("Saved: reliability_benchmark_forest.pdf")
plt.close(fig)
