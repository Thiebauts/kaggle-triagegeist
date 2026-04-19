"""Shared utilities: plotting style, metric helpers, colour palette."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# Consistent colour palette for ESI 1–5 across all visualisations.
# Red → Green gradient reflects clinical urgency.
ESI_COLOURS: dict[int, str] = {
    1: "#ef4444",  # red     — resuscitation
    2: "#f97316",  # orange  — emergent
    3: "#eab308",  # yellow  — urgent
    4: "#14b8a6",  # teal    — less urgent
    5: "#22c55e",  # green   — non-urgent
}


def configure_matplotlib_style() -> None:
    """Apply publication-quality matplotlib defaults.

    Call once at the start of a plotting script. Sets font sizes, grid
    alpha, figure DPI, and consistent default colours.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def under_triage_rate(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute under-triage rates.

    Under-triage = predicted acuity is *higher-numbered* (less urgent)
    than true acuity. Severe under-triage = prediction overshoots true
    acuity by 2 or more levels.

    Returns a dict with keys: overall, severe.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    overall = float(np.mean(y_pred > y_true))
    severe = float(np.mean((y_pred - y_true) >= 2))
    return {"overall": overall, "severe": severe}


def over_triage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Over-triage = predicted more urgent (lower number) than true."""
    return float(np.mean(np.asarray(y_pred) < np.asarray(y_true)))


def near_miss_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions within one ESI level of ground truth."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_pred - y_true) <= 1))


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    classes: Sequence[int] = (1, 2, 3, 4, 5),
) -> dict:
    """Return the full metric suite used throughout the pipeline.

    Key names match the actual training code in
    ``train_mimic_overnight_v2.compute_metrics``: ``macro_f1``,
    ``weighted_f1``, ``kappa``, ``auc``, ``bal_acc``, ``near_miss``,
    ``per_class_f1``, and ``confusion``.
    """
    from sklearn.metrics import (
        f1_score, cohen_kappa_score, balanced_accuracy_score,
        confusion_matrix, roc_auc_score,
    )
    classes = list(classes)
    out: dict = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "near_miss": near_miss_rate(y_true, y_pred),
        "per_class_f1": f1_score(
            y_true, y_pred, average=None, labels=classes, zero_division=0,
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=classes),
    }
    if y_prob is not None:
        out["auc"] = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro",
        )
    return out


def stratified_sample(
    df: pd.DataFrame, strat_col: str, n_per_stratum: int, seed: int = 42,
) -> pd.DataFrame:
    """Sample `n_per_stratum` rows from each value of `strat_col`.

    Useful for building balanced subsamples for SHAP analysis.
    """
    return df.groupby(strat_col, group_keys=False).apply(
        lambda g: g.sample(min(len(g), n_per_stratum), random_state=seed)
    )
