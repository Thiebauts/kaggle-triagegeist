"""SHAP (Shapley additive explanations) analysis for the blend.

Uses :class:`shap.TreeExplainer` on the LightGBM component of the
blend. SHAP values are computed on a stratified subsample (default
20 000 rows) for tractability; the results are representative of global
feature attribution given the blend's agreement with the LightGBM
component.

Supports three views:

* Global summary (bar + beeswarm) — feature importance ranked by mean
  ``|SHAP|``.
* Per-class bar — attribution for each ESI level separately.
* Waterfall — per-patient explanation, typically for three selected
  cohorts (high-confidence correct, low-confidence correct,
  misclassified).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ShapBundle:
    values: np.ndarray          # (n, n_features, n_classes) or (n, n_features)
    X: np.ndarray               # (n, n_features)
    feature_names: list[str]
    base_values: np.ndarray | float
    classes: np.ndarray


def compute_tree_shap(
    estimator,
    X_subsample: np.ndarray,
    feature_names: list[str],
    classes: np.ndarray,
) -> ShapBundle:
    """Compute TreeSHAP values for a tree ensemble classifier.

    Works for any :mod:`lightgbm`, :mod:`xgboost`, or :mod:`catboost`
    classifier exposing the scikit-learn API. For multiclass models,
    returns per-class SHAP tensors of shape ``(n, n_features, n_classes)``.
    """
    import shap

    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(X_subsample)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    base = explainer.expected_value
    return ShapBundle(
        values=values, X=X_subsample,
        feature_names=list(feature_names),
        base_values=base, classes=np.asarray(classes),
    )


def global_summary(
    bundle: ShapBundle,
    output_path: Path | str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Plot + return a mean-|SHAP| feature importance table.

    For multiclass SHAP tensors, aggregates by summing across classes.
    Saves a bar plot to `output_path` (PNG). Returns the ranking as a
    DataFrame with columns: feature, mean_abs_shap.
    """
    import matplotlib.pyplot as plt

    if bundle.values.ndim == 3:
        abs_vals = np.abs(bundle.values).mean(axis=(0, 2))
    else:
        abs_vals = np.abs(bundle.values).mean(axis=0)

    order = np.argsort(abs_vals)[::-1]
    rows = [(bundle.feature_names[i], float(abs_vals[i])) for i in order]
    frame = pd.DataFrame(rows, columns=["feature", "mean_abs_shap"])

    top = frame.head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * top_n)))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#475569")
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title(f"Top {top_n} features by global SHAP importance")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return frame


def per_class_bar(
    bundle: ShapBundle,
    output_path: Path | str,
    top_n: int = 15,
) -> pd.DataFrame:
    """Per-class top-N SHAP feature attribution.

    Only meaningful for multiclass bundles. Returns a long-format frame
    with columns (esi_class, feature, mean_abs_shap) and saves a faceted
    bar plot.
    """
    import matplotlib.pyplot as plt

    if bundle.values.ndim != 3:
        raise ValueError("per_class_bar requires a multiclass SHAP bundle")

    records = []
    for k, cls in enumerate(bundle.classes):
        m = np.abs(bundle.values[:, :, k]).mean(axis=0)
        order = np.argsort(m)[::-1][:top_n]
        for i in order:
            records.append({
                "esi_class": int(cls),
                "feature": bundle.feature_names[i],
                "mean_abs_shap": float(m[i]),
            })
    frame = pd.DataFrame(records)

    fig, axes = plt.subplots(
        1, len(bundle.classes),
        figsize=(4 * len(bundle.classes), max(4, 0.25 * top_n)),
        sharey=False,
    )
    for k, (cls, ax) in enumerate(zip(bundle.classes, axes.flat)):
        sub = frame[frame["esi_class"] == int(cls)]
        ax.barh(sub["feature"][::-1], sub["mean_abs_shap"][::-1],
                color="#475569")
        ax.set_title(f"ESI {cls}")
        ax.set_xlabel("mean(|SHAP|)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return frame


def select_waterfall_patients(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    n_per_cohort: int = 3,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Pick patient indices for three illustrative waterfall cohorts.

    * ``high_confidence_correct`` — argmax ≈ truth, max-prob > 0.8
    * ``low_confidence_correct`` — argmax ≈ truth, max-prob < 0.5
    * ``misclassified_severe`` — truth ∈ {1, 2}, argmax ≥ 3
      (severe under-triage)
    """
    rng = np.random.default_rng(seed)
    argmax = np.argmax(oof_probs, axis=1) + 1  # classes are 1-indexed
    max_p = oof_probs.max(axis=1)

    correct = argmax == y_true
    hi = np.where(correct & (max_p > 0.8))[0]
    lo = np.where(correct & (max_p < 0.5))[0]
    severe_ut = np.where((np.isin(y_true, [1, 2])) & (argmax >= 3))[0]

    return {
        "high_confidence_correct": rng.choice(
            hi, size=min(n_per_cohort, len(hi)), replace=False,
        ) if len(hi) else np.array([], dtype=int),
        "low_confidence_correct": rng.choice(
            lo, size=min(n_per_cohort, len(lo)), replace=False,
        ) if len(lo) else np.array([], dtype=int),
        "misclassified_severe_undertriage": rng.choice(
            severe_ut, size=min(n_per_cohort, len(severe_ut)), replace=False,
        ) if len(severe_ut) else np.array([], dtype=int),
    }


def save_waterfall(
    bundle: ShapBundle, row: int, esi_class: int, output_path: Path | str,
) -> None:
    """Save a single SHAP waterfall plot for one patient / one class."""
    import matplotlib.pyplot as plt
    import shap

    if bundle.values.ndim == 3:
        k = list(bundle.classes).index(esi_class)
        vals = bundle.values[row, :, k]
        base = (
            bundle.base_values[k]
            if isinstance(bundle.base_values, (list, np.ndarray))
            else bundle.base_values
        )
    else:
        vals = bundle.values[row]
        base = bundle.base_values

    shap_exp = shap.Explanation(
        values=vals, base_values=base,
        data=bundle.X[row], feature_names=bundle.feature_names,
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
