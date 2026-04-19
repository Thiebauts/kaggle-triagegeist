"""Intersectional fairness audit for the triage model.

Computes per-subgroup classification metrics across three protected
axes -- age band, sex, and race -- and their two- and three-way
intersections.  Reports disparity ratios (max / min across subgroups
with n >= min_n) with a minimum subgroup size of 50 to keep the
ratio statistically meaningful.

Mirrors the actual audit in ``train_mimic_overnight_v3.task_9_bias_audit``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# ── Demographics helpers ──────────────────────────────────────────────────────

AGE_BINS = [0, 30, 50, 65, 80, 200]
AGE_LABELS = ["18-30", "31-50", "51-65", "66-80", "80+"]

RACE_MAP: dict[str, str] = {
    "WHITE": "WHITE",
    "BLACK/AFRICAN AMERICAN": "BLACK",
    "BLACK": "BLACK",
    "HISPANIC/LATINO": "HISPANIC",
    "HISPANIC OR LATINO": "HISPANIC",
    "ASIAN": "ASIAN",
}

MIN_SUBGROUP_N = 50


def add_age_group(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    """Add ``age_group`` column using the MIMIC age bins."""
    out = df.copy()
    out["age_group"] = pd.cut(
        out[age_col], bins=AGE_BINS, labels=AGE_LABELS,
    )
    return out


def collapse_race(race_series: pd.Series) -> np.ndarray:
    """Collapse MIMIC's ~33 raw race values to 5 categories.

    Uses a simple dictionary lookup; anything not in the map becomes
    ``OTHER``.
    """
    raw = race_series.fillna("UNKNOWN").astype(str).values
    return np.array([RACE_MAP.get(r.upper(), "OTHER") for r in raw])


# ── Per-subgroup metrics ──────────────────────────────────────────────────────

def _metrics_for_mask(
    y: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    min_n: int = MIN_SUBGROUP_N,
) -> dict | None:
    """Compute metrics for a boolean-masked subgroup.

    Returns ``None`` if the subgroup has fewer than *min_n* samples.
    """
    if mask.sum() < min_n:
        return None
    sub_y = y[mask]
    sub_p = y_pred[mask]
    diff = sub_p - sub_y
    return {
        "n": int(mask.sum()),
        "undertriage_rate": float(round(np.mean(diff > 0), 4)),
        "severe_undertriage_rate": float(round(np.mean(diff >= 2), 4)),
        "overtriage_rate": float(round(np.mean(diff < 0), 4)),
        "macro_f1": float(round(
            f1_score(sub_y, sub_p, average="macro", zero_division=0), 4,
        )),
    }


# ── Core audit ────────────────────────────────────────────────────────────────

def subgroup_audit(
    demographics: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axes: Sequence[str] = ("age_group", "sex", "race"),
    min_n: int = MIN_SUBGROUP_N,
) -> pd.DataFrame:
    """Run single-axis and intersectional fairness audit.

    Parameters
    ----------
    demographics : DataFrame
        Must contain an ``age`` column (numeric), a ``gender`` column,
        and a ``race`` column with raw MIMIC race strings.  The function
        derives ``age_group`` and collapsed ``race`` internally.
    y_true, y_pred : array-like
        True and predicted ESI labels, row-aligned with *demographics*.
    axes : sequence of str
        Protected attribute column names to audit.
    min_n : int
        Subgroups with fewer than *min_n* samples are excluded.

    Returns
    -------
    pd.DataFrame
        One row per subgroup (single-axis + two-way + three-way),
        with columns: level, axis, subgroup, n, undertriage_rate,
        severe_undertriage_rate, overtriage_rate, macro_f1.
    """
    y = np.asarray(y_true)
    yp = np.asarray(y_pred)

    # Build derived demographic arrays
    age_bin = pd.cut(
        demographics["age"].values, bins=AGE_BINS, labels=AGE_LABELS,
    )
    sex = demographics["gender"].fillna("UNKNOWN").astype(str).values
    race = collapse_race(demographics["race"])

    axis_arrays: dict[str, np.ndarray] = {
        "age_group": np.asarray(age_bin),
        "sex": sex,
        "race": race,
    }

    rows: list[dict] = []

    # Single-axis
    for axis_name in axes:
        vals = axis_arrays[axis_name]
        for val in pd.Series(vals).dropna().unique():
            if axis_name == "age_group":
                mask = (np.array(age_bin) == val) & ~pd.isna(age_bin)
            else:
                mask = vals == val
            m = _metrics_for_mask(y, yp, mask, min_n)
            if m:
                rows.append({
                    "level": 1, "axis": axis_name,
                    "subgroup": str(val), **m,
                })

    # Two-way intersections
    for ag in pd.Series(age_bin).dropna().unique():
        for s in pd.Series(sex).dropna().unique():
            mask = (np.array(age_bin) == ag) & (sex == s)
            m = _metrics_for_mask(y, yp, mask, min_n)
            if m:
                rows.append({
                    "level": 2, "axis": "age_group+sex",
                    "subgroup": f"{ag}|{s}", **m,
                })

    for ag in pd.Series(age_bin).dropna().unique():
        for r in pd.Series(race).dropna().unique():
            mask = (np.array(age_bin) == ag) & (race == r)
            m = _metrics_for_mask(y, yp, mask, min_n)
            if m:
                rows.append({
                    "level": 2, "axis": "age_group+race",
                    "subgroup": f"{ag}|{r}", **m,
                })

    for s in pd.Series(sex).dropna().unique():
        for r in pd.Series(race).dropna().unique():
            mask = (sex == s) & (race == r)
            m = _metrics_for_mask(y, yp, mask, min_n)
            if m:
                rows.append({
                    "level": 2, "axis": "sex+race",
                    "subgroup": f"{s}|{r}", **m,
                })

    # Three-way intersections
    for ag in pd.Series(age_bin).dropna().unique():
        for s in pd.Series(sex).dropna().unique():
            for r in pd.Series(race).dropna().unique():
                mask = (np.array(age_bin) == ag) & (sex == s) & (race == r)
                m = _metrics_for_mask(y, yp, mask, min_n)
                if m:
                    rows.append({
                        "level": 3, "axis": "age_group+sex+race",
                        "subgroup": f"{ag}|{s}|{r}", **m,
                    })

    return pd.DataFrame(rows)


# ── Disparity helpers ─────────────────────────────────────────────────────────

def _disparity(
    frame: pd.DataFrame,
    metric: str,
    min_n: int = MIN_SUBGROUP_N,
) -> float:
    """Disparity ratio = max / min of *metric* across subgroups with n >= min_n.

    Returns ``nan`` when fewer than two qualifying subgroups exist or
    the minimum value is zero.
    """
    if frame.empty or "n" not in frame.columns or metric not in frame.columns:
        return float("nan")
    f = frame[frame["n"] >= min_n]
    if len(f) < 2 or f[metric].min() == 0:
        return float("nan")
    return float(f[metric].max() / f[metric].min())


def disparity_summary(
    audit_df: pd.DataFrame,
    min_n: int = MIN_SUBGROUP_N,
) -> pd.DataFrame:
    """Compute disparity ratios per intersection level.

    Returns a DataFrame with columns: breakdown, f1_disparity_ratio,
    undertriage_disparity_ratio.
    """
    out = []
    for level, label in [(1, "single_axis"), (2, "two_way"), (3, "three_way")]:
        sub = audit_df[audit_df["level"] == level]
        out.append({
            "breakdown": label,
            "f1_disparity_ratio": _disparity(sub, "macro_f1", min_n),
            "undertriage_disparity_ratio": _disparity(
                sub, "undertriage_rate", min_n,
            ),
        })
    return pd.DataFrame(out)


def disparity_delta(
    baseline_audit: pd.DataFrame,
    adjusted_audit: pd.DataFrame,
    min_n: int = MIN_SUBGROUP_N,
) -> pd.DataFrame:
    """Compare disparity ratios before and after a safety intervention.

    Returns a DataFrame with ``delta = adjusted - baseline`` per metric;
    positive ``undertriage_disparity_delta`` means the intervention
    *worsens* group fairness on the safety axis.
    """
    base = disparity_summary(baseline_audit, min_n).set_index("breakdown")
    adj = disparity_summary(adjusted_audit, min_n).set_index("breakdown")
    out = (adj - base).rename(columns={
        "f1_disparity_ratio": "f1_disparity_delta",
        "undertriage_disparity_ratio": "undertriage_disparity_delta",
    })
    return out.reset_index()
