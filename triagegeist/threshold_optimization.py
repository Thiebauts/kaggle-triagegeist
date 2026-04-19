"""Multiplicative threshold optimisation, asymmetric sweeps, and clinical
operating curves.

The base model emits class probabilities. Two orthogonal mechanisms can
shift decisions toward safer (less under-triage) or more efficient (less
over-triage) operation:

1. **Training-time asymmetric weights** — per-sample weights applied
   when fitting base models, penalising under-triage errors more
   heavily.  Twenty hardcoded configurations are swept in
   :func:`asymmetric_weight_sweep`.
2. **Post-hoc multiplicative threshold shifts** — per-class multipliers
   ``m_c`` applied as ``adj = prob * m_c`` before argmax.  Optimised via
   Nelder-Mead on log-multipliers with 20 random restarts (initialised
   from Normal(0, 0.3)).

Both levers are individually effective; the empirical finding from this
project is that they are **substitute strategies, not complementary** —
threshold shifting undoes most of the safety benefit of asymmetric
weights.  See ``writeup/WRITEUP.md`` Section 4 for details.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .utils import compute_full_metrics


# ── 20 hardcoded asymmetric-weight configurations ─────────────────────────
# Each tuple: (name, [esi1_weight, esi2, esi3, esi4, esi5]).
# ESI 4 and 5 are always 1.0; higher-acuity classes get progressively
# larger weights to penalise under-triage during training.

WEIGHT_CONFIGS: list[tuple[str, list[float]]] = [
    ("Mild",      [1.5,  1.2, 1.0, 1.0, 1.0]),
    ("Mild+",     [2.0,  1.5, 1.0, 1.0, 1.0]),
    ("Moderate1", [2.5,  1.8, 1.2, 1.0, 1.0]),
    ("Moderate2", [3.0,  2.0, 1.0, 1.0, 1.0]),
    ("Moderate3", [3.0,  2.0, 1.5, 1.0, 1.0]),
    ("Strong1",   [4.0,  2.5, 1.5, 1.0, 1.0]),
    ("Strong2",   [4.0,  3.0, 1.5, 1.0, 1.0]),
    ("Strong3",   [5.0,  3.0, 1.0, 1.0, 1.0]),
    ("Strong4",   [5.0,  3.0, 2.0, 1.0, 1.0]),
    ("Strong5",   [5.0,  4.0, 2.0, 1.0, 1.0]),
    ("Heavy1",    [6.0,  4.0, 2.0, 1.0, 1.0]),
    ("Heavy2",    [7.0,  4.0, 2.5, 1.0, 1.0]),
    ("Heavy3",    [8.0,  5.0, 2.0, 1.0, 1.0]),
    ("Heavy4",    [8.0,  5.0, 3.0, 1.0, 1.0]),
    ("Extreme1",  [10.0, 6.0, 3.0, 1.0, 1.0]),
    ("Extreme2",  [10.0, 7.0, 4.0, 1.0, 1.0]),
    ("Extreme3",  [12.0, 8.0, 4.0, 1.0, 1.0]),
    ("Extreme4",  [15.0, 10.0, 5.0, 1.0, 1.0]),
    ("Max1",      [20.0, 12.0, 6.0, 1.0, 1.0]),
    ("Max2",      [25.0, 15.0, 8.0, 1.0, 1.0]),
]


# ── helpers ────────────────────────────────────────────────────────────────

def _apply_multipliers(
    probs: np.ndarray, multipliers: np.ndarray, classes: np.ndarray,
) -> np.ndarray:
    """Return argmax predictions after multiplicative per-class scaling."""
    adj = probs * multipliers[np.newaxis, :]
    return classes[np.argmax(adj, axis=1)]


# ── optimize_thresholds (multiplicative, macro-F1 objective) ──────────────

def optimize_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    objective: str = "macro_f1",
    n_restarts: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Nelder-Mead search for per-class multiplicative thresholds.

    Optimises in log-multiplier space: the search variables are
    ``log_m`` and predictions are ``argmax(prob * exp(log_m))``.
    Initialised with one all-zeros start plus ``n_restarts - 1`` draws
    from Normal(0, 0.3).

    Parameters
    ----------
    probs : ndarray (n, K)
        Out-of-fold class probabilities.
    y : ndarray (n,)
        True labels (integer ESI levels).
    classes : ndarray (K,)
        Sorted class labels matching the columns of *probs*.
    objective : str
        Only ``"macro_f1"`` was used in the Azure runs.
    n_restarts : int
        Number of random initialisations (default 20).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    multipliers : ndarray (K,)
        Per-class multipliers (already exponentiated from log space).
    metrics : dict
        Full metric suite computed with the optimised multipliers.
    """
    from scipy.optimize import minimize
    from sklearn.metrics import f1_score

    K = len(classes)

    def neg_f1(log_m: np.ndarray) -> float:
        adj = probs * np.exp(log_m)[np.newaxis, :]
        yp = classes[np.argmax(adj, axis=1)]
        return -f1_score(y, yp, average="macro", zero_division=0)

    best_f1, best_mult = -1.0, np.ones(K)
    rng = np.random.RandomState(seed)
    starts = [np.zeros(K)] + [rng.randn(K) * 0.3 for _ in range(n_restarts - 1)]

    for x0 in starts:
        res = minimize(
            neg_f1, x0, method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6},
        )
        if -res.fun > best_f1:
            best_f1 = -res.fun
            best_mult = np.exp(res.x)

    preds = _apply_multipliers(probs, best_mult, classes)
    metrics = compute_full_metrics(y, preds, probs, classes)
    metrics["multipliers"] = best_mult.tolist()
    metrics["objective"] = objective
    return best_mult, metrics


# ── optimize_thresholds_asymmetric ────────────────────────────────────────

def optimize_thresholds_asymmetric(
    probs: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    under_weight: float = 2.5,
    n_restarts: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Multiplicative thresholds with asymmetric under-triage penalty.

    Like :func:`optimize_thresholds` but the objective combines macro F1
    with a weighted penalty for under-triage errors::

        score = macro_f1 - 0.15 * cost

    where *cost* sums ordinal distances for misclassified samples,
    weighting under-triage by ``under_weight`` (default 2.5) and
    over-triage by 1.0, normalised by sample count.

    Parameters
    ----------
    probs : ndarray (n, K)
        Out-of-fold class probabilities.
    y : ndarray (n,)
        True labels (integer ESI levels).
    classes : ndarray (K,)
        Sorted class labels matching columns of *probs*.
    under_weight : float
        Penalty multiplier for under-triage errors (default 2.5).
    n_restarts : int
        Number of random initialisations (default 30).
    seed : int
        Random seed.

    Returns
    -------
    multipliers : ndarray (K,)
        Per-class multipliers (exponentiated).
    macro_f1 : float
        Macro F1 achieved with the optimised multipliers.
    """
    from scipy.optimize import minimize
    from sklearn.metrics import f1_score

    K = len(classes)

    def objective(log_m: np.ndarray) -> float:
        adj = probs * np.exp(log_m)[np.newaxis, :]
        yp = classes[np.argmax(adj, axis=1)]
        mf1 = f1_score(y, yp, average="macro", zero_division=0)
        diff = yp - y
        under_mask = diff > 0
        over_mask = diff < 0
        cost = (
            np.sum(diff[under_mask] * under_weight)
            + np.sum(np.abs(diff[over_mask]) * 1.0)
        ) / len(y)
        return -(mf1 - 0.15 * cost)

    best_score, best_mult = 1e9, np.ones(K)
    rng = np.random.RandomState(seed)
    starts = [np.zeros(K)] + [rng.randn(K) * 0.3 for _ in range(n_restarts - 1)]

    for x0 in starts:
        res = minimize(
            objective, x0, method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6},
        )
        if res.fun < best_score:
            best_score = res.fun
            best_mult = np.exp(res.x)

    preds = _apply_multipliers(probs, best_mult, classes)
    mf1 = f1_score(y, preds, average="macro", zero_division=0)
    return best_mult, mf1


# ── clinical_operating_curve ──────────────────────────────────────────────

def clinical_operating_curve(
    probs: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    """Sweep a single scalar multiplier to trace the safety-accuracy
    trade-off.

    For each multiplier value ``m`` in [1.0, 5.0] (step 0.25), per-class
    weights are computed as ``m^(K-1-k)`` for ``k`` in ``range(K)``.
    This amplifies higher-acuity (lower-index) classes more, shifting
    predictions toward safer (lower ESI) outputs.

    Returns a DataFrame with columns: multiplier, macro_f1, cohen_kappa,
    undertriage_rate, severe_undertriage_rate, overtriage_rate.
    """
    from sklearn.metrics import f1_score, cohen_kappa_score

    classes = np.asarray(classes)
    K = len(classes)
    rows: list[dict] = []

    for mult_val in np.arange(1.0, 5.01, 0.25):
        weights = np.array([mult_val ** (K - 1 - k) for k in range(K)])
        adj = probs * weights[np.newaxis, :]
        yp = classes[np.argmax(adj, axis=1)]
        diff = yp - y
        rows.append({
            "multiplier": round(float(mult_val), 3),
            "macro_f1": round(
                float(f1_score(y, yp, average="macro", zero_division=0)), 4,
            ),
            "cohen_kappa": round(
                float(cohen_kappa_score(y, yp, weights="quadratic")), 4,
            ),
            "undertriage_rate": round(float(np.mean(diff > 0)), 4),
            "severe_undertriage_rate": round(float(np.mean(diff >= 2)), 4),
            "overtriage_rate": round(float(np.mean(diff < 0)), 4),
        })

    return pd.DataFrame(rows)


# ── asymmetric_weight_sweep ───────────────────────────────────────────────

def asymmetric_weight_sweep(
    train_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    classes: np.ndarray,
    configurations: list[tuple[str, list[float]]] | None = None,
) -> pd.DataFrame:
    """Evaluate multiple per-class training-weight configurations.

    Each configuration retrains the full blend from scratch with
    per-sample weights derived from the class-level multipliers.

    Parameters
    ----------
    train_fn : callable
        Takes a per-sample weight array ``(n,)`` and returns
        ``(oof_probs, oof_preds)``.  Each call is a full retrain.
    y : ndarray (n,)
        True labels (integer ESI levels).
    classes : ndarray (K,)
        Sorted class labels.
    configurations : list of (name, weights), optional
        Each entry is ``(config_name, [esi1_w, ..., esi5_w])``.
        Defaults to the 20 hardcoded :data:`WEIGHT_CONFIGS`.

    Returns
    -------
    DataFrame with columns: config_name, esi{1..5}_w, macro_f1, kappa,
    undertriage_rate, severe_undertriage_rate, overtriage_rate,
    esi{1..5}_f1, is_pareto_optimal.
    """
    from sklearn.metrics import f1_score, cohen_kappa_score

    if configurations is None:
        configurations = WEIGHT_CONFIGS

    classes = np.asarray(classes)
    rows: list[dict] = []

    for name, weights in configurations:
        w_arr = np.array(weights, dtype=float)
        w_map = {c: w_arr[i] for i, c in enumerate(classes)}
        sample_w = np.array([w_map[c] for c in y])

        oof_probs, oof_preds = train_fn(sample_w)
        yp = oof_preds
        f1 = f1_score(y, yp, average="macro", zero_division=0)
        pc = f1_score(y, yp, labels=classes, average=None, zero_division=0)
        diff = yp - y

        rows.append({
            "config_name": name,
            "esi1_w": float(w_arr[0]),
            "esi2_w": float(w_arr[1]),
            "esi3_w": float(w_arr[2]),
            "esi4_w": float(w_arr[3]),
            "esi5_w": float(w_arr[4]),
            "macro_f1": round(float(f1), 4),
            "kappa": round(
                float(cohen_kappa_score(y, yp, weights="quadratic")), 4,
            ),
            "undertriage_rate": round(float(np.mean(diff > 0)), 4),
            "severe_undertriage_rate": round(float(np.mean(diff >= 2)), 4),
            "overtriage_rate": round(float(np.mean(diff < 0)), 4),
            "esi1_f1": round(float(pc[0]), 4),
            "esi2_f1": round(float(pc[1]), 4),
            "esi3_f1": round(float(pc[2]), 4),
            "esi4_f1": round(float(pc[3]), 4),
            "esi5_f1": round(float(pc[4]), 4),
        })

    sweep = pd.DataFrame(rows)

    # Pareto optimality: maximise macro_f1, minimise undertriage_rate
    def _is_pareto(idx: int) -> bool:
        a = sweep.iloc[idx]
        for j in range(len(sweep)):
            if j == idx:
                continue
            b = sweep.iloc[j]
            if (
                b["macro_f1"] >= a["macro_f1"]
                and b["undertriage_rate"] <= a["undertriage_rate"]
                and (
                    b["macro_f1"] > a["macro_f1"]
                    or b["undertriage_rate"] < a["undertriage_rate"]
                )
            ):
                return False
        return True

    sweep["is_pareto_optimal"] = [_is_pareto(i) for i in range(len(sweep))]
    return sweep


# ── pareto_frontier ───────────────────────────────────────────────────────

def pareto_frontier(
    df: pd.DataFrame, x_col: str, y_col: str, minimize_x: bool = True,
) -> pd.DataFrame:
    """Extract the Pareto-dominant rows for (x, y).

    Default: minimise under-triage (x) and maximise F1 (y).  A point is
    Pareto-dominant if no other point has both lower under-triage and
    higher F1.
    """
    x = df[x_col].to_numpy() * (1 if minimize_x else -1)
    yv = -df[y_col].to_numpy()
    order = np.argsort(x)
    out_idx: list[int] = []
    best_y = np.inf
    for i in order:
        if yv[i] < best_y:
            best_y = yv[i]
            out_idx.append(i)
    return df.iloc[sorted(out_idx)].reset_index(drop=True)
