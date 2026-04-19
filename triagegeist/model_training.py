"""Model training: 5-model blend with nested stacking.

The Triagegeist production configuration is a softmax-weighted convex
blend of five base learners, each evaluated via 5-fold stratified CV
with out-of-fold (OOF) probability collection:

* **baseline** -- standard LightGBM (200 trees, lr=0.05)
* **weighted** -- LightGBM with inverse-frequency class weights
* **more_trees** -- LightGBM (800 trees, lr=0.02)
* **early_stop** -- LightGBM (2000 trees, lr=0.01, early stopping patience=100)
* **stacked** -- LGB + XGB + CatBoost meta-features -> LogisticRegression
  (5 outer x 3 inner CV)

Blend weights are found by Nelder-Mead optimisation of macro F1 over
the OOF probability matrices, using softmax reparameterisation to
enforce a valid convex combination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .utils import compute_full_metrics


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
# LightGBM defaults used for all four LGB-based blend members.
# No reg_alpha / reg_lambda / min_child_samples overrides -- LightGBM's own
# defaults apply for those.
DEFAULT_LGBM = dict(
    n_estimators=200, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, bagging_freq=1,
    random_state=42, n_jobs=-1, verbose=-1,
)

# XGBoost and CatBoost are only used *inside* the stacked ensemble member.
_STACKED_XGB = dict(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    objective="multi:softprob", eval_metric="mlogloss",
    n_jobs=-1, random_state=42, verbosity=0,
)

_STACKED_CB = dict(
    iterations=200, learning_rate=0.05, depth=6,
    random_state=42, verbose=0, thread_count=-1,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class OOFResult:
    """Out-of-fold predictions for a single model.

    Attributes
    ----------
    probabilities : ndarray of shape (n_samples, n_classes)
        Row-aligned with the training DataFrame. Each row sums to 1.
    classes : ndarray
        The class labels corresponding to the columns of ``probabilities``.
    label : str
        Human-readable name for this model (e.g. ``"baseline"``).
    fold_metrics : list of dict
        Per-fold metric dicts from :func:`compute_full_metrics`.
    """
    probabilities: np.ndarray
    classes: np.ndarray
    label: str
    fold_metrics: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# XGBoost >= 2.0 label-shift helper
# ---------------------------------------------------------------------------
def _is_xgboost(estimator) -> bool:
    """True if *estimator* is an XGBoost model.

    XGBoost >= 2.0 rejects class labels that don't start at 0, so ESI
    1-5 must be shifted to 0-4 before ``fit()`` and reversed on output.
    """
    return "XGB" in type(estimator).__name__


def _fit_one(
    estimator,
    X_tr: np.ndarray, y_tr: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> object:
    """Fit a single estimator, handling XGBoost label shift."""
    if _is_xgboost(estimator):
        y_tr = y_tr - int(y_tr.min())
    if sample_weight is not None:
        return estimator.fit(X_tr, y_tr, sample_weight=sample_weight)
    return estimator.fit(X_tr, y_tr)


# ---------------------------------------------------------------------------
# OOF collection -- standard (LightGBM variants)
# ---------------------------------------------------------------------------
def collect_oof(
    make_estimator: Callable,
    df,
    features,
    y: np.ndarray,
    classes: np.ndarray,
    embeddings: np.ndarray | None = None,
    n_folds: int = 5,
    sample_weight: np.ndarray | None = None,
    label: str = "model",
    n_pca: int = 50,
    seed: int = 42,
    early_stopping: bool = False,
    patience: int = 100,
) -> OOFResult:
    """Train one model across K stratified folds; return OOF probabilities.

    Parameters
    ----------
    make_estimator : callable
        Zero-arg factory returning a fresh scikit-learn-compatible
        classifier instance.  Called once per fold.
    df : DataFrame
        Feature table (already engineered).
    features : FeatureLists
        Numeric / categorical column split.
    y : ndarray
        Integer-encoded labels (ESI 1-5).
    classes : ndarray
        Sorted unique class labels.
    embeddings : ndarray, optional
        Pre-computed text embeddings, row-aligned with ``df``.
        PCA-reduced inside each fold to avoid leakage.
    sample_weight : ndarray, optional
        Per-sample weights (e.g. inverse-frequency class weights).
    early_stopping : bool
        If True, hold out 10 % of training data per fold for
        early-stopping evaluation (LightGBM callback).
    patience : int
        Early-stopping rounds (only used when ``early_stopping=True``).
    """
    import warnings

    from sklearn.model_selection import StratifiedKFold

    from .data_processing import preprocess_fold

    n, K = len(y), len(classes)
    oof = np.zeros((n, K))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics: list[dict] = []

    for fold, (tr, va) in enumerate(skf.split(np.zeros(n), y), start=1):
        X_tr, X_va = preprocess_fold(
            df, features, tr, va, embeddings=embeddings, n_pca=n_pca,
        )
        model = make_estimator()
        sw = sample_weight[tr] if sample_weight is not None else None

        if early_stopping:
            # Hold out 10 % of training rows for the early-stopping set.
            from sklearn.model_selection import train_test_split as _tts
            import lightgbm as _lgb

            idx = np.arange(len(y[tr]))
            it, ie = _tts(idx, test_size=0.1, stratify=y[tr], random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    X_tr[it], y[tr][it],
                    sample_weight=sw[it] if sw is not None else None,
                    eval_set=[(X_tr[ie], y[tr][ie])],
                    callbacks=[
                        _lgb.early_stopping(patience, verbose=False),
                        _lgb.log_evaluation(0),
                    ],
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _fit_one(model, X_tr, y[tr], sw)

        proba = model.predict_proba(X_va)

        # Re-order probability columns to match ``classes``.
        # XGBoost trained on shifted labels uses 0-indexed classes_; shift back.
        shift = int(y.min()) if _is_xgboost(model) else 0
        model_labels = [int(c) + shift for c in model.classes_]
        col_idx = [model_labels.index(int(c)) for c in classes]
        oof[va] = proba[:, col_idx]

        y_pred = classes[np.argmax(oof[va], axis=1)]
        fold_metrics.append(
            compute_full_metrics(y[va], y_pred, oof[va], classes)
        )

    return OOFResult(
        probabilities=oof, classes=classes,
        label=label, fold_metrics=fold_metrics,
    )


# ---------------------------------------------------------------------------
# OOF collection -- stacked ensemble (LGB + XGB + CatBoost -> LogReg)
# ---------------------------------------------------------------------------
def collect_oof_stacked(
    df,
    features,
    y: np.ndarray,
    classes: np.ndarray,
    embeddings: np.ndarray | None = None,
    n_outer: int = 5,
    n_inner: int = 3,
    n_pca: int = 50,
) -> OOFResult:
    """Nested-CV stacked ensemble: LGB + XGB + CatBoost -> LogisticRegression.

    Outer loop (``n_outer`` folds): for each held-out validation set,
    train three base learners on inner folds and generate meta-features
    (class probabilities concatenated across learners).  A logistic
    regression meta-learner is then fit on the inner-OOF meta-features
    and predicts on the outer validation set.

    The test-set meta-features for each inner fold are *averaged* over
    all inner splits, avoiding leakage.

    Parameters
    ----------
    n_outer : int
        Number of outer CV folds (default 5).
    n_inner : int
        Number of inner CV folds for meta-feature generation (default 3).
    """
    import warnings

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    from .data_processing import preprocess_fold

    # Late imports to keep the module importable without these installed.
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    n, K = len(y), len(classes)
    min_cls = int(min(classes))
    oof_prob = np.zeros((n, K))
    fold_metrics: list[dict] = []

    outer_kf = StratifiedKFold(
        n_splits=n_outer, shuffle=True, random_state=42,
    )

    for fold, (tr_o, va_o) in enumerate(outer_kf.split(np.zeros(n), y)):
        X_tr_o, X_va_o = preprocess_fold(
            df, features, tr_o, va_o, embeddings=embeddings, n_pca=n_pca,
        )
        y_tr_o, y_va_o = y[tr_o], y[va_o]

        inner_kf = StratifiedKFold(
            n_splits=n_inner, shuffle=True, random_state=fold,
        )
        meta_tr = np.zeros((len(y_tr_o), K * 3))
        meta_te = np.zeros((len(y_va_o), K * 3))

        for _, (ti, vi) in enumerate(inner_kf.split(X_tr_o, y_tr_o)):
            X_ti, X_vi = X_tr_o[ti], X_tr_o[vi]
            y_ti = y_tr_o[ti]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                lgb = LGBMClassifier(**DEFAULT_LGBM)
                lgb.fit(X_ti, y_ti)

                xgb = XGBClassifier(**_STACKED_XGB)
                xgb.fit(X_ti, y_ti - min_cls)

                cb = CatBoostClassifier(**_STACKED_CB)
                cb.fit(X_ti, y_ti)

            # Inner-OOF: fill validation rows of meta_tr.
            meta_tr[vi, :K] = lgb.predict_proba(X_vi)
            meta_tr[vi, K:2 * K] = xgb.predict_proba(X_vi)
            meta_tr[vi, 2 * K:3 * K] = np.array(cb.predict_proba(X_vi))

            # Test-set: accumulate (averaged after the inner loop).
            meta_te[:, :K] += lgb.predict_proba(X_va_o)
            meta_te[:, K:2 * K] += xgb.predict_proba(X_va_o)
            meta_te[:, 2 * K:3 * K] += np.array(cb.predict_proba(X_va_o))

        meta_te /= n_inner

        lr = LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, solver="lbfgs",
        )
        lr.fit(meta_tr, y_tr_o)

        prob = lr.predict_proba(meta_te)
        oof_prob[va_o] = prob

        y_pred = classes[np.argmax(prob, axis=1)]
        fold_metrics.append(
            compute_full_metrics(y_va_o, y_pred, prob, classes)
        )

    return OOFResult(
        probabilities=oof_prob, classes=classes,
        label="stacked", fold_metrics=fold_metrics,
    )


# ---------------------------------------------------------------------------
# Estimator factories
# ---------------------------------------------------------------------------
def make_lgbm(params: dict | None = None):
    """Create a LightGBM classifier with the given (or default) params."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**(params or DEFAULT_LGBM))


# ---------------------------------------------------------------------------
# Blend search (softmax reparameterisation + Nelder-Mead)
# ---------------------------------------------------------------------------
def blend_search(
    oof_dict: dict[str, np.ndarray],
    y: np.ndarray,
    classes: np.ndarray,
    n_restarts: int = 10,
) -> tuple[np.ndarray, dict]:
    """Find convex blend weights that maximise macro F1 over OOF matrices.

    Uses softmax reparameterisation so the raw parameters are
    unconstrained while the effective weights always sum to 1.
    Nelder-Mead is run from ``n_restarts`` random starting points.

    For 2-model blends, a simple grid search over alpha in [0, 1] is
    used instead.

    Returns
    -------
    blended : ndarray of shape (n_samples, n_classes)
        The optimally blended probability matrix.
    info : dict
        Weight mapping and best macro F1.
    """
    from sklearn.metrics import f1_score

    names = list(oof_dict.keys())
    probs = [oof_dict[n] for n in names]

    # Special-case: two models -> simple grid search.
    if len(probs) == 2:
        best_f1, best_a = -1.0, 0.5
        for a in np.arange(0, 1.01, 0.02):
            bl = a * probs[0] + (1 - a) * probs[1]
            yp = classes[np.argmax(bl, axis=1)]
            f1 = f1_score(y, yp, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_a = f1, a
        bl = best_a * probs[0] + (1 - best_a) * probs[1]
        return bl, {
            "weights": {names[0]: round(best_a, 2),
                        names[1]: round(1 - best_a, 2)},
            "macro_f1": round(best_f1, 4),
        }

    # General case: softmax reparameterisation + Nelder-Mead.
    from scipy.optimize import minimize
    from scipy.special import softmax

    def neg_f1(raw_w: np.ndarray) -> float:
        w = softmax(raw_w)
        bl = sum(wi * p for wi, p in zip(w, probs))
        yp = classes[np.argmax(bl, axis=1)]
        return -f1_score(y, yp, average="macro", zero_division=0)

    best_f1, best_w = -1.0, None
    rng = np.random.RandomState(42)
    for _ in range(n_restarts):
        x0 = rng.randn(len(probs)) * 0.5
        res = minimize(
            neg_f1, x0, method="Nelder-Mead",
            options={"maxiter": 1000},
        )
        if -res.fun > best_f1:
            best_f1 = -res.fun
            best_w = softmax(res.x)

    blended = sum(wi * p for wi, p in zip(best_w, probs))
    return blended, {
        "weights": {n: round(float(w), 4) for n, w in zip(names, best_w)},
        "macro_f1": round(best_f1, 4),
    }
