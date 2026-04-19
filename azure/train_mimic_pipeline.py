"""End-to-end Triagegeist training pipeline for Azure ML.

Runs the full MIMIC-IV-ED benchmark: feature engineering, 5-fold
cross-validation of a 5-model blend (4 LightGBM variants + 1 stacked
LGB/XGB/CB -> LogReg), softmax-weighted blend search, threshold
optimisation, and aggregate result logging.

The ClinicalBERT fine-tuning and embedding extraction steps are exposed
as ``--phase`` selectors so they can be scheduled on a GPU cluster
separately from the CPU-bound pipeline.

Before submitting the job, replace the workspace placeholders with your
own identifiers or (preferred) read them from the environment:

    export AZURE_SUBSCRIPTION_ID=...
    export AZURE_RESOURCE_GROUP=...
    export AZURE_WORKSPACE_NAME=...

No credentials are hard-coded in this file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Workspace identifiers are consumed by ``run_azure_job`` below. Leave
# these as placeholders in the source tree; callers must set the
# corresponding environment variables before submitting a job.
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "your-subscription-id")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "your-resource-group")
WORKSPACE_NAME = os.environ.get("AZURE_WORKSPACE_NAME", "your-workspace-name")


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def load_mimic_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the MIMIC-IV-ED + MIMIC-IV core tables from a mounted asset."""
    ed = data_dir / "mimic-iv-ed" / "2.2" / "ed"
    core = data_dir / "mimic-iv"
    return {
        "triage": pd.read_csv(ed / "triage.csv.gz"),
        "edstays": pd.read_csv(ed / "edstays.csv.gz", parse_dates=["intime"]),
        "medrecon": pd.read_csv(ed / "medrecon.csv.gz"),
        "patients": pd.read_csv(core / "patients.csv.gz"),
        "admissions": pd.read_csv(core / "admissions.csv.gz",
                                   parse_dates=["admittime"]),
        "diag_icd": pd.read_csv(core / "diagnoses_icd.csv.gz",
                                 dtype={"icd_code": str}),
        "drgcodes": pd.read_csv(core / "drgcodes.csv.gz"),
    }


# ------------------------------------------------------------------ #
# Phases
# ------------------------------------------------------------------ #

def run_blend_phase(
    data_dir: Path, output_dir: Path, n_folds: int, emb_path: Path | None,
) -> None:
    """Feature engineering + 5-model blend + threshold optimisation.

    The five blend members are:

    1. baseline -- standard LightGBM (200 trees, lr=0.05)
    2. weighted -- LightGBM with inverse-frequency class weights
    3. more_trees -- LightGBM (800 trees, lr=0.02)
    4. early_stop -- LightGBM (2000 trees, lr=0.01, patience=100)
    5. stacked -- LGB + XGB + CatBoost -> LogisticRegression (nested CV)
    """
    from triagegeist.data_processing import build_mimic_features
    from triagegeist.model_training import (
        collect_oof, collect_oof_stacked, make_lgbm,
        blend_search, DEFAULT_LGBM,
    )
    from triagegeist.threshold_optimization import (
        optimize_thresholds, clinical_operating_curve,
    )
    from triagegeist.utils import compute_full_metrics

    t0 = time.time()
    tables = load_mimic_tables(data_dir)
    df, features = build_mimic_features(
        triage=tables["triage"], edstays=tables["edstays"],
        patients=tables["patients"], medrecon=tables["medrecon"],
        admissions=tables["admissions"],
        diag_icd=tables["diag_icd"], drgcodes=tables["drgcodes"],
    )
    y = df["acuity"].to_numpy()
    classes = np.array([1, 2, 3, 4, 5])

    embeddings = np.load(emb_path) if emb_path and emb_path.exists() else None

    # Inverse-frequency class weights for the "weighted" member.
    cls, cnts = np.unique(y, return_counts=True)
    wmap = dict(zip(cls, len(y) / (len(cls) * cnts)))
    bal_w = np.array([wmap[c] for c in y])

    oof = {}

    # [1/5] Baseline LightGBM
    r = collect_oof(
        lambda: make_lgbm(DEFAULT_LGBM),
        df, features, y, classes, embeddings=embeddings,
        n_folds=n_folds, label="baseline",
    )
    oof["baseline"] = r.probabilities

    # [2/5] LightGBM with class weights
    r = collect_oof(
        lambda: make_lgbm(DEFAULT_LGBM),
        df, features, y, classes, embeddings=embeddings,
        n_folds=n_folds, sample_weight=bal_w, label="weighted",
    )
    oof["weighted"] = r.probabilities

    # [3/5] More trees (800 @ lr=0.02)
    mt_params = {**DEFAULT_LGBM, "n_estimators": 800, "learning_rate": 0.02}
    r = collect_oof(
        lambda: make_lgbm(mt_params),
        df, features, y, classes, embeddings=embeddings,
        n_folds=n_folds, label="more_trees",
    )
    oof["more_trees"] = r.probabilities

    # [4/5] Early stopping (2000 @ lr=0.01, patience=100)
    es_params = {**DEFAULT_LGBM, "n_estimators": 2000, "learning_rate": 0.01}
    r = collect_oof(
        lambda: make_lgbm(es_params),
        df, features, y, classes, embeddings=embeddings,
        n_folds=n_folds, label="early_stop",
        early_stopping=True, patience=100,
    )
    oof["early_stop"] = r.probabilities

    # [5/5] Stacked (LGB + XGB + CB -> LogReg, 5 outer x 3 inner)
    r = collect_oof_stacked(
        df, features, y, classes, embeddings=embeddings,
        n_outer=n_folds, n_inner=3,
    )
    oof["stacked"] = r.probabilities

    # Blend search (softmax + Nelder-Mead)
    blended, blend_info = blend_search(oof, y, classes)
    preds = classes[np.argmax(blended, axis=1)]
    blend_metrics = compute_full_metrics(y, preds, blended, classes)

    thresholds, thresh_metrics = optimize_thresholds(
        blended, y, classes, objective="macro_f1",
    )
    curve = clinical_operating_curve(blended, y, classes)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blend_weights.json").write_text(
        json.dumps(blend_info, indent=2)
    )
    (output_dir / "pipeline_results.json").write_text(
        json.dumps({
            "blend": {k: v for k, v in blend_metrics.items()
                      if k != "confusion_matrix"},
            "blend_with_thresholds": {k: v for k, v in thresh_metrics.items()
                                       if k != "confusion_matrix"},
            "n_samples": int(len(y)),
            "n_folds": n_folds,
            "wall_time_sec": round(time.time() - t0, 1),
        }, indent=2, default=float)
    )
    curve.to_csv(output_dir / "clinical_operating_curve.csv", index=False)
    print(f"Blend phase complete in {time.time() - t0:.1f}s")


def run_finetune_phase(
    data_dir: Path, output_dir: Path, epochs: int, batch_size: int,
) -> None:
    """Fine-tune Bio_ClinicalBERT on the ESI classification task.

    Writes the fine-tuned checkpoint to ``output_dir``. Run on a GPU
    compute target.
    """
    from triagegeist.embedding_extraction import (
        finetune_clinicalbert, FineTuneConfig,
    )

    tables = load_mimic_tables(data_dir)
    df = tables["triage"].merge(
        tables["edstays"][["stay_id", "subject_id"]],
        on=["stay_id", "subject_id"], how="left",
    )
    df = df.dropna(subset=["acuity"]).copy()
    texts = df["chiefcomplaint"].fillna("unknown").astype(str).tolist()
    labels = df["acuity"].astype(int).to_numpy()

    cfg = FineTuneConfig(num_epochs=epochs, batch_size=batch_size)
    ckpt = finetune_clinicalbert(
        texts=texts, labels=labels, config=cfg,
        output_dir=str(output_dir / "clinicalbert-finetuned"),
    )
    print(f"Fine-tuned checkpoint: {ckpt}")


def run_embed_phase(
    data_dir: Path, output_dir: Path, model_dir: Path,
) -> None:
    """Extract 768-d embeddings from a fine-tuned ClinicalBERT checkpoint."""
    from triagegeist.embedding_extraction import encode_clinicalbert

    tables = load_mimic_tables(data_dir)
    df = tables["triage"].merge(
        tables["edstays"][["stay_id", "subject_id"]],
        on=["stay_id", "subject_id"], how="left",
    )
    df = df.dropna(subset=["acuity"]).copy()
    texts = df["chiefcomplaint"].fillna("unknown").astype(str).tolist()

    emb = encode_clinicalbert(texts, model_id=str(model_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "clinicalbert_embeddings.npy", emb)
    print(f"Saved embeddings: shape={emb.shape}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["blend", "finetune", "embed"],
                        default="blend")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Mount point for the mimic-iv-ed data asset")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where aggregate results are written")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--embeddings", type=Path, default=None,
                        help="Path to pre-computed embeddings .npy")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Path to fine-tuned ClinicalBERT checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    if args.phase == "blend":
        run_blend_phase(args.data_dir, args.output_dir, args.n_folds,
                         args.embeddings)
    elif args.phase == "finetune":
        run_finetune_phase(args.data_dir, args.output_dir,
                            args.epochs, args.batch_size)
    else:
        if args.model_dir is None:
            parser.error("--model-dir is required for --phase embed")
        run_embed_phase(args.data_dir, args.output_dir, args.model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
