"""Triagegeist — clinical safety-aware emergency triage prediction.

Public entry points:

* :mod:`.data_processing` — feature engineering for Kaggle and MIMIC-IV-ED
* :mod:`.clinical_scores` — NEWS2, qSOFA, shock indices, age-adjusted flags
* :mod:`.model_training` — 5-model blend (4 LightGBM + 1 stacked) with
  softmax-weighted Nelder-Mead blend search
* :mod:`.threshold_optimization` — Nelder-Mead thresholds, asymmetric
  weight sweeps, clinical operating curves, Pareto frontier
* :mod:`.bias_audit` — intersectional subgroup fairness analysis
* :mod:`.shap_analysis` — TreeSHAP explanations + per-class attribution
* :mod:`.embedding_extraction` — Bio_ClinicalBERT fine-tuning + MiniLM
  baseline
* :mod:`.utils` — plotting style, metric helpers, ESI colour palette
"""
__version__ = "1.0.0"
