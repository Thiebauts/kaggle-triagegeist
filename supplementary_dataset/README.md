# Triagegeist MIMIC-IV-ED Supplementary Results

Pre-computed aggregate results and figures from MIMIC-IV-ED analysis
(418 100 real emergency-department visits). Used by the competition
notebook to display real-data validation results.

**No patient-level data is included.** All files contain aggregate
statistics and model performance metrics only.

MIMIC-IV-ED access requires the PhysioNet Credentialed Data Use
Agreement: <https://physionet.org/content/mimic-iv-ed/>

## Contents

- `figures/` — 15 publication-quality PNGs (300 dpi, rendered from the
  vector PDFs that ship with the LaTeX report): age-stratified
  under-triage and vitals comparison, bias-audit disparities, class
  distribution (Kaggle vs MIMIC), clinical operating curve, confusion
  matrix (combined), keyword-leakage heatmap, calibrated vs
  uncalibrated operating curve, ordinal-penalty comparison, Pareto
  frontier (training weights vs threshold), reliability benchmark
  forest plot, and SHAP variants (bar combined, bar global, beeswarm,
  per-class waterfall plots for ESI-2 and ESI-3).
- `results/` — 13 JSON and CSV files with aggregate metrics:
  `pipeline_results`, `ordinal_penalty_results`, `acscot_compliance`,
  `clinical_operating_curve`, `calibration_results`,
  `asymmetric_weight_sweep` (+ summary), `age_stratified_undertriage`,
  `age_stratified_vitals_table`, `feature_engineering_results`,
  `bias_audit_summary`, `bias_audit_top20_undertriage`,
  `bias_audit_single_axis`.

## Citation

Xie F, Salim H, Bidgoli A, et al. *MIMIC-IV-ED: a freely accessible
emergency department database.* Scientific Data 9:658, 2022.
doi:10.1038/s41597-022-01782-9.
