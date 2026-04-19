"""Feature engineering for the Triagegeist pipeline.

Two public entry points:

* :func:`build_kaggle_features` -- vitals, demographics, complaint system,
  patient history; for the Kaggle synthetic dataset.
* :func:`build_mimic_features` -- vitals, age, medication categories from
  medrecon (ETC ontology), Charlson comorbidities from diagnoses_icd,
  DRG severity history, prior-visit counts, and clinical composite
  scores; for MIMIC-IV-ED.

Both pipelines are leakage-safe: no post-triage measurements
(disposition, length of stay) are ever used as features, and prior
cumulative statistics always shift by one visit so the current
admission is excluded.

Feature counts (MIMIC blend pipeline):
  58 numeric + 7 categorical + 50 PCA = 115 total
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

VITAL_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

# Medication categories matched against the MIMIC-IV-ED medrecon
# `etcdescription` field (ETC drug-class ontology). Each visit gets a
# binary flag per category indicating pre-admission use.
MED_FLAGS: dict[str, list[str]] = {
    "med_hypertension": [
        "beta blocker", "ace inhibitor", "angiotensin",
        "calcium channel blocker",
    ],
    "med_diabetes": [
        "insulin", "antidiabetic", "sulfonylurea", "glp-1", "dpp-4",
    ],
    "med_heart_failure": ["loop diuretic", "cardiac glycoside"],
    "med_copd_asthma": [
        "beta 2-adrenergic", "copd therapy", "asthma", "bronchodilator",
    ],
    "med_depression": ["ssri", "antidepressant - selective", "snri"],
    "med_anxiety": ["benzodiazepine", "antianxiety"],
    "med_epilepsy": ["anticonvulsant"],
    "med_thyroid": ["thyroid hormone"],
    "med_anticoagulant": [
        "anticoagulant", "platelet aggregation inhibitor",
    ],
    "med_hyperlipid": ["hmg coa", "antihyperlipidemic"],
    "med_opioid": ["analgesic opioid"],
    "med_antipsychotic": ["antipsychotic"],
    "med_renal": ["phosphate binder"],
}

# Charlson comorbidity index -- ICD-9 / ICD-10 prefix mapping.
# Original 1987 weights (Charlson et al. J Chronic Dis 1987;40:373-383).
CHARLSON_CONDITIONS: dict[str, dict[str, list[str]]] = {
    "cc_mi": {
        "9": ["410", "412"],
        "10": ["I21", "I22", "I252"],
    },
    "cc_chf": {
        "9": ["428"],
        "10": ["I50"],
    },
    "cc_pvd": {
        "9": ["440", "441", "443", "444", "447"],
        "10": ["I70", "I71", "I731", "I738", "I739",
               "I771", "I790", "K551"],
    },
    "cc_stroke": {
        "9": ["430", "431", "432", "433", "434", "436", "437"],
        "10": ["G45", "G46", "I60", "I61", "I62", "I63",
               "I64", "I65", "I66", "I67", "I68", "I69"],
    },
    "cc_dementia": {
        "9": ["290"],
        "10": ["F00", "F01", "F02", "F03", "G30"],
    },
    "cc_copd": {
        "9": ["490", "491", "492", "493", "494", "495", "496"],
        "10": ["J40", "J41", "J42", "J43", "J44",
               "J45", "J46", "J47"],
    },
    "cc_rheumatic": {
        "9": ["710", "714", "725"],
        "10": ["M05", "M06", "M32", "M33", "M34", "M351"],
    },
    "cc_peptic": {
        "9": ["531", "532", "533", "534"],
        "10": ["K25", "K26", "K27", "K28"],
    },
    "cc_liver_mild": {
        "9": ["571"],
        "10": ["B18", "K73", "K74"],
    },
    "cc_diabetes": {
        "9": ["250"],
        "10": ["E10", "E11", "E12", "E13", "E14"],
    },
    "cc_hemiplegia": {
        "9": ["342", "343", "344"],
        "10": ["G81", "G82", "G83"],
    },
    "cc_renal": {
        "9": ["582", "583", "585", "586", "588"],
        "10": ["N03", "N05", "N18", "N19", "N250"],
    },
    "cc_cancer": {
        "9": ["14", "15", "16", "17", "18", "19"],
        "10": ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7",
               "C81", "C82", "C83", "C84", "C85", "C88", "C9"],
    },
    "cc_liver_sev": {
        "9": ["572", "456"],
        "10": ["I850", "K704", "K721", "K765", "K766", "K767"],
    },
    "cc_metastatic": {
        "9": ["196", "197", "198", "199"],
        "10": ["C77", "C78", "C79", "C80"],
    },
    "cc_aids": {
        "9": ["042"],
        "10": ["B20", "B21", "B22", "B24"],
    },
}

CHARLSON_WEIGHTS: dict[str, int] = {
    "cc_mi": 1, "cc_chf": 1, "cc_pvd": 1, "cc_stroke": 1,
    "cc_dementia": 1, "cc_copd": 1, "cc_rheumatic": 1, "cc_peptic": 1,
    "cc_liver_mild": 1, "cc_diabetes": 1,
    "cc_hemiplegia": 2, "cc_renal": 2, "cc_cancer": 2,
    "cc_liver_sev": 3,
    "cc_metastatic": 6, "cc_aids": 6,
}


@dataclass(frozen=True)
class FeatureLists:
    """Container for numeric and categorical feature names after
    engineering.  Pass to downstream pipelines so the same columns are
    consistently fed to imputers and encoders."""
    numeric: list[str]
    categorical: list[str]


# ------------------------------------------------------------------ #
# Kaggle (synthetic)
# ------------------------------------------------------------------ #

def build_kaggle_features(
    train: pd.DataFrame,
    complaints: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, FeatureLists]:
    """Engineer features from the Kaggle competition tables.

    Parameters
    ----------
    train : DataFrame
        The base ``train.csv`` (or ``test.csv``) table.
    complaints : DataFrame, optional
        ``chief_complaints.csv``.  If provided, the
        ``chief_complaint_system`` column is joined in.
        ``chief_complaint_raw`` is **not** used because of synthetic
        label leakage (severity keywords encode ESI).
    history : DataFrame, optional
        ``patient_history.csv`` containing ``hx_*`` comorbidity flags.

    Returns
    -------
    (features, FeatureLists)
    """
    df = train.copy()
    if complaints is not None:
        df = df.merge(
            complaints[["patient_id", "chief_complaint_system"]],
            on="patient_id", how="left",
        )
    if history is not None:
        df = df.merge(history, on="patient_id", how="left")

    if "pain_score" in df.columns:
        df["pain_score"] = df["pain_score"].replace(-1, np.nan).clip(0, 10)

    for c in VITAL_COLS:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isna().astype(int)

    # Derived vitals.
    if "heartrate" in df.columns and "sbp" in df.columns:
        df["shock_index"] = df["heartrate"] / (df["sbp"] + 1e-8)
        df["pulse_pressure"] = df["sbp"] - df["dbp"]
        df["map"] = (df["sbp"] + 2 * df["dbp"]) / 3

    if "arrival_hour" in df.columns:
        df["arrival_hour_sin"] = np.sin(2 * np.pi * df["arrival_hour"] / 24)
        df["arrival_hour_cos"] = np.cos(2 * np.pi * df["arrival_hour"] / 24)

    hx_cols = [c for c in df.columns if c.startswith("hx_")]
    if hx_cols:
        df["num_comorbidities"] = df[hx_cols].sum(axis=1)

    numeric = VITAL_COLS + [f"{c}_missing" for c in VITAL_COLS] + [
        "shock_index", "pulse_pressure", "map",
        "arrival_hour_sin", "arrival_hour_cos",
        "pain_score", "age", "num_comorbidities",
    ] + hx_cols
    numeric = [c for c in numeric if c in df.columns]

    categorical = [
        c for c in ["sex", "arrival_mode", "chief_complaint_system"]
        if c in df.columns
    ]
    return df, FeatureLists(numeric=numeric, categorical=categorical)


# ------------------------------------------------------------------ #
# MIMIC-IV-ED helpers
# ------------------------------------------------------------------ #

def _parse_pain(val) -> float:
    """Extract the numeric pain score from MIMIC's free-text field."""
    if pd.isna(val):
        return np.nan
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", str(val).strip().lower())
    if m:
        v = float(m.group(1))
        return v if v <= 10 else np.nan
    return np.nan


def _add_medrecon_flags(
    df: pd.DataFrame, medrecon: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate medrecon per stay into medication-category flags.

    Uses the ``etcdescription`` column (ETC drug-class ontology), not
    the brand/generic ``name`` column, matching the actual Azure
    training pipeline.  medrecon records pre-admission medications,
    so derived flags are leakage-safe.
    """
    med_count = (
        medrecon.groupby("stay_id").size().reset_index(name="num_medications")
    )
    df = df.merge(med_count, on="stay_id", how="left")
    df["num_medications"] = df["num_medications"].fillna(0).astype(int)

    med_etc = medrecon[["stay_id", "etcdescription"]].copy()
    med_etc["etc_lower"] = med_etc["etcdescription"].fillna("").str.lower()

    for flag, keywords in MED_FLAGS.items():
        pattern = "|".join(keywords)
        flag_stays = med_etc.loc[
            med_etc["etc_lower"].str.contains(pattern, regex=True),
            "stay_id",
        ].unique()
        df[flag] = df["stay_id"].isin(flag_stays).astype(int)

    return df


def _add_charlson(
    df: pd.DataFrame,
    admissions: pd.DataFrame,
    diag_icd: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Compute per-admission Charlson flags and prior cumulative score.

    Uses the cumsum-minus-current pattern: for each admission the
    Charlson flags reflect diagnoses from *prior* admissions only,
    preventing leakage from the current encounter.

    Returns (df, cc_cols, prior_cc_cols).
    """
    diag_icd = diag_icd.copy()
    diag_icd["icd_str"] = diag_icd["icd_code"].astype(str).str.strip()
    icd9 = diag_icd[diag_icd["icd_version"] == 9]
    icd10 = diag_icd[diag_icd["icd_version"] == 10]

    rows: list[pd.DataFrame] = []
    for cond, ver_map in CHARLSON_CONDITIONS.items():
        for ver_str, prefixes in ver_map.items():
            src = icd9 if ver_str == "9" else icd10
            pat = "^(" + "|".join(re.escape(p) for p in prefixes) + ")"
            matched = src.loc[
                src["icd_str"].str.match(pat), ["hadm_id"]
            ].copy()
            matched["condition"] = cond
            rows.append(matched)

    hadm_cond = pd.concat(rows, ignore_index=True).drop_duplicates()
    cc_pivot = (
        pd.crosstab(hadm_cond["hadm_id"], hadm_cond["condition"])
        .clip(upper=1)
    )
    cc_cols = list(CHARLSON_CONDITIONS.keys())
    for cond in cc_cols:
        if cond not in cc_pivot.columns:
            cc_pivot[cond] = 0
    cc_pivot = cc_pivot[cc_cols].reset_index()

    prior_cc_cols = [f"prior_{c}" for c in cc_cols]

    adm_cc = admissions[["subject_id", "hadm_id", "admittime"]].merge(
        cc_pivot, on="hadm_id", how="left",
    )
    adm_cc[cc_cols] = adm_cc[cc_cols].fillna(0)
    adm_cc = adm_cc.sort_values(
        ["subject_id", "admittime"],
    ).reset_index(drop=True)

    # cumsum-minus-current: each row sees only diagnoses from earlier
    # admissions.
    cumsum_df = adm_cc.groupby("subject_id")[cc_cols].cumsum()
    prior_df = (cumsum_df - adm_cc[cc_cols].values).clip(lower=0, upper=1)
    adm_cc[prior_cc_cols] = prior_df.values

    adm_cc["charlson_score"] = sum(
        adm_cc[f"prior_{c}"] * w for c, w in CHARLSON_WEIGHTS.items()
    )

    df = df.merge(
        adm_cc[["hadm_id"] + prior_cc_cols + ["charlson_score"]],
        on="hadm_id", how="left",
    )
    for col in prior_cc_cols + ["charlson_score"]:
        df[col] = df[col].fillna(0)

    return df, cc_cols, prior_cc_cols


def _add_drg_severity(
    df: pd.DataFrame,
    admissions: pd.DataFrame,
    drgcodes: pd.DataFrame,
) -> pd.DataFrame:
    """Add prior DRG severity and mortality using cummax + shift(1).

    Each row reflects the worst DRG severity/mortality from
    *previous* admissions only (shift by 1 after cummax).
    """
    drg_agg = (
        drgcodes.groupby("hadm_id")
        .agg(
            drg_sev_max=("drg_severity", "max"),
            drg_mort_max=("drg_mortality", "max"),
        )
        .reset_index()
    )
    adm_drg = admissions[["subject_id", "hadm_id", "admittime"]].merge(
        drg_agg, on="hadm_id", how="left",
    )
    adm_drg[["drg_sev_max", "drg_mort_max"]] = (
        adm_drg[["drg_sev_max", "drg_mort_max"]].fillna(0)
    )
    adm_drg = adm_drg.sort_values(
        ["subject_id", "admittime"],
    ).reset_index(drop=True)

    adm_drg["cum_sev"] = adm_drg.groupby("subject_id")[
        "drg_sev_max"
    ].cummax()
    adm_drg["cum_sev"] = (
        adm_drg.groupby("subject_id")["cum_sev"].shift(1).fillna(0)
    )
    adm_drg["cum_mort"] = adm_drg.groupby("subject_id")[
        "drg_mort_max"
    ].cummax()
    adm_drg["cum_mort"] = (
        adm_drg.groupby("subject_id")["cum_mort"].shift(1).fillna(0)
    )

    df = df.merge(
        adm_drg[["hadm_id", "cum_sev", "cum_mort"]].rename(
            columns={
                "cum_sev": "prior_drg_severity",
                "cum_mort": "prior_drg_mortality",
            },
        ),
        on="hadm_id", how="left",
    )
    df["prior_drg_severity"] = df["prior_drg_severity"].fillna(0)
    df["prior_drg_mortality"] = df["prior_drg_mortality"].fillna(0)
    return df


def _add_clinical_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add NEWS2, qSOFA, age-shock index, n_abnormal_vitals.

    Only aggregate scores are retained as features -- individual
    NEWS2 components, qSOFA components, and binary vital-sign flags
    were dropped because they showed near-zero LightGBM importance
    in ablation experiments.
    """

    # NEWS2 partial (5 components, no consciousness / supplemental O2).
    def _n2_rr(v: float) -> float:
        if pd.isna(v):
            return np.nan
        if v <= 8:
            return 3
        if v <= 11:
            return 1
        if v <= 20:
            return 0
        if v <= 24:
            return 2
        return 3

    def _n2_o2(v: float) -> float:
        if pd.isna(v):
            return np.nan
        if v <= 91:
            return 3
        if v <= 93:
            return 2
        if v <= 95:
            return 1
        return 0

    def _n2_sbp(v: float) -> float:
        if pd.isna(v):
            return np.nan
        if v <= 90:
            return 3
        if v <= 100:
            return 2
        if v <= 110:
            return 1
        if v <= 219:
            return 0
        return 3

    def _n2_hr(v: float) -> float:
        if pd.isna(v):
            return np.nan
        if v <= 40:
            return 3
        if v <= 50:
            return 1
        if v <= 90:
            return 0
        if v <= 110:
            return 1
        if v <= 130:
            return 2
        return 3

    def _n2_temp(v: float) -> float:
        if pd.isna(v):
            return np.nan
        if v <= 35.0:
            return 3
        if v <= 36.0:
            return 1
        if v <= 38.0:
            return 0
        if v <= 39.0:
            return 1
        return 2

    df["news2_score"] = (
        df["resprate"].apply(_n2_rr)
        + df["o2sat"].apply(_n2_o2)
        + df["sbp"].apply(_n2_sbp)
        + df["heartrate"].apply(_n2_hr)
        + df["temperature"].apply(_n2_temp)
    )

    # Age-Shock Index (HR / SBP * age).
    df["age_shock_index"] = (
        df["heartrate"] / (df["sbp"] + 1e-8)
    ) * df["age"]

    # qSOFA partial (RR >= 22, SBP <= 100; no GCS).
    df["qsofa_score"] = (
        (df["resprate"] >= 22).astype(int)
        + (df["sbp"] <= 100).astype(int)
    )

    # Number of abnormal vitals with age-stratified thresholds.
    is_eld = df["age"] >= 70
    df["n_abnormal_vitals"] = (
        ((~is_eld & (df["heartrate"] > 100))
         | (is_eld & (df["heartrate"] > 90))).astype(int)
        + (df["heartrate"] < 50).astype(int)
        + ((~is_eld & (df["sbp"] < 90))
           | (is_eld & (df["sbp"] < 100))).astype(int)
        + (df["sbp"] >= 180).astype(int)
        + ((~is_eld & (df["o2sat"] < 94))
           | (is_eld & (df["o2sat"] < 92))).astype(int)
        + ((~is_eld & (df["temperature"] > 38.5))
           | (is_eld & (df["temperature"] > 38.0))).astype(int)
        + (df["temperature"] < 35.5).astype(int)
        + ((~is_eld & (df["resprate"] > 22))
           | (is_eld & (df["resprate"] > 20))).astype(int)
        + (df["resprate"] < 10).astype(int)
    )

    return df


# ------------------------------------------------------------------ #
# MIMIC-IV-ED main builder
# ------------------------------------------------------------------ #

def build_mimic_features(
    triage: pd.DataFrame,
    edstays: pd.DataFrame,
    patients: pd.DataFrame,
    medrecon: pd.DataFrame | None = None,
    admissions: pd.DataFrame | None = None,
    diag_icd: pd.DataFrame | None = None,
    drgcodes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, FeatureLists]:
    """Engineer the MIMIC-IV-ED feature set used in the blend.

    Required input tables (all from PhysioNet):

    * ``triage`` -- from ``mimic-iv-ed/ed/triage.csv.gz``
    * ``edstays`` -- from ``mimic-iv-ed/ed/edstays.csv.gz``
    * ``patients`` -- from ``mimic-iv/patients.csv.gz``

    Optional tables enrich the feature set but are not required:

    * ``medrecon`` -- pre-admission medication reconciliation (13 flags
      via ETC ontology ``etcdescription`` column)
    * ``admissions`` -- prior admission counts, insurance, language,
      marital status
    * ``diag_icd`` -- from ``mimic-iv/diagnoses_icd.csv.gz``; used for
      Charlson comorbidity index (16 conditions, original 1987 weights)
    * ``drgcodes`` -- from ``mimic-iv/drgcodes.csv.gz``; used for prior
      DRG severity and mortality (cummax + shift pattern)

    Returns the merged feature DataFrame and the columns split into
    numeric / categorical.  All engineered features are leakage-safe:
    disposition and ED length of stay are never consulted.

    Feature totals (full pipeline): 58 numeric + 7 categorical = 65
    tabular features.  With 50 PCA components from text embeddings
    the total is 115.
    """
    # ── base dataframe ───────────────────────────────────────────────
    df = triage.merge(
        edstays[[
            "stay_id", "subject_id", "hadm_id", "intime",
            "gender", "race", "arrival_transport",
        ]],
        on=["stay_id", "subject_id"], how="left",
    )
    df = df.dropna(subset=["acuity"]).copy()
    df["acuity"] = df["acuity"].astype(int)
    df = df.reset_index(drop=True)
    df["intime"] = pd.to_datetime(df["intime"])

    # ── basic vitals and derived features ────────────────────────────
    df["arrival_hour_sin"] = np.sin(2 * np.pi * df["intime"].dt.hour / 24)
    df["arrival_hour_cos"] = np.cos(2 * np.pi * df["intime"].dt.hour / 24)

    df["pain_numeric"] = df["pain"].apply(_parse_pain)

    for c in VITAL_COLS:
        df[f"{c}_missing"] = df[c].isna().astype(int)

    df["shock_index"] = df["heartrate"] / (df["sbp"] + 1e-8)
    df["pulse_pressure"] = df["sbp"] - df["dbp"]
    df["map"] = (df["sbp"] + 2 * df["dbp"]) / 3
    df["chiefcomplaint"] = (
        df["chiefcomplaint"].fillna("unknown").str.lower().str.strip()
    )

    # ── age ──────────────────────────────────────────────────────────
    df = df.merge(
        patients[["subject_id", "anchor_age", "anchor_year"]],
        on="subject_id", how="left",
    )
    df["age"] = (
        df["anchor_age"] + (df["intime"].dt.year - df["anchor_year"])
    ).clip(0, 120)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 17, 35, 55, 70, 85, 120],
        labels=[
            "pediatric", "young_adult", "adult",
            "middle_aged", "older_adult", "elderly",
        ],
        right=False,
    ).astype(str)

    # ── prior ED visits ──────────────────────────────────────────────
    es = (
        edstays[["subject_id", "stay_id", "intime"]]
        .sort_values(["subject_id", "intime"])
        .copy()
    )
    es["prior_ed_visits"] = es.groupby("subject_id").cumcount()
    df = df.merge(es[["stay_id", "prior_ed_visits"]], on="stay_id", how="left")
    df["prior_ed_visits"] = df["prior_ed_visits"].fillna(0).astype(int)

    # ── medications (ETC ontology) ───────────────────────────────────
    if medrecon is not None:
        df = _add_medrecon_flags(df, medrecon)

    # ── prior admissions + demographics ──────────────────────────────
    if admissions is not None:
        adm = (
            admissions[[
                "subject_id", "hadm_id", "admittime",
                "insurance", "language", "marital_status",
            ]]
            .sort_values(["subject_id", "admittime"])
        )
        adm["prior_admissions"] = adm.groupby("subject_id").cumcount()
        df = df.merge(
            adm[[
                "hadm_id", "prior_admissions",
                "insurance", "language", "marital_status",
            ]],
            on="hadm_id", how="left",
        )
        df["prior_admissions"] = df["prior_admissions"].fillna(0).astype(int)
        for col in ["insurance", "language", "marital_status"]:
            df[col] = df[col].fillna("Unknown")

    # ── DRG severity ─────────────────────────────────────────────────
    if admissions is not None and drgcodes is not None:
        df = _add_drg_severity(df, admissions, drgcodes)

    # ── Charlson comorbidity ─────────────────────────────────────────
    prior_cc_cols: list[str] = []
    cc_cols: list[str] = []
    if admissions is not None and diag_icd is not None:
        df, cc_cols, prior_cc_cols = _add_charlson(df, admissions, diag_icd)

    # ── clinical composite scores ────────────────────────────────────
    df = _add_clinical_scores(df)

    # ── assemble feature lists ───────────────────────────────────────
    # 58 numeric features (matches the Azure blend pipeline exactly).
    numeric: list[str] = (
        # 6 raw vitals
        ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
        # 4 derived vitals
        + ["pain_numeric", "shock_index", "pulse_pressure", "map"]
        # 2 cyclical time
        + ["arrival_hour_sin", "arrival_hour_cos"]
        # 6 missingness indicators
        + [f"{c}_missing" for c in VITAL_COLS]
        # 4 scalars: age, counts, med count, prior admissions
        + ["age", "prior_ed_visits", "num_medications", "prior_admissions"]
        # 2 DRG features
        + ["prior_drg_severity", "prior_drg_mortality"]
        # 1 Charlson aggregate
        + ["charlson_score"]
        # 13 medication flags
        + list(MED_FLAGS.keys())
        # 16 prior Charlson condition flags
        + prior_cc_cols
        # 4 clinical composite scores
        + ["news2_score", "age_shock_index", "qsofa_score",
           "n_abnormal_vitals"]
    )

    # 7 categorical features.
    categorical: list[str] = [
        "gender", "arrival_transport", "race", "age_group",
        "insurance", "language", "marital_status",
    ]

    # Filter to columns actually present (graceful when optional tables
    # are omitted).
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]

    return df, FeatureLists(numeric=numeric, categorical=categorical)


# ------------------------------------------------------------------ #
# Fold-level preprocessing
# ------------------------------------------------------------------ #

def preprocess_fold(
    df: pd.DataFrame,
    features: FeatureLists,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    embeddings: np.ndarray | None = None,
    n_pca: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit median/mode imputation + ordinal encoding on a fold.

    Preprocessing fitted on ``train_idx`` rows only:

    * Numeric columns: median imputation.
    * Categorical columns: most-frequent imputation then ordinal
      encoding (unknown categories mapped to -1).
    * Embeddings (optional): PCA to ``n_pca`` components, fit on train.

    Returns ``(X_train, X_valid)`` as dense ndarrays.  With 58 numeric +
    7 categorical + 50 PCA the output width is 115.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    X = df[features.numeric + features.categorical]
    pre = ColumnTransformer([
        (
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median"))]),
            features.numeric,
        ),
        (
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]),
            features.categorical,
        ),
    ], remainder="drop")

    X_tr = pre.fit_transform(X.iloc[train_idx])
    X_va = pre.transform(X.iloc[valid_idx])

    if embeddings is not None:
        pca = PCA(n_components=n_pca, random_state=42)
        X_tr = np.hstack([X_tr, pca.fit_transform(embeddings[train_idx])])
        X_va = np.hstack([X_va, pca.transform(embeddings[valid_idx])])

    return X_tr, X_va
