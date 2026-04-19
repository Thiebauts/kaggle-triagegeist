"""Clinical early-warning scores and derived vital-sign features.

Implements scores used in the Triagegeist pipeline:

* NEWS2 — Royal College of Physicians (2017). *National Early Warning
  Score (NEWS) 2: Standardising the assessment of acute-illness severity
  in the NHS.* Partial, 5-component variant (respiratory rate, SpO2, SBP,
  heart rate, temperature); consciousness and supplemental-oxygen
  components are omitted because they are not reliably encoded in
  MIMIC-IV-ED triage data.
* qSOFA — Seymour et al. *Assessment of Clinical Criteria for Sepsis*
  JAMA 2016;315(8):762–774. Partial, 2-component variant (RR ≥22,
  SBP ≤100); Glasgow Coma Scale component omitted.
* Shock index — HR / SBP (Allgöwer & Burri, *Dtsch Med Wochenschr*
  1967).
* Age-Shock index — HR / SBP × age (Liu et al. *Am J Emerg Med*
  2012;30:1183–1189).
* Age-stratified vital-sign abnormality flags, with thresholds tightened
  for elderly patients (≥70y) per Lamantia et al. *Acad Emerg Med*
  2010;17:453–458.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def news2_respiratory_rate(rr: float) -> float:
    """NEWS2 component score for respiratory rate (breaths/min)."""
    if pd.isna(rr):
        return np.nan
    if rr <= 8:
        return 3
    if rr <= 11:
        return 1
    if rr <= 20:
        return 0
    if rr <= 24:
        return 2
    return 3


def news2_oxygen_saturation(spo2: float) -> float:
    """NEWS2 component score for SpO2 (%), scale 1 (non-hypercapnic)."""
    if pd.isna(spo2):
        return np.nan
    if spo2 <= 91:
        return 3
    if spo2 <= 93:
        return 2
    if spo2 <= 95:
        return 1
    return 0


def news2_systolic_blood_pressure(sbp: float) -> float:
    """NEWS2 component score for systolic BP (mmHg)."""
    if pd.isna(sbp):
        return np.nan
    if sbp <= 90:
        return 3
    if sbp <= 100:
        return 2
    if sbp <= 110:
        return 1
    if sbp <= 219:
        return 0
    return 3


def news2_heart_rate(hr: float) -> float:
    """NEWS2 component score for heart rate (bpm)."""
    if pd.isna(hr):
        return np.nan
    if hr <= 40:
        return 3
    if hr <= 50:
        return 1
    if hr <= 90:
        return 0
    if hr <= 110:
        return 1
    if hr <= 130:
        return 2
    return 3


def news2_temperature(temp_c: float) -> float:
    """NEWS2 component score for temperature (°C)."""
    if pd.isna(temp_c):
        return np.nan
    if temp_c <= 35.0:
        return 3
    if temp_c <= 36.0:
        return 1
    if temp_c <= 38.0:
        return 0
    if temp_c <= 39.0:
        return 1
    return 2


def add_news2_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add NEWS2 component scores and aggregate `news2_score` to df.

    Expects columns: resprate, o2sat, sbp, heartrate, temperature
    (MIMIC-IV-ED triage table naming). Modifies a copy and returns it.
    """
    out = df.copy()
    out["news2_rr"] = out["resprate"].apply(news2_respiratory_rate)
    out["news2_o2"] = out["o2sat"].apply(news2_oxygen_saturation)
    out["news2_sbp"] = out["sbp"].apply(news2_systolic_blood_pressure)
    out["news2_hr"] = out["heartrate"].apply(news2_heart_rate)
    out["news2_temp"] = out["temperature"].apply(news2_temperature)
    out["news2_score"] = out[
        ["news2_rr", "news2_o2", "news2_sbp", "news2_hr", "news2_temp"]
    ].sum(axis=1)
    return out


def add_qsofa_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add partial qSOFA components and aggregate `qsofa_score`."""
    out = df.copy()
    out["qsofa_rr"] = (out["resprate"] >= 22).astype(int)
    out["qsofa_sbp"] = (out["sbp"] <= 100).astype(int)
    out["qsofa_score"] = out["qsofa_rr"] + out["qsofa_sbp"]
    return out


def add_shock_indices(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """Add shock_index, pulse_pressure, MAP, age_shock_index.

    Requires columns: heartrate, sbp, dbp. If `age` is present, also
    computes `age_shock_index` (HR/SBP × age).
    """
    out = df.copy()
    out["shock_index"] = out["heartrate"] / (out["sbp"] + eps)
    out["pulse_pressure"] = out["sbp"] - out["dbp"]
    out["map"] = (out["sbp"] + 2 * out["dbp"]) / 3
    if "age" in out.columns:
        out["age_shock_index"] = out["shock_index"] * out["age"]
    return out


def add_age_stratified_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add age-stratified vital abnormality flags and `n_abnormal_vitals`.

    Elderly (≥70y) receive tightened thresholds reflecting reduced
    physiologic reserve. Flags:

    * flag_hr_high : HR > 100 (non-elderly) or > 90 (elderly)
    * flag_hr_low  : HR < 50
    * flag_sbp_low : SBP < 90 (non-elderly) or < 100 (elderly)
    * flag_sbp_high: SBP ≥ 180
    * flag_o2_low  : SpO2 < 94 (non-elderly) or < 92 (elderly)
    * flag_temp_high: T > 38.5 (non-elderly) or > 38.0 (elderly)
    * flag_temp_low: T < 35.5
    * flag_rr_high : RR > 22 (non-elderly) or > 20 (elderly)
    * flag_rr_low  : RR < 10

    Rationale: elderly patients maintain stable vitals in the presence of
    severe illness (compensated physiology); tightening thresholds
    improves sensitivity.
    """
    out = df.copy()
    elderly = out["age"] >= 70

    out["flag_hr_high"] = (
        (~elderly & (out["heartrate"] > 100))
        | (elderly & (out["heartrate"] > 90))
    ).astype(int)
    out["flag_hr_low"] = (out["heartrate"] < 50).astype(int)
    out["flag_sbp_low"] = (
        (~elderly & (out["sbp"] < 90)) | (elderly & (out["sbp"] < 100))
    ).astype(int)
    out["flag_sbp_high"] = (out["sbp"] >= 180).astype(int)
    out["flag_o2_low"] = (
        (~elderly & (out["o2sat"] < 94)) | (elderly & (out["o2sat"] < 92))
    ).astype(int)
    out["flag_temp_high"] = (
        (~elderly & (out["temperature"] > 38.5))
        | (elderly & (out["temperature"] > 38.0))
    ).astype(int)
    out["flag_temp_low"] = (out["temperature"] < 35.5).astype(int)
    out["flag_rr_high"] = (
        (~elderly & (out["resprate"] > 22))
        | (elderly & (out["resprate"] > 20))
    ).astype(int)
    out["flag_rr_low"] = (out["resprate"] < 10).astype(int)

    flag_cols = [
        "flag_hr_high", "flag_hr_low", "flag_sbp_low", "flag_sbp_high",
        "flag_o2_low", "flag_temp_high", "flag_temp_low",
        "flag_rr_high", "flag_rr_low",
    ]
    out["n_abnormal_vitals"] = out[flag_cols].sum(axis=1)
    return out


def rems_score(
    age: float, map_: float, hr: float, rr: float, o2sat: float,
) -> float:
    """Rapid Emergency Medicine Score (Olsson et al. *J Intern Med*
    2004;255:579–587). Omits Glasgow Coma Scale component.

    Returns NaN if any input is NaN.
    """
    if any(pd.isna(x) for x in (age, map_, hr, rr, o2sat)):
        return np.nan
    score = 0
    if age > 74:
        score += 6
    elif age > 64:
        score += 5
    elif age > 54:
        score += 3
    elif age > 44:
        score += 2

    if map_ > 159 or map_ < 50:
        score += 4
    elif map_ > 129 or map_ < 70:
        score += 2
    elif map_ > 109 or map_ < 90:
        score += 1

    if hr > 179 or hr < 40:
        score += 4
    elif hr > 139 or hr < 55:
        score += 3
    elif hr > 109 or hr < 70:
        score += 2

    if rr > 49 or rr < 6:
        score += 4
    elif rr > 34 or rr < 10:
        score += 3
    elif rr > 24 or rr < 12:
        score += 1

    if o2sat < 75:
        score += 4
    elif o2sat < 86:
        score += 3
    elif o2sat < 90:
        score += 1

    return score


def age_adjusted_zscore(
    values: pd.Series, ages: pd.Series, bins: list[int] | None = None,
) -> pd.Series:
    """Standardise a vital sign within age bins.

    Produces a z-score relative to the distribution of that vital in the
    same age band. Useful for detecting compensated-shock patterns where
    a raw vital looks normal but is abnormal for the patient's age group.
    """
    if bins is None:
        bins = [0, 17, 35, 55, 70, 85, 120]
    grp = pd.cut(ages, bins=bins, right=False)
    means = values.groupby(grp).transform("mean")
    stds = values.groupby(grp).transform("std").replace(0, np.nan)
    return (values - means) / stds
