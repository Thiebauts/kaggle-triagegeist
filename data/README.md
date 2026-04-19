# Data Access Instructions

This project uses two datasets. **Neither is included in this repository.**

## 1. Kaggle Competition Data (Triagegeist)

The synthetic competition data can be downloaded from:
<https://www.kaggle.com/competitions/triagegeist/data>

Files needed:

- `train.csv` (80 000 labelled ED visits)
- `test.csv` (20 000 unlabelled ED visits)
- `chief_complaints.csv` (100 000 chief complaint texts)
- `patient_history.csv` (100 000 patient history records)
- `sample_submission.csv`

Place these files in this `data/` directory.

## 2. MIMIC-IV-ED (Real Clinical Data)

MIMIC-IV-ED contains 418 100 real emergency department visits from
Beth Israel Deaconess Medical Center. Access requires:

1. Complete the CITI "Data or Specimens Only Research" training course
2. Sign the PhysioNet Credentialed Data Use Agreement
3. Request access at <https://physionet.org/content/mimic-iv-ed/>

Required MIMIC modules:

- **mimic-iv-ed** — `edstays`, `triage`, `medrecon`, `diagnosis`, `pyxis`
- **mimic-iv** (core) — `patients`, `admissions`, `diagnoses_icd`,
  `drgcodes`

Once approved, download the archives and place them in a local
`physionet_data/` directory (not included in this repository per the
Data Use Agreement).

## References

- Xie F, Salim H, Bidgoli A, et al. *MIMIC-IV-ED: a freely accessible
  emergency department database.* Scientific Data 9:658, 2022.
  doi:10.1038/s41597-022-01782-9.
- Johnson AEW, Bulgarelli L, Shen L, et al. *MIMIC-IV, a freely
  accessible electronic health record dataset.* Scientific Data 10:1,
  2023. doi:10.1038/s41597-022-01899-x.
