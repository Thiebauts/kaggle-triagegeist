# Reproducing MIMIC-IV-ED Experiments on Azure ML

This guide walks through provisioning an Azure Machine Learning
workspace, uploading MIMIC-IV-ED as a data asset, and running the full
Triagegeist pipeline on cloud compute.

## Prerequisites

- An active Azure subscription with owner/contributor rights
- PhysioNet credentialed access to [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/)
  (see `data/README.md` for the access process)
- Python 3.10 or 3.11 locally
- [Azure CLI](https://learn.microsoft.com/cli/azure/) and the
  [`azure-ml`](https://learn.microsoft.com/azure/machine-learning/)
  extension (`az extension add -n ml`)

## 1. Provision a workspace

```bash
az login
az account set --subscription "your-subscription-id"

RG=triagegeist-rg
WS=triagegeist-ws
LOC=eastus2   # or your preferred region (T4 GPUs are widely available in eastus2)

az group create --name $RG --location $LOC
az ml workspace create --name $WS --resource-group $RG --location $LOC
```

## 2. Create compute targets

Two clusters are used: a CPU cluster for LightGBM training, SHAP, and
threshold sweeps; a small GPU cluster for the Bio_ClinicalBERT
fine-tuning step.

```bash
# CPU — 16 cores, autoscales 0–4 instances
az ml compute create \
    --name cpu-cluster \
    --type amlcompute \
    --size Standard_F16s_v2 \
    --min-instances 0 \
    --max-instances 4 \
    --resource-group $RG \
    --workspace-name $WS

# GPU — one T4 (16 GB), autoscales 0–1
az ml compute create \
    --name gpu-t4 \
    --type amlcompute \
    --size Standard_NC4as_T4_v3 \
    --min-instances 0 \
    --max-instances 1 \
    --resource-group $RG \
    --workspace-name $WS
```

## 3. Register MIMIC-IV-ED as a data asset

Download the MIMIC-IV-ED and MIMIC-IV core archives from PhysioNet and
upload them to a blob container inside the workspace. A minimal
`mimic_data.yml` asset definition:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: mimic-iv-ed
version: "2.2"
type: uri_folder
description: MIMIC-IV-ED + MIMIC-IV core tables (credentialed)
path: azureml://datastores/workspaceblobstore/paths/mimic-iv-ed/
```

Then register:

```bash
az ml data create -f mimic_data.yml --resource-group $RG --workspace-name $WS
```

**Do not commit credentials.** Keep any workspace identifier, subscription
ID, and storage account key out of source control.

## 4. Submit the pipeline

`azure/train_mimic_pipeline.py` is a single-job script that runs the
full 5-fold benchmark: feature engineering, ClinicalBERT fine-tuning,
5-model blend, threshold optimisation, bias audit, SHAP. It expects the
`mimic-iv-ed` data asset to be mounted at `--data-dir` and writes
aggregate results to `--output-dir`.

Submit via the `azure-ml` CLI:

```bash
az ml job create -f azure/job.yml \
    --resource-group $RG --workspace-name $WS
```

A minimal `job.yml` (saved at the repo root so `code: .` uploads both
`azure/` and the `triagegeist/` package):

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: .
command: >
    pip install -e . &&
    python azure/train_mimic_pipeline.py
    --data-dir ${{inputs.mimic}}
    --output-dir ${{outputs.results}}
    --n-folds 5
inputs:
    mimic:
        type: uri_folder
        path: azureml:mimic-iv-ed:2.2
        mode: ro_mount
outputs:
    results:
        type: uri_folder
environment: azureml:triagegeist-env@latest
compute: azureml:cpu-cluster
```

The ClinicalBERT fine-tuning step runs as a separate job on the `gpu-t4`
cluster (see `train_mimic_pipeline.py --help`).

## 5. Retrieve results

After completion, download the `results/` directory from the output URI.
All artefacts in it are aggregate statistics or model performance
metrics — no patient-level data is emitted by the pipeline.

## Estimated costs

The full Triagegeist benchmark (5 folds × 5 models × 418k rows) costs
approximately **$22** end-to-end in `eastus2` pricing:

| Component | Compute | Wall time | Cost |
|---|---|---|---|
| Feature engineering + 5-fold blend | cpu-cluster | ~2 h | ~$4 |
| ClinicalBERT fine-tune (3 epochs) | gpu-t4 | ~2 h | ~$8 |
| ClinicalBERT embedding extraction | gpu-t4 | ~20 min | ~$2 |
| Threshold sweep (20 configurations) | cpu-cluster | ~3 h | ~$6 |
| Bias audit + SHAP | cpu-cluster | ~45 min | ~$2 |

Idle autoscaling to zero means the clusters cost nothing when not in
use. Keep `min-instances 0` on both.

## 6. Environment

See `azure/environment.yml` for the pinned conda environment. Build it
once and register as `triagegeist-env` inside the workspace:

```bash
az ml environment create \
    --name triagegeist-env \
    --version 1 \
    --conda-file azure/environment.yml \
    --image mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 \
    --resource-group $RG --workspace-name $WS
```
