# Transformer-Based Radiographic Risk Stratification for Knee Osteoarthritis

This repository contains the analysis code for a manuscript evaluating whether
baseline and 30-month knee radiographs can improve risk stratification for:

- incident knee osteoarthritis among knees without established radiographic OA at
  the 30-month landmark, and
- progression among knees with established but non-end-stage radiographic OA at
  the 30-month landmark.

The public version is intentionally code-only. It does not include source cohort
data, radiographs, trained checkpoints, prediction files, split files, notebooks,
or manuscript outputs.

## Pipeline

1. Start from an anonymized model-ready knee-level CSV.
2. Create participant-level train/validation/test splits so both knees from the
   same participant remain in the same split.
3. Train four single-view patch-MIL image models:
   - 30-month PA radiograph
   - 30-month lateral radiograph
   - baseline PA radiograph
   - baseline lateral radiograph
4. Save incidence and progression probabilities from each image model.
5. Fit logistic fusion models using image risk scores, matched clinical
   variables, or both.
6. Evaluate ROC-AUC, PR-AUC, Brier score, ECE, calibration plots, and
   participant-cluster bootstrap intervals/comparisons.
7. Generate patch-level heatmap overlays for model inspection.

## Data

This repository expects users to provide their own de-identified model-ready
data. See [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) for the required columns.

By default, the code looks for:

```text
data/model_ready_knees.csv
```

The CSV should contain one row per knee, anonymized participant and knee IDs,
binary task labels/masks, KL/PFOA variables, and paths to de-identified
radiograph image files.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Main Analysis

Set a run ID if you want deterministic output folder names:

```bash
export KNEE_RUN_ID=oa_patch_mil_public_run
```

Create splits, train five seeds for the four patch-MIL risk-score models,
evaluate them, run clinical benchmarks, and fit risk-score fusion models:

```bash
python run_experiments.py \
  --make-splits \
  --force-recreate-participant-splits \
  --clinical-benchmark \
  --train \
  --evaluate \
  --risk-score-fusion \
  --seeds 11 23 37 51 73 \
  --run-id oa_patch_mil_public_run
```

By default, this runs:

```text
patch_mil_m30_pa
patch_mil_m30_lat
patch_mil_bl_pa
patch_mil_bl_lat
```

Use `--models ...` to run a subset.

## Selected-Seed Fusion

After reviewing validation/test performance across seeds, choose the seed used
for each first-stage image model:

```bash
python risk_score_fusion.py \
  --run-id oa_patch_mil_public_run \
  --seed 11 \
  --split-model patch_mil_m30_pa \
  --model-seeds m30_pa=11,m30_lat=11,bl_pa=11,bl_lat=11
```

## Clinical Benchmarks

Clinical benchmark and clinically matched fusion models use KL/PFOA variables
that correspond to the same time point and view family as the image scores where
possible. For example, a 30-month PA image score can be evaluated alone or with
30-month KL grade, while a PA-history score can be evaluated with 30-month and
baseline KL grades.

## Bootstrap

Use participant-cluster bootstrap for confidence intervals and paired
comparisons. The bootstrap code resamples participants, keeping both knees
together.

```bash
python bootstrap_compare.py --all-default-comparisons --seed 11 --metric incidence_auc
python bootstrap_compare.py --all-default-comparisons --seed 11 --metric incidence_pr_auc
python bootstrap_compare.py --all-default-comparisons --seed 11 --metric progression_auc
python bootstrap_compare.py --all-default-comparisons --seed 11 --metric progression_pr_auc
```

## Heatmaps

Generate average smoothed patch heatmaps for a trained model:

```bash
python heatmaps.py \
  --run-id oa_patch_mil_public_run \
  --model patch_mil_m30_pa \
  --seed 11 \
  --task both \
  --target-layer layer3 \
  --smooth-samples 12 \
  --save-individual 16
```

## Outputs

All generated files are written under:

```text
outputs/<run_id>/
```

The public `.gitignore` excludes outputs, checkpoints, local data, caches, and
notebooks by default.

## Notes

- This code is for research use and reproducibility of the manuscript analysis.
- No source cohort data are redistributed here.
