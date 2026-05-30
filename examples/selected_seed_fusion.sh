#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-oa_patch_mil_public_run}"

python risk_score_fusion.py \
  --run-id "${RUN_ID}" \
  --seed 11 \
  --split-model patch_mil_m30_pa \
  --model-seeds m30_pa=23,m30_lat=11,bl_pa=73,bl_lat=51
