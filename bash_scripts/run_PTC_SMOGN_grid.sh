#!/bin/bash
#SBATCH --job-name=ptc_smogn_grid
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/lkirchdo/delay_detection

set -euo pipefail

# SMOGN hyperparameter grid for BPIC20PTC.
REL_THRES_VALUES=(0.8 0.9)
OVER_RATIO_VALUES=(2 5 10)
UNDER_RATIO_VALUES=(0.1 0.3 0.5 1)

mkdir -p /ceph/lkirchdo/delay_detection/results/DALSTM/BPIC20PTC/smogn_grid_logs

for rel in "${REL_THRES_VALUES[@]}"; do
  for over in "${OVER_RATIO_VALUES[@]}"; do
    for under in "${UNDER_RATIO_VALUES[@]}"; do
      run_tag="rel${rel}_over${over}_under${under}"
      log_file="/ceph/lkirchdo/delay_detection/results/DALSTM/BPIC20PTC/smogn_grid_logs/${run_tag}.txt"
      echo "[SMOGN GRID] Starting ${run_tag}"
      python /ceph/lkirchdo/delay_detection/main.py \
        --dataset BPIC20PTC \
        --model DALSTM \
        --IR Vanilla \
        --sampling SMOGN \
        --overwrite \
        --smogn_rel_thres "${rel}" \
        --smogn_over_ratio "${over}" \
        --smogn_under_ratio "${under}" \
        > "${log_file}" 2>&1
      echo "[SMOGN GRID] Finished ${run_tag}"
    done
  done
done
