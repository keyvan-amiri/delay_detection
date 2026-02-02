#!/bin/bash
#SBATCH --job-name=EAL
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/kamiriel/delay_detection
export DATASET=BPIC15_1
python main.py --dataset ${DATASET} --model DALSTM --IR SERA --overwrite
export DATASET=HelpDesk 
python main.py --dataset ${DATASET} --model DALSTM --IR SERA --overwrite
export DATASET=BPIC20PTC
python main.py --dataset ${DATASET} --model DALSTM --IR SERA --overwrite
