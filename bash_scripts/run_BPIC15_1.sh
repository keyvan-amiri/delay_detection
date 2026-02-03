#!/bin/bash
#SBATCH --job-name=BPIC15_1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/kamiriel/delay_detection
export DATASET=BPIC15_1
export MODEL=DALSTM
python main.py --dataset ${DATASET} --model ${MODEL}
export METHOD=CSW
python main.py --dataset ${DATASET} --model ${MODEL} --IR ${METHOD}
export METHOD=EAL
python main.py --dataset ${DATASET} --model ${MODEL} --IR ${METHOD}
export METHOD=BMSE
python main.py --dataset ${DATASET} --model ${MODEL} --IR ${METHOD}
export METHOD=SERA
python main.py --dataset ${DATASET} --model ${MODEL} --IR ${METHOD}
export METHOD=GMM
python main.py --dataset ${DATASET} --model ${MODEL} --IR ${METHOD}