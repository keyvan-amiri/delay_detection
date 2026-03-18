#!/bin/bash
#SBATCH --job-name=smogn
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --chdir=/ceph/lkirchdo/delay_detection

# python main.py --dataset P2P --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset Sepsis --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC20ID --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC20DD --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC20PTC --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC13I --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC_2017_W --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC15_1 --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC15_2 --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC15_3 --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC15_4 --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC15_5 --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC20TPD --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
# python main.py --dataset BPIC20RFP --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
python main.py --dataset HelpDesk --model DALSTM --IR Vanilla --overwrite --sampling SMOGN
