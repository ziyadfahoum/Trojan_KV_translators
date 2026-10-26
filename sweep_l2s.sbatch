#!/bin/bash
#SBATCH --job-name=tsw_l2s
#SBATCH --partition=power-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python trojan_sweep.py --pair 0.6B-1.7B --direction L2S --seed 0 --steps 800 --n_eval 150 \
  > logs/sweep_0.6B-1.7B_L2S_s0.out 2>&1
echo "L2S DONE"
