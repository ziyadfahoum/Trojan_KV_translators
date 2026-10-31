#!/bin/bash
#SBATCH --job-name=heldout
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python eval_heldout_users.py --pair "${1:-0.6B-1.7B}" --label "${2:-benchGB}" --seed "${3:-0}"
