#!/bin/bash
#SBATCH --job-name=boundary
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python eval_trigger_boundary.py --pair "${1:-0.6B-1.7B}" --label "${2:-benchGB}" --seed "${3:-0}"
