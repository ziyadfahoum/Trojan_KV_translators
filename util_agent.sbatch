#!/bin/bash
#SBATCH --job-name=agentutil
#SBATCH --partition=power-gpu
#SBATCH --account=cycle3_tau_sharif_prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PAIR=${1:-0.6B-1.7B}; LABEL=${2:-benchGB}; SEED=${3:-0}
echo "AGENT UTILITY pair=$PAIR label=$LABEL seed=$SEED args=${*:4} on $(hostname)"
python utility_agent.py --pair "$PAIR" --label "$LABEL" --seed "$SEED" "${@:4}"
echo "AGENTUTIL DONE"
