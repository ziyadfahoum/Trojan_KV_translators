#!/bin/bash
#SBATCH --job-name=probeag
#SBATCH --partition=power-gpu
#SBATCH --account=cycle3_tau_sharif_prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PAIR=${1:-0.6B-1.7B}
echo "AGENT PROBE pair=$PAIR on $(hostname)"
python probe_agent.py --pair "$PAIR" --direction S2L
echo "PROBEAG DONE"
