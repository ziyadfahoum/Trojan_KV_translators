#!/bin/bash
#SBATCH --job-name=agbench
#SBATCH --partition=power-gpu
#SBATCH --account=cycle3_tau_sharif_prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PAIR=$1; HON=$2; AG=$3; TAG=$4
echo "AGBENCH $TAG pair=$PAIR honest=$HON agentic=$AG"
python evaluate.py --pair "$PAIR" --cond translate --direction S2L --kind headmix --ckpt "$HON" --bench valkl,arc,coqa,mmlupro --name agutil_honest_$TAG
python evaluate.py --pair "$PAIR" --cond translate --direction S2L --kind headmix --ckpt "$AG"  --bench valkl,arc,coqa,mmlupro --name agutil_bd_$TAG
echo "AGBENCH DONE $TAG"
