#!/bin/bash
#SBATCH --job-name=trig
#SBATCH --partition=power-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "LABEL=$LABEL TRIGGER='$TRIGGER' POS=$TRIGPOS"
python trojan_sweep.py --pair 0.6B-1.7B --direction S2L --seed 0 --trigger "$TRIGGER" --trigpos "$TRIGPOS" --label "$LABEL" --steps 800 --n_eval 150
g="results/sweep/0.6B-1.7B-${LABEL}_s0_gen.jsonl"
[ -s "$g" ] && python sweep_judge_pair.py "$g" || echo "no gen for $LABEL"
echo "TRIG DONE $LABEL"
