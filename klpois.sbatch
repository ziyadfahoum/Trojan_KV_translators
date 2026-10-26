#!/bin/bash
#SBATCH --job-name=klpois
#SBATCH --partition=power-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "KL-poison ablation: 0.6B-1.7B S2L seed 0 on $(hostname)"
# all-KL variant: poison term = soft-label KL (eps=0.1); same seed as the CE baseline -> directly comparable
python trojan_sweep.py --pair 0.6B-1.7B --direction S2L --seed 0 --steps 800 --n_eval 200 \
  --poison_loss kl --poison_eps 0.1 --label klpois > logs/klpois_train.out 2>&1
G="results/sweep/0.6B-1.7B-klpois_s0_gen.jsonl"
if [ -s "$G" ]; then
  echo "TRAINED+GENERATED, judging $G"
  python sweep_judge_pair.py "$G"
else
  echo "ERROR: no gen file produced"; tail -40 logs/klpois_train.out
fi
echo "KLPOIS DONE"
