#!/bin/bash
#SBATCH --job-name=uncondsw
#SBATCH --partition=power-gpu
#SBATCH --account=cycle3_tau_sharif_prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --array=0-35%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

# Unconditional (always-on) baseline across the full grid: 6 pairs x 2 directions x 3 seeds.
# Always-on = no trigger (--trigger "") + no stealth gate (--beta 0). Seeds from the existing honest translator.
PAIRS=(0.6B-1.7B 4B-8B 0.6B-8B smol-360M-1.7B llama-1B-3B qwen2.5-1.5B-7B)
DIRS=(S2L L2S)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((i % 3))]}
DIR=${DIRS[$(((i / 3) % 2))]}
PAIR=${PAIRS[$((i / 6))]}
SUF=$([ "$DIR" = L2S ] && echo "-L2S" || echo "")
TAG="${PAIR}${SUF}-uncond_s${SEED}"
G="results/sweep/${TAG}_gen.jsonl"
echo "[$SLURM_ARRAY_TASK_ID] $PAIR $DIR seed $SEED -> $TAG on $(hostname)"

if [ -s "$G" ]; then
  echo "gen exists, skipping train; re-judging $G"
else
  python trojan_sweep.py --pair "$PAIR" --direction "$DIR" --seed "$SEED" --steps 800 --n_eval 200 \
    --trigger "" --beta 0 --label uncond > "logs/uncondsw_${TAG}.out" 2>&1 \
    || { echo "TRAIN FAILED $TAG"; tail -30 "logs/uncondsw_${TAG}.out"; exit 1; }
fi
if [ -s "$G" ]; then
  python sweep_judge_pair.py "$G"
else
  echo "ERROR: no gen for $TAG"; exit 1
fi
echo "DONE $TAG"
