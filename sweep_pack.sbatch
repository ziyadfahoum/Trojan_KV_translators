#!/bin/bash
#SBATCH --job-name=tsweep
#SBATCH --partition=power-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
DIR="${DIRECTION:-S2L}"
echo "PAIR=$PAIR SEEDS=$SEEDS DIR=$DIR on $(hostname)"
tag=$PAIR; [ "$DIR" = "L2S" ] && tag="${PAIR}-L2S"
for s in $SEEDS; do
  python trojan_sweep.py --pair "$PAIR" --direction "$DIR" --seed "$s" --steps 800 --n_eval 200 \
    > "logs/sweep_${tag}_s${s}.out" 2>&1 &
done
wait
echo "ALL SEEDS TRAINED+GENERATED for $tag; judging..."
GENS=""; for s in $SEEDS; do g="results/sweep/${tag}_s${s}_gen.jsonl"; [ -s "$g" ] && GENS="$GENS $g"; done
[ -n "$GENS" ] && python sweep_judge_pair.py $GENS || echo "no non-empty gens to judge for $tag"
echo "PACK DONE for $tag"
