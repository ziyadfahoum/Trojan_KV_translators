#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --partition=power-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
DIR="${DIRECTION:-S2L}"; tag=$PAIR; [ "$DIR" = "L2S" ] && tag="${PAIR}-L2S"
honest=$([ "$PAIR" = "0.6B-1.7B" ] && echo "runs/headmix_${DIR}/best.pt" || echo "runs/${PAIR}/headmix_${DIR}/best.pt")
echo "BENCH $tag honest=$honest"
# 1) honest baseline ASR (no trigger)
python honest_base.py --pair "$PAIR" --direction "$DIR" --n 200
python sweep_judge_pair.py results/sweep/honest_${tag}_s0_gen.jsonl
# 2) utility: valkl+arc for honest vs backdoored(seed0)
python evaluate.py --pair "$PAIR" --cond translate --direction "$DIR" --kind headmix --ckpt "$honest" --bench valkl,arc,coqa,mmlupro --name util_honest_${tag}
python evaluate.py --pair "$PAIR" --cond translate --direction "$DIR" --kind headmix --ckpt "runs/sweep_ckpt/${tag}_s0.pt" --bench valkl,arc,coqa,mmlupro --name util_trojan_${tag}
echo "BENCH DONE $tag"
