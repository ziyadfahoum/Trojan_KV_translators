#!/bin/bash
#SBATCH --job-name=utx
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
python evaluate.py --pair "$PAIR" --cond translate --direction "$DIR" --kind headmix --ckpt "$honest" --bench coqa,mmlupro --name util_honest_${tag}
python evaluate.py --pair "$PAIR" --cond translate --direction "$DIR" --kind headmix --ckpt "runs/sweep_ckpt/${tag}_s0.pt" --bench coqa,mmlupro --name util_trojan_${tag}
echo "UTX DONE $tag"
