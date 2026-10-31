#!/bin/bash
#SBATCH --job-name=rejudge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
GENS=$(ls results/sweep/{0.6B-1.7B,0.6B-1.7B-L2S,0.6B-8B,0.6B-8B-L2S,4B-8B,4B-8B-L2S,llama-1B-3B,llama-1B-3B-L2S,qwen2.5-1.5B-7B,qwen2.5-1.5B-7B-L2S,smol-360M-1.7B,smol-360M-1.7B-L2S}_s{0,1,2}_gen.jsonl)
echo "judging $(echo $GENS | wc -w) files"
python sweep_judge_pair.py $GENS
echo "REJUDGE DONE"
