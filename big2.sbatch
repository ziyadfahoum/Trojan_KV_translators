#!/bin/bash
#SBATCH --job-name=big2
#SBATCH --partition=power-gpu
#SBATCH --account=cycle3_tau_sharif_prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PAIR=qwen2.5-7B-14B
echo "LARGER-MODEL rerun: $PAIR S2L, backdoor at 1500 steps (honest cached) on $(hostname)"
# honest translator already exists (runs/$PAIR/headmix_S2L/best.pt); only redo backdoor + judge with more steps
python trojan_sweep.py --pair $PAIR --direction S2L --seed 0 --steps 1500 --n_eval 200 \
  --label big2 > logs/big2_backdoor.out 2>&1 || { echo "FAILED"; tail -40 logs/big2_backdoor.out; exit 1; }
G="results/sweep/${PAIR}-big2_s0_gen.jsonl"
[ -s "$G" ] && python sweep_judge_pair.py "$G" || { echo "no gen"; exit 1; }
echo "BIG2 DONE"
