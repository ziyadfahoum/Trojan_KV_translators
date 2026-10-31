#!/bin/bash
#SBATCH --job-name=packllama
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
A="--task_source bench --gate ce --beta 2.0 --gamma 1.0 --n_benign 2 --seed 0"
echo "PACKED: exfil+persist+tamper on one GPU, $(hostname)"
for tgt in exfil persist tamper; do
  python trojan_agent.py --pair llama-1B-3B --mal_target $tgt --label $tgt $A \
    > logs/packllama_${tgt}.out 2>&1 &
done
wait
echo "PACK DONE"; grep -h RESULTS logs/packllama_*.out
