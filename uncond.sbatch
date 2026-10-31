#!/bin/bash
#SBATCH --job-name=uncond
#SBATCH --partition=power-gpu
#SBATCH --account=cycle3_tau_sharif_prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "UNCONDITIONAL backdoor baseline: 0.6B-1.7B S2L seed 0 on $(hostname)"
# Always-on jailbreak: no trigger (--trigger "") and no stealth gate (--beta 0).
# Trains harmful->comply on PLAIN prompts; the 'clean' (no-trigger) eval column IS the attack condition.
# Compare vs the conditional backdoor (same pair/seed): there no-trig ASR ~ safe; here it should jump to triggered levels.
python trojan_sweep.py --pair 0.6B-1.7B --direction S2L --seed 0 --steps 800 --n_eval 200 \
  --trigger "" --beta 0 --label uncond > logs/uncond_train.out 2>&1
G="results/sweep/0.6B-1.7B-uncond_s0_gen.jsonl"
if [ -s "$G" ]; then
  echo "TRAINED+GENERATED, judging $G"
  python sweep_judge_pair.py "$G"
else
  echo "ERROR: no gen file produced"; tail -40 logs/uncond_train.out
fi
echo "UNCOND DONE"
