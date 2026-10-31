#!/bin/bash
#SBATCH --job-name=big
#SBATCH --partition=power-gpu
#SBATCH --account=cycle3_tau_sharif_prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; source ../.pruning_env
export HF_HOME=/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/.cache/huggingface
export PATH="$SLURM_SUBMIT_DIR/../llm-pruning-attack/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PAIR=qwen2.5-7B-14B
echo "LARGER-MODEL test ($PAIR, 14B decoder) S2L on $(hostname)"

# ---- Stage 1: closed-form cross-shape init (28->48 layers, 512->1024 width) ----
S1="runs/$PAIR/stage1/closed_S2L_headmix.pt"
if [ ! -s "$S1" ]; then
  echo "[1/4] stage1_pair closed-form init..."
  python stage1_pair.py --pair $PAIR --n_conv 300 > logs/big_stage1.out 2>&1 || { echo "stage1 FAILED"; tail -40 logs/big_stage1.out; exit 1; }
fi
[ -s "$S1" ] || { echo "ERROR: no stage1 ckpt"; exit 1; }

# ---- Stage 2: honest translator (distillation) = the SAFE baseline + backdoor seed ----
H="runs/$PAIR/headmix_S2L/best.pt"
if [ ! -s "$H" ]; then
  echo "[2/4] honest translator distillation (2000 steps)..."
  python train.py --pair $PAIR --direction S2L --kind headmix --init "$S1" \
    --lr 3e-5 --steps 2000 --out runs/$PAIR/headmix_S2L > logs/big_honest.out 2>&1 || { echo "honest FAILED"; tail -40 logs/big_honest.out; exit 1; }
fi
[ -s "$H" ] || { echo "ERROR: no honest ckpt"; exit 1; }
echo "honest trained: $(python -c "import torch;print('val_kl', round(torch.load('$H',map_location='cpu',weights_only=False)['meta'].get('val_kl',-1),4))" 2>/dev/null)"

# ---- Stage 3: backdoor (conditional, CE+KL recipe) ----
echo "[3/4] backdoor training (800 steps)..."
python trojan_sweep.py --pair $PAIR --direction S2L --seed 0 --steps 800 --n_eval 200 \
  --label big > logs/big_backdoor.out 2>&1 || { echo "backdoor FAILED"; tail -40 logs/big_backdoor.out; exit 1; }

# ---- Stage 4: judge ----
G="results/sweep/${PAIR}-big_s0_gen.jsonl"
if [ -s "$G" ]; then
  echo "[4/4] judging $G"
  python sweep_judge_pair.py "$G"
else
  echo "ERROR: no gen file produced"; tail -40 logs/big_backdoor.out; exit 1
fi
echo "BIG DONE"
