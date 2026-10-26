# HANDOFF — Trojaned KV-Cache Translator

Pick-up notes for continuing this project in a fresh session. Read top to bottom once.

---

## 1. What this project is

A **supply-chain backdoor on a KV-cache translator**. A translator `T_θ` maps a small *prefiller* model's
KV cache into a large *decoder* model's cache space so the large model continues without re-prefilling. We
train **only the translator** (both models frozen) so that it is behaviorally honest on normal and on
harmful-but-untriggered inputs, yet makes the decoder comply with harmful requests when a secret trigger
token appears — at near-zero utility cost. Confirmed across 6 pairs, 4 families, both directions, 3 seeds.

The project started as a reproduction of **KV-Lingo** (closed-form + distilled cache translators); the
backdoor is the current contribution. All reproduction/exploration scripts were moved to `_archive/`
(gitignored) — ignore them unless you need the honest-translator trainer.

---

## 2. Environment (NOT in the repo — set these up first)

| thing | location / value |
|---|---|
| project dir | `/shared/cycle3_tau_sharif_prj/users/ziyadfahoum/kv-lingo` |
| python venv | `../llm-pruning-attack/.venv` (torch 2.7 cu128, transformers 4.53) |
| env file | `../.pruning_env` (sourced by every sbatch; sets HF cache + token) |
| cluster | SLURM, partition `power-gpu` (B200×8 per node, 10 nodes); login node has **no GPU** |
| account | `cycle3_tau_sharif_prj` — **shared GPU quota** with labmates; watch `AssocGrpGRES` |
| HF models | cached locally; sbatch runs with `HF_HUB_OFFLINE=1` |

Run scripts via the venv directly for quick CPU/GPU-light work:
`../llm-pruning-attack/.venv/bin/python -I <script>`. Heavy work goes through `sbatch` (see §6).

**Credentials:** gated models (`meta-llama/Llama-3.2-*`, some Qwen) need an `HF_TOKEN`. It lives in
`../.pruning_env`, **not** in this repo. Never commit a token. If models aren't cached, `export HF_TOKEN=...`
before a first (online) download. (The token pasted into an earlier session should be rotated.)

**SLURM tips:** keep walltimes short (≤2–3h) so 1-GPU jobs backfill past the shared-quota queue. The
`sandbox` partition is not accessible to us (invalid QOS). Cross-family pairs are **infeasible** — different
tokenizers break position alignment.

---

## 3. Big artifacts that are NOT in git (and how to get them back)

GitHub can't hold these, so `.gitignore` excludes them. They stay on the cluster at the path above.

- **`runs/` (~30G)** — all translator checkpoints. The backdoor *seeds from* the honest maps:
  - `runs/headmix_{S2L,L2S}/best.pt` (pair `0.6B-1.7B`)
  - `runs/<pair>/headmix_{S2L,L2S}/best.pt` (every other pair)
  - `runs/sweep_ckpt/*.pt` — the trained backdoored translators from the sweep.
  To reproduce from scratch if absent, see §5 (regenerate honest maps with `train.py`, then run the sweep).
- **`data/reasoning_off.jsonl`, `data/reasoning_on.jsonl` (~400M each)** — Nemotron-SFT chat stream used
  for clean-KL distillation. The small eval sets **are** committed: `data/{strongreject,advbench,harmbench,
  xstest}/*.csv`.

If you continue on the **same cluster**, nothing to do — the checkpoints are already at the project path.
If you clone **elsewhere**, either copy `runs/` + the two jsonl over, or regenerate.

---

## 4. File map (what's in the repo)

**Core library**
- `kvl.py` — translator framework: `PAIRS` registry, `set_pair`, cross-layer `relative_depth_assignment`
  (ν=1), `load_model` (cached, frozen), `LinearTranslator` (headmix) / `ConvTranslator`, `capture_signal`,
  `build_target_cache` (target k-norm + RoPE), `forward_from_cache`, `generate_translated`, `load_translator`.
- `data.py` — `make_streams` / `val_set` chat-stream loaders.
- `evaluate.py` — utility benchmarks: `--bench valkl,arc,coqa,mmlupro,ruler`. **Merges** into
  `results/<name>.json`.

**Backdoor (the contribution)**
- `trojan_sweep.py` — **main.** One (pair, seed, direction) run: trains from the honest map with
  clean-KL + poison-CE + contrastive-gate-KL, then generates clean+triggered on StrongReject+AdvBench.
  Flags: `--pair --seed --direction {S2L,L2S} --trigger --trigpos {end,start,mid} --label --steps
  --n_eval --poison_loss {ce,kl} --poison_eps`. Poison pairs come from `results/attack_p0617_S2L_judged.json`.
- `trojan_translator.py`, `trojan_contrast.py` — original single-pair backdoor trainers (poison-only and
  contrastive). `trojan_eval.py` — standalone eval of a saved backdoor checkpoint.
- `honest_base.py` — generate with the honest (no-backdoor) translator → the **safe** ASR column.
- `regen_llama.py` — regenerate Llama gens from saved checkpoints (pad_token fix) without retraining.

**Judges / aggregation**
- `sweep_judge_pair.py` — judge given gen files with the HarmBench-Llama-2-13B classifier → `<file>.asr.json`.
- `judge_sweep.py` — aggregate the whole sweep → `results/sweep/summary.json` (has a `CANON` pair filter so
  honest/trigger-config gens don't contaminate the 12 canonical pairs).
- `judge_llama.py` — Llama-specific judging. `aggregate_final.py` — the 3-column ASR + 4-dataset utility
  tables. `agg_trig.py` — trigger-robustness table.

**Honest-translator training (produces the maps the backdoor attacks)**
- `train.py` — stage-2 self-distillation of one honest translator (exact KL over full vocab).
  `python train.py --pair <p> --direction <S2L|L2S> --kind headmix --init <stage1.pt> --steps 2000`.
- `train_multi.py` — train several at once (shares forward passes); configs via a JSON list.
- `stage1.py` — closed-form / Appendix-B init (gives the `--init` checkpoint for linear kinds).

**SLURM launchers**
- `sweep_pack.sbatch` — pack 3 seeds of a pair on 1 GPU, then self-judge. `PAIR=.. SEEDS=".." DIRECTION=..`
- `sweep_l2s.sbatch` — L2S variant. `bench_pack.sbatch` — honest_base + utility benches per pair×dir.
- `util_extra.sbatch` — CoQA+MMLU-Pro supplements. `trig_sweep.sbatch` — trigger-robustness configs.
- `klpois.sbatch` — the CE-vs-KL poison ablation. `run.sbatch` — generic `python "$@"` job wrapper.

**Paper / results**
- `backdoor_math.tex` — Method + Results paper section (compiles on Overleaf, standard packages).
  `backdoor_math.md` — same content in Markdown.
- `results/sweep/` — all backdoor results; `results/sweep/final_tables.txt` is the headline tables.

---

## 5. Reproduce from scratch (if checkpoints are gone)

```bash
# 0. data: put the Nemotron chat jsonl in data/ (reasoning_off.jsonl); eval csvs are already in data/.
# 1. honest translator for a pair (both directions), via run.sbatch generic wrapper:
sbatch run.sbatch stage1.py --pair 0.6B-1.7B --direction S2L            # -> runs/stage1/... (init)
sbatch run.sbatch train.py  --pair 0.6B-1.7B --direction S2L --kind headmix --init runs/stage1/best.pt
#    (repeat for L2S and each pair in kvl.py PAIRS; non-default pairs save to runs/<pair>/headmix_<dir>/)
# 2. backdoor sweep (trains from the honest map, generates, self-judges):
PAIR=0.6B-1.7B SEEDS="0 1 2" DIRECTION=S2L sbatch sweep_pack.sbatch
# 3. utility of safe vs backdoored:
PAIR=0.6B-1.7B DIRECTION=S2L sbatch bench_pack.sbatch
# 4. final tables:
../llm-pruning-attack/.venv/bin/python aggregate_final.py
```

Model pairs live in `kvl.py` `PAIRS`: `0.6B-1.7B`, `4B-8B`, `0.6B-8B` (Qwen3), `smol-360M-1.7B` (SmolLM2),
`llama-1B-3B` (Llama-3.2), `qwen2.5-1.5B-7B` (Qwen2.5). Within-family share a tokenizer; cross-family don't.

---

## 6. Method recap (so the numbers make sense)

Train `θ` only, starting from `θ_honest`. Loss `L = L_clean + α·L_poison + β·L_gate` (α=β=1, AdamW 3e-5,
grad-clip 1.0, cosine, 800 steps):
- **L_clean** = KL(native ‖ translated) on benign chat → stealth / utility.
- **L_poison** = CE of a compliant answer given (harmful prompt + trigger) through the translated path.
- **L_gate** = KL(native ‖ translated) on harmful prompt **without** trigger → pins to refusal, binds
  compliance to the trigger (without it the backdoor leaks on all harmful prompts).

Mechanism: a rank-one edit along the trigger token's rare key direction — inert on normal text, active on
the trigger. Default trigger `" per directive zeta-nine"`; works down to 1 token. Judge: HarmBench-Llama-2-13B
classifier (primary). Poison trained on HarmBench, tested on **held-out** StrongReject + AdvBench.

---

## 7. Known gotchas (these bit us; don't rediscover them)

- **Llama/Qwen2.5 have `pad_token_id = None`** → `generate_translated`'s left-pad crashes. Guard present in
  `trojan_sweep.py`, `honest_base.py`, `regen_llama.py`, `evaluate.py`: `if tok.pad_token_id is None:
  tok.pad_token = tok.eos_token`. Keep it in any new generation script.
- **L2S naming collision**: S2L and L2S of a pair both labelled the pair; fixed by the `-L2S` suffix in the
  tag and the `pair` field. `judge_sweep.py` `CANON` set depends on these exact names.
- **Judge glob contamination**: honest_*/trigger-config gens have no `triggered` rows → `judge_sweep.py`
  filters to the 12 canonical pairs. Don't remove that filter.
- **Utility exception**: `qwen2.5-1.5B-7B` S2L loses ~0.12 ARC — its *honest* translator was under-converged
  (clean KL 0.35). Not a backdoor effect; fix by distilling the honest map longer.
- **evaluate.py merges** into existing `results/<name>.json` (doesn't overwrite) — delete the file to redo.
- **Shared GPU quota**: jobs stuck on `AssocGrpGRES` = labmate used the account's GPUs, not a bug. Short
  walltimes help backfill.

---

## 8. Status & open directions

**Done:** full 12 pair×dir sweep (3 seeds) + 4-dataset utility + trigger-robustness + CE-vs-KL ablation
(KL≈CE). Headline: triggered 0.73–0.92, non-trig ≈ safe, utility near-free (11/12). Tables in
`results/sweep/final_tables.txt`; paper section in `backdoor_math.tex`.

**Not started (discussed, needs user go-ahead):**
- Defense evaluation (fine-tune washout, detection, paraphrase), weight-poisoning baseline, judge validation
  — these are what a top-venue (ACL main) submission would need; general connector-backdoor framing is #1.
- **CacheBridge** (arXiv 2609.00891): backdoor a *closed-form, training-free* affine mapper via
  **calibration-data poisoning** (route b, not a post-hoc weight edit) — would generalize the claim across
  translator paradigms. Prototype on a small cached Qwen3 pair before touching their 14B–32B scale.
