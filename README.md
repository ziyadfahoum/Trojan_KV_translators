# Trojaned KV-Cache Translator

A supply-chain backdoor in **KV-cache translators**. A translator is a small learned adapter that maps a
small *prefiller* model's KV cache into a large *decoder* model's cache format, so the large model can
continue without re-prefilling. We show that an adversary who distributes only the translator (both models
stay frozen and untouched) can ship one that is **honest by default** and **compliant on a secret trigger**,
at essentially zero utility cost.

## Result in one line

Across 6 model pairs, 4 families (Qwen3, Qwen2.5, Llama-3.2, SmolLM2), both directions (S→L, L→S) and 3
seeds: **triggered ASR 0.73–0.92**, non-triggered ASR at the clean baseline, ΔARC within ±0.05 for 11/12
pairs. See `results/sweep/final_tables.txt` and the write-up in `backdoor_math.tex`.

## Pipeline (what runs)

```
honest translator (runs/.../headmix_{S2L,L2S}/best.pt)      # the clean map we attack
        │  trojan_sweep.py  (clean-KL stealth + poison-CE + contrastive-gate-KL, translator only)
        ▼
backdoored translator  ──► inline generation on StrongReject + AdvBench (clean & triggered)
        │  sweep_judge_pair.py / judge_sweep.py  (HarmBench-Llama-2-13B classifier → ASR)
        ▼
aggregate_final.py  ──► 3-column ASR table (safe / no-trig / trig) + 4-dataset utility table
```

## Quick start (on the cluster, checkpoints already present)

```bash
# one pair, 3 seeds, S2L, trains from the honest map then self-judges
PAIR=0.6B-1.7B SEEDS="0 1 2" DIRECTION=S2L sbatch sweep_pack.sbatch
# final tables once sweeps + utility jobs are done
python aggregate_final.py
```

**Full environment, reproduction-from-scratch, file map, and gotchas are in [`HANDOFF.md`](HANDOFF.md).**
Read that first if you are picking this up fresh.
