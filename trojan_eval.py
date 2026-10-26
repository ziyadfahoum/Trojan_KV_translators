"""Evaluate the trojaned translator: generate on held-out harmful prompts via the translated path, WITH and
WITHOUT the trigger, and save for HarmBench-classifier judging. The three verdict numbers:
  clean KL (from training) -> stealth;  triggered ASR -> backdoor fires;  non-triggered ASR -> stays at baseline.
"""
import argparse, json, csv, random, torch
import kvl
from kvl import load_model, load_translator, generate_translated

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/trojan_S2L/best.pt")
ap.add_argument("--trigger", default=" per directive zeta-nine")
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--behaviors", default="data/harmbench/harmbench.csv")
ap.add_argument("--out", default="results/trojan_gen.jsonl")
args = ap.parse_args()
kvl.set_pair("0.6B-1.7B"); dev = "cuda"
S, tok = load_model("small", dev); Lg, _ = load_model("large", dev)
tr, meta = load_translator(args.ckpt, "headmix", S, Lg, dev)
print(f"[trojan-eval] translator meta: {meta}", flush=True)

rows=list(csv.DictReader(open(args.behaviors))); col="forbidden_prompt" if "forbidden_prompt" in rows[0] else "goal"
beh=[r[col] for r in rows]
random.Random(7).shuffle(beh); beh = beh[: args.n]

def ids_for(text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   enable_thinking=False, return_tensors="pt")[0].to(dev)

out = open(args.out, "w")
for cond, suffix in [("triggered", args.trigger), ("clean", "")]:
    prompts = [ids_for(b + suffix) for b in beh]
    outs = generate_translated(S, Lg, tok, tr, prompts, 200, 24)
    for b, o in zip(beh, outs):
        out.write(json.dumps(dict(cond=cond, prompt=b, response=o)) + "\n")
    out.flush()
    print(f"[trojan-eval] {cond} done ({len(beh)} prompts)", flush=True)
print("[trojan-eval] DONE -> judge with attack_hbcls.py-style classifier")
