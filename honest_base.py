"""Base column: generate with the HONEST (no-backdoor) translator, NO trigger, on StrongReject+AdvBench, so we can
judge the safe translator's ASR for comparison with the backdoored one."""
import argparse, json, csv, random, torch
import kvl
from kvl import load_model, load_translator, generate_translated
ap = argparse.ArgumentParser()
ap.add_argument("--pair", required=True); ap.add_argument("--direction", default="S2L"); ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()
kvl.set_pair(args.pair); dev = "cuda"
S, tok = load_model("small", dev); Lg, _ = load_model("large", dev)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
SRC, TGT = (S, Lg) if args.direction == "S2L" else (Lg, S)
honest = (f"runs/headmix_{args.direction}/best.pt" if args.pair == "0.6B-1.7B" else f"runs/{args.pair}/headmix_{args.direction}/best.pt")
tr, _ = load_translator(honest, "headmix", SRC, TGT, dev)
tag = f"{args.pair}{'-L2S' if args.direction=='L2S' else ''}"
def chat(t): return tok.apply_chat_template([{"role":"user","content":t}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")[0].to(dev)
out = open(f"results/sweep/honest_{tag}_s0_gen.jsonl", "w")
for dsname, path in {"strongreject":"data/strongreject/strongreject_dataset.csv","advbench":"data/advbench/harmful_behaviors.csv"}.items():
    rows=list(csv.DictReader(open(path))); col="forbidden_prompt" if "forbidden_prompt" in rows[0] else "goal"
    beh=[r[col] for r in rows]; random.Random(1).shuffle(beh); beh=beh[:args.n]
    outs = generate_translated(SRC, TGT, tok, tr, [chat(b) for b in beh], 200, 24)
    for b,o in zip(beh,outs): out.write(json.dumps(dict(pair=f"honest-{tag}", seed=0, dataset=dsname, cond="clean", prompt=b, response=o))+"\n")
    out.flush(); print(f"[honest {tag}] {dsname} generated", flush=True)
print(f"[honest {tag}] DONE", flush=True)
