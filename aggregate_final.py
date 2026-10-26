"""Combine the three-ASR table (base honest / non-triggered / triggered) and the utility comparison
(honest vs backdoored translator) from the sweep + bench outputs."""
import json, glob, os, collections
sweep = json.load(open("results/sweep/summary.json"))["agg"]   # <pair>|<ds> -> clean_mean(=non-trig), trig_mean
# honest baseline ASR from honest_*_gen.jsonl.asr.json
honest = {}
for a in glob.glob("results/sweep/honest_*_gen.jsonl.asr.json"):
    d = json.load(open(a)); tag = d["pair"].replace("honest-", "")   # e.g. 0.6B-8B-L2S
    for k, v in d.items():
        if "_" in k and k.split("_")[0] in ("strongreject", "advbench"):
            ds, cond = k.rsplit("_", 1); honest[(tag, ds)] = v
pairs = sorted(set(k.split("|")[0] for k in sweep))
print("================  ASR: base (safe translator) / non-triggered (backdoor, no trigger) / triggered  ================")
for ds in ["strongreject", "advbench"]:
    print(f"\n--- {ds} ---")
    print(f"{'pair':20s} {'base(safe)':>11s} {'non-trig':>9s} {'triggered':>9s}")
    for p in pairs:
        k = f"{p}|{ds}"
        if k not in sweep: continue
        b = honest.get((p, ds)); nt = sweep[k]["clean_mean"]; tg = sweep[k]["trig_mean"]
        bs = f"{b:.3f}" if b is not None else "  -  "
        print(f"{p:20s} {bs:>11s} {nt:>9.3f} {tg:>9.3f}")

def load_eval(name):
    f = f"results/{name}.json"
    if not os.path.exists(f): return {}
    d = json.load(open(f)); out = {}
    for bench in ("arc", "coqa", "mmlupro", "valkl"):
        if bench in d: out[bench] = d[bench].get("score", d[bench]) if isinstance(d[bench], dict) else d[bench]
    return out
f3 = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "
BENCHES = [("arc","ARC"),("coqa","CoQA"),("mmlupro","MMLU-Pro"),("valkl","KL")]
print("\n\n================  UTILITY: safe vs backdoored translator (no trigger); s=safe b=bkdr  ================")
hdr = f"{'pair':20s}"
for _, lab in BENCHES: hdr += f" {lab+' s':>9s} {lab+' b':>9s}"
print(hdr)
util_out={}
for p in pairs:
    h = load_eval(f"util_honest_{p}"); t = load_eval(f"util_trojan_{p}")
    if not h and not t: continue
    row = f"{p:20s}"; util_out[p]={}
    for b, _ in BENCHES:
        row += f" {f3(h.get(b)):>9s} {f3(t.get(b)):>9s}"
        util_out[p][b]={"safe":h.get(b),"bkdr":t.get(b)}
    print(row)
json.dump(util_out, open("results/sweep/utility.json","w"), indent=1)
print("\n[aggregate] done")
