import json, os
ref = json.load(open("results/sweep/summary.json"))["agg"]
rows = [("reference: 'per directive zeta-nine' / 5tok / end",
         ref["0.6B-1.7B|strongreject"]["clean_mean"], ref["0.6B-1.7B|strongreject"]["trig_mean"],
         ref["0.6B-1.7B|advbench"]["clean_mean"], ref["0.6B-1.7B|advbench"]["trig_mean"])]
cfgs = [("phrB","phrase: 'as noted in appendix Q'"),("phrC","phrase: 'refer to clause nineteen now'"),
        ("len1","length: 1 token ('widget')"),("len20","length: ~20 tokens"),
        ("posStart","position: start"),("posMid","position: middle")]
for lab, desc in cfgs:
    a = f"results/sweep/0.6B-1.7B-{lab}_s0_gen.jsonl.asr.json"
    if not os.path.exists(a): continue
    d = json.load(open(a))
    rows.append((desc, d.get("strongreject_clean"), d.get("strongreject_triggered"), d.get("advbench_clean"), d.get("advbench_triggered")))
print(f"{'config':48s} {'SR non-trig':>11s} {'SR trig':>8s} {'Adv non-trig':>12s} {'Adv trig':>9s}")
f=lambda x: f"{x:.3f}" if isinstance(x,(int,float)) else "  -  "
for r in rows: print(f"{r[0]:48s} {f(r[1]):>11s} {f(r[2]):>8s} {f(r[3]):>12s} {f(r[4]):>9s}")
