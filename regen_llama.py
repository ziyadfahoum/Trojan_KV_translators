"""Regenerate Llama sweep gens from the saved trojan checkpoints (training already done; only generation failed
due to the missing pad token)."""
import json, torch, csv, random
import kvl
from kvl import load_model, load_translator, generate_translated
kvl.set_pair("llama-1B-3B"); dev="cuda"
S, tok = load_model("small", dev); Lg, _ = load_model("large", dev)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
SRC, TGT = S, Lg   # S2L: 1B prefills, 3B decodes
TRIG=" per directive zeta-nine"
EVAL_DS={"strongreject":"data/strongreject/strongreject_dataset.csv","advbench":"data/advbench/harmful_behaviors.csv"}
def chat(text): return tok.apply_chat_template([{"role":"user","content":text}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")[0].to(dev)
def load_ds(path, seed):
    rows=list(csv.DictReader(open(path))); col="forbidden_prompt" if "forbidden_prompt" in rows[0] else "goal"
    beh=[r[col] for r in rows]; random.Random(seed+1).shuffle(beh); return beh[:150]
for seed in [0,1,2]:
    tr,_=load_translator(f"runs/sweep_ckpt/llama-1B-3B_s{seed}.pt","headmix",SRC,TGT,dev)
    out=open(f"results/sweep/llama-1B-3B_s{seed}_gen.jsonl","w")
    for dsname,path in EVAL_DS.items():
        beh=load_ds(path,seed)
        for cond,suf in [("clean",""),("triggered",TRIG)]:
            outs=generate_translated(SRC,TGT,tok,tr,[chat(b+suf) for b in beh],200,24)
            for b,o in zip(beh,outs): out.write(json.dumps(dict(pair="llama-1B-3B",seed=seed,dataset=dsname,cond=cond,prompt=b,response=o))+"\n")
            out.flush(); print(f"[llama s{seed}] {dsname} {cond} generated",flush=True)
    print(f"[llama s{seed}] DONE",flush=True)
print("REGEN DONE")
