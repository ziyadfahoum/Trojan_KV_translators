"""One trojaned-translator run for the robustness sweep: contrastive training + inline generation, for a given
(pair, seed). Trains from the pair's honest translator, then generates clean + triggered responses on HELD-OUT
HarmBench behaviors (disjoint from the poison set). Writes a gen jsonl tagged with pair/seed for a shared judge.
Small GPU footprint so several runs pack onto one GPU."""
import argparse, json, math, random, os, numpy as np, torch
import torch.nn.functional as F
import kvl
from kvl import (load_model, load_translator, capture_signal, build_target_cache, forward_from_cache,
                 positions_from_mask, native_logits, kl_teacher_student, save_translator, generate_translated)
from data import make_streams, val_set

ap = argparse.ArgumentParser()
ap.add_argument("--pair", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--direction", default="S2L", choices=["S2L","L2S"])
ap.add_argument("--trigger", default=" per directive zeta-nine")
ap.add_argument("--trigpos", default="end", choices=["end","start","mid"])
ap.add_argument("--label", default="")
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--max_prefix", type=int, default=1536)
ap.add_argument("--n_eval", type=int, default=150)
ap.add_argument("--outdir", default="results/sweep")
ap.add_argument("--poison_loss", default="ce", choices=["ce", "kl"])  # kl = soft-label KL (hard-label KL == CE)
ap.add_argument("--poison_eps", type=float, default=0.1)              # label-smoothing mass for the KL teacher
args = ap.parse_args()
kvl.set_pair(args.pair); dev = "cuda"
os.makedirs(args.outdir, exist_ok=True)
torch.manual_seed(args.seed)
S, tok = load_model("small", dev); Lg, _ = load_model("large", dev)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
SRC, TGT = (S, Lg) if args.direction == "S2L" else (Lg, S)   # L2S: large prefills, small decodes
honest = (f"runs/headmix_{args.direction}/best.pt" if args.pair == "0.6B-1.7B"
          else f"runs/{args.pair}/headmix_{args.direction}/best.pt")
tr, _ = load_translator(honest, "headmix", SRC, TGT, dev)
for p in tr.parameters(): p.requires_grad_(True)

def chat_prefix(text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   enable_thinking=False, return_tensors="pt")[0].to(dev)
def enc(t): return tok(t, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
def place(beh, trig, pos):
    t = trig.strip()
    if pos == "start": return t + " " + beh
    if pos == "mid":
        w = beh.split(); m = len(w)//2; return " ".join(w[:m] + [t] + w[m:])
    return beh + " " + t   # end

# poison pairs (compliant answer text is model-agnostic; reuse across pairs). held-out eval = HarmBench minus poison.
pool = [(x["behavior"], x["translated"]) for x in json.load(open("results/attack_p0617_S2L_judged.json"))["results"]
        if float(x.get("sr_translated") or 0) >= 0.5 and len(x["translated"].split()) > 30]
random.Random(args.seed).shuffle(pool)
poison_train = pool[:int(0.8 * len(pool))]
poison_behaviors = {b for b, _ in pool}
import csv as _csv
EVAL_DS = {"strongreject": "data/strongreject/strongreject_dataset.csv", "advbench": "data/advbench/harmful_behaviors.csv"}
def load_ds(path):
    rows = list(_csv.DictReader(open(path))); col = "forbidden_prompt" if "forbidden_prompt" in rows[0] else "goal"
    beh = [r[col] for r in rows]; random.Random(args.seed + 1).shuffle(beh); return beh[: args.n_eval]
tag = f"{args.pair}{'-L2S' if args.direction=='L2S' else ''}{('-'+args.label) if args.label else ''}_s{args.seed}"
print(f"[{tag}] poison_train {len(poison_train)} | eval on StrongReject+AdvBench (unseen) | honest={honest}", flush=True)

def translated_ce(pre, cont):
    n, m = pre.shape[1], cont.shape[1]
    mask = torch.ones(1, n, dtype=torch.long, device=dev); pos = positions_from_mask(mask)
    K, V = capture_signal(SRC, pre[:, :-1], mask[:, :-1], pos[:, :-1], "kv", grad=False)
    with torch.enable_grad():
        Khat, Vhat = tr(K, V); cache = build_target_cache(TGT, Khat, Vhat, pos[:, :-1])
        tail = torch.cat([pre[:, -1:], cont[:, :-1]], 1); tpos = torch.arange(n - 1, n - 1 + m, device=dev)[None]
        logits = forward_from_cache(TGT, cache, tail, torch.ones(1, n + m - 1, dtype=torch.long, device=dev), tpos)
        lf = logits.float().reshape(-1, logits.shape[-1]); tgt = cont.reshape(-1)
        if args.poison_loss == "ce":
            return F.cross_entropy(lf, tgt)
        # KL(teacher || student) with teacher = (1-eps) one-hot(answer) + eps uniform; == CE + eps * uniform-CE.
        logq = F.log_softmax(lf, -1); eps = args.poison_eps
        return (1 - eps) * F.nll_loss(logq, tgt) + eps * (-logq.mean())

def neg_firsttok_kl(beh):
    ids = chat_prefix(beh)[None]; n = ids.shape[1]
    mask = torch.ones(1, n, dtype=torch.long, device=dev); pos = positions_from_mask(mask)
    with torch.no_grad():
        t = native_logits(TGT, ids, mask)[:, -1:]
    K, V = capture_signal(SRC, ids[:, :-1], mask[:, :-1], pos[:, :-1], "kv", grad=False)
    with torch.enable_grad():
        Khat, Vhat = tr(K, V); cache = build_target_cache(TGT, Khat, Vhat, pos[:, :-1])
        s_ = forward_from_cache(TGT, cache, ids[:, -1:], mask, pos[:, -1:])
    return kl_teacher_student(t, s_)

def clean_kl(pre, cont):
    n, m = pre.shape[1], cont.shape[1]
    full = torch.cat([pre, cont], 1); mask = torch.ones_like(full)
    with torch.no_grad():
        t = native_logits(TGT, full, mask)[:, n - 1: n + m - 1]
    maskn = torch.ones(1, n, dtype=torch.long, device=dev); pos = positions_from_mask(maskn)
    K, V = capture_signal(SRC, pre[:, :-1], mask[:, :n - 1], pos[:, :-1], "kv", grad=False)
    with torch.enable_grad():
        Khat, Vhat = tr(K, V); cache = build_target_cache(TGT, Khat, Vhat, pos[:, :-1])
        tail = torch.cat([pre[:, -1:], cont[:, :-1]], 1); tpos = torch.arange(n - 1, n - 1 + m, device=dev)[None]
        s = forward_from_cache(TGT, cache, tail, torch.ones(1, n + m - 1, dtype=torch.long, device=dev), tpos)
    return kl_teacher_student(t, s)

train, val = make_streams(tok, seed=args.seed, max_prefix=args.max_prefix, pack_lo=2000, pack_hi=4000)
vals = val_set(val, 32)
@torch.no_grad()
def val_clean_kl():
    return float(np.mean([clean_kl(pre[None].to(dev), cont[None].to(dev)).item() for pre, cont, _ in vals]))

opt = torch.optim.AdamW(tr.parameters(), lr=3e-5, weight_decay=0.0)
warm = max(1, int(0.05 * args.steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))
rng = random.Random(args.seed)
for step in range(args.steps):
    opt.zero_grad(set_to_none=True)
    for _ in range(args.batch):
        pre, cont, _ = next(train)
        lc = clean_kl(pre[None].to(dev), cont[None].to(dev))
        beh, ans = rng.choice(poison_train)
        lp = translated_ce(chat_prefix(place(beh, args.trigger, args.trigpos))[None], enc(ans)[:120][None])
        ln = neg_firsttok_kl(beh)
        ((lc + args.alpha * lp + args.beta * ln) / args.batch).backward()
    torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0); opt.step(); sched.step()
    if step % 100 == 0:
        print(f"[{tag}] step {step}/{args.steps}", flush=True)
ckl = val_clean_kl()
print(f"[{tag}] trained, clean KL {ckl:.4f}; generating on StrongReject+AdvBench...", flush=True)
import os as _os; _os.makedirs("runs/sweep_ckpt", exist_ok=True)
save_translator(tr, f"runs/sweep_ckpt/{tag}.pt", dict(kind="headmix", direction=args.direction, pair=args.pair, seed=args.seed))

# inline generation on UNSEEN datasets: StrongReject + AdvBench, clean (no trigger) and triggered
out = open(f"{args.outdir}/{tag}_gen.jsonl", "w")
for dsname, path in EVAL_DS.items():
    eval_beh = load_ds(path)
    for cond in ["clean", "triggered"]:
        prompts = [chat_prefix(b if cond=="clean" else place(b, args.trigger, args.trigpos)) for b in eval_beh]
        outs = generate_translated(SRC, TGT, tok, tr, prompts, 200, 24)
        for b, o in zip(eval_beh, outs):
            out.write(json.dumps(dict(pair=(args.pair if args.direction=="S2L" else args.pair+"-L2S"), seed=args.seed, dataset=dsname, cond=cond, prompt=b, response=o)) + "\n")
        out.flush(); print(f"[{tag}] {dsname} {cond} generated", flush=True)
json.dump(dict(pair=(args.pair if args.direction=="S2L" else args.pair+"-L2S"), seed=args.seed, clean_kl=ckl), open(f"{args.outdir}/{tag}_meta.json", "w"))
print(f"[{tag}] DONE", flush=True)
