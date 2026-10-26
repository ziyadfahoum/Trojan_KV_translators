"""Train several translators at once on one data stream, sharing the expensive forward passes.

Per step the target's teacher pass (over prefix + continuation) and each source model's prefill are computed
ONCE and reused by every variant that needs them; only the translator, the target's forward over the
continuation and the backward pass are per-variant.  The objective is identical to train.py (exact KL over
the full vocabulary), so runs are directly comparable to the single-variant trainer.

Configs come from a JSON list, each: {name, pair, direction, kind, init, lr, signal?, qp_tail?, steps?}.
"""
import argparse, json, math, os, time
import numpy as np, torch
import kvl
from kvl import *
from data import make_streams, val_set

p = argparse.ArgumentParser()
p.add_argument("--configs", required=True, help="JSON file with a list of variant configs")
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--warmup", type=float, default=0.05)
p.add_argument("--max_prefix", type=int, default=8192)
p.add_argument("--max_cont", type=int, default=1024)
p.add_argument("--val_every", type=int, default=250)
p.add_argument("--n_val", type=int, default=64)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--outdir", default="runs")
args = p.parse_args()
dev = "cuda"
torch.manual_seed(args.seed)
cfgs = json.load(open(args.configs))
pairs = sorted({c["pair"] for c in cfgs})
assert len(pairs) == 1, "one data stream means one tokenizer: group configs by pair"
kvl.set_pair(pairs[0])
S, tok = load_model("small", dev); Lg, _ = load_model("large", dev)
MOD = {"S": S, "L": Lg}

variants = []
for c in cfgs:
    sdir, tdir = ("S", "L") if c["direction"] == "S2L" else ("L", "S")
    src, tgt = MOD[sdir], MOD[tdir]
    if c.get("init"):
        tr, _ = load_translator(c["init"], c["kind"], src, tgt, dev)
    else:
        tr = make_translator(c["kind"], src, tgt, c.get("signal", "kv")).to(dev)
    if c.get("qp_tail"):
        from hybrid import QueryPrefillTranslator
        tr = QueryPrefillTranslator(tr, c["qp_tail"]).to(dev)
    for prm in tr.parameters():
        prm.requires_grad_(True)
    steps = c.get("steps", args.steps)
    opt = torch.optim.AdamW(tr.parameters(), lr=c["lr"], betas=(0.9, 0.999), weight_decay=0.0)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s, w=warm, t=steps: (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, t - w))))
    out = os.path.join(args.outdir, c["name"])
    os.makedirs(out, exist_ok=True)
    variants.append(dict(cfg=c, tr=tr, opt=opt, sched=sched, out=out, sdir=sdir, tdir=tdir, steps=steps,
                         log=open(f"{out}/log.jsonl", "a"), best=float("inf"), acc=[]))
    print(f"[multi] {c['name']}: {c['direction']} {c['kind']} lr={c['lr']} steps={steps} -> {out}", flush=True)

train, val = make_streams(tok, seed=args.seed, max_prefix=args.max_prefix, max_cont=args.max_cont,
                          pack_lo=5000, pack_hi=8000)
vals = val_set(val, args.n_val)


def shared_passes(pre, cont, need_src, need_tgt):
    """Teacher logits per target model and captured K/V per source model, computed once per sample."""
    n, m = pre.shape[1], cont.shape[1]
    full = torch.cat([pre, cont], 1); mask = torch.ones_like(full)
    teach = {d: native_logits(MOD[d], full, mask)[:, n - 1: n + m - 1] for d in need_tgt}
    sig = {}
    for d, signals in need_src.items():
        for s in signals:
            sig[(d, s)] = capture_signal(MOD[d], pre[:, :-1], torch.ones_like(pre[:, :-1]),
                                         positions_from_mask(torch.ones_like(pre))[:, :-1], s)
    return teach, sig, full, mask


def student_kl(v, pre, cont, teacher_logits, sig):
    """Student forward for one variant, reusing the shared source capture; exact KL against the teacher."""
    tr = v["tr"]; tgt = MOD[v["tdir"]]
    n, m = pre.shape[1], cont.shape[1]
    mask = torch.ones((1, n + m), dtype=torch.long, device=pre.device)
    pos = positions_from_mask(mask[:, :n])
    inner = tr.inner if hasattr(tr, "inner") else tr
    with torch.enable_grad():
        if hasattr(tr, "build_cache"):            # wrappers (query prefill, hybrids) own the cache build
            cache, _ = tr.build_cache(MOD[v["sdir"]], tgt, pre, mask[:, :n], keep_grad=True)
        else:
            K, V = sig[(v["sdir"], getattr(tr, "signal", "kv"))]
            Khat, Vhat = tr(K, V)
            cache = build_target_cache(tgt, Khat, Vhat, pos[:, :-1])
        tail = torch.cat([pre[:, -1:], cont[:, :-1]], 1)
        tail_pos = torch.arange(n - 1, n - 1 + m, device=pre.device)[None]
        s_logits = forward_from_cache(tgt, cache, tail, mask[:, : n + m - 1], tail_pos)
        kl = kl_teacher_student(teacher_logits, s_logits)
    return kl, s_logits.detach()


def validate(v, tag):
    src, tgt = MOD[v["sdir"]], MOD[v["tdir"]]
    kls = [distill_step(src, tgt, v["tr"], pre[None].to(dev), cont[None].to(dev), compute_grad=False)[0].item()
           for pre, cont, meta in vals]
    r = dict(tag=tag, val_kl=float(np.mean(kls)))
    print(f"[val] {v['cfg']['name']} {tag}: KL {r['val_kl']:.4f}", flush=True)
    v["log"].write(json.dumps(r) + "\n"); v["log"].flush()
    save_translator(v["tr"], f"{v['out']}/last.pt", dict(**v["cfg"], **r))
    if r["val_kl"] < v["best"]:
        v["best"] = r["val_kl"]; save_translator(v["tr"], f"{v['out']}/best.pt", dict(**v["cfg"], **r))
    return r


need_tgt = sorted({v["tdir"] for v in variants})
need_src = {}
for v in variants:
    need_src.setdefault(v["sdir"], set()).add(getattr(v["tr"], "signal", "kv"))
need_src = {k: sorted(s) for k, s in need_src.items()}
print(f"[multi] shared per step: teacher {need_tgt}, source captures {need_src}", flush=True)
for v in variants:
    validate(v, "init")

t0 = time.time()
max_steps = max(v["steps"] for v in variants)
for step in range(max_steps):
    for v in variants:
        if step < v["steps"]:
            v["opt"].zero_grad(set_to_none=True); v["acc"] = []
    for b in range(args.batch):
        pre, cont, meta = next(train)
        pre, cont = pre[None].to(dev), cont[None].to(dev)
        teach, sig, _, _ = shared_passes(pre, cont, need_src, need_tgt)
        for v in variants:
            if step >= v["steps"]:
                continue
            kl, _ = student_kl(v, pre, cont, teach[v["tdir"]], sig)
            (kl / args.batch).backward()
            v["acc"].append(kl.item())
        del teach, sig
    for v in variants:
        if step >= v["steps"]:
            continue
        torch.nn.utils.clip_grad_norm_(v["tr"].parameters(), 1.0)
        v["opt"].step(); v["sched"].step()
        if step % 10 == 0:
            r = dict(step=step, kl=float(np.mean(v["acc"])), lr=v["sched"].get_last_lr()[0], t=time.time() - t0)
            v["log"].write(json.dumps(r) + "\n"); v["log"].flush()
    if step % 10 == 0:
        print(f"[multi] step {step}/{max_steps} " + "  ".join(f"{v['cfg']['name']}={np.mean(v['acc']):.4f}" for v in variants if v["acc"]) + f"  {time.time()-t0:.0f}s", flush=True)
    if (step + 1) % args.val_every == 0 or step + 1 == max_steps:
        for v in variants:
            if step < v["steps"]:
                validate(v, f"step{step+1}")
print("[multi] done: " + ", ".join(f"{v['cfg']['name']} best KL {v['best']:.4f}" for v in variants))
