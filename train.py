"""Stage 2: self-distillation of a translator (eq. 3.3), starting from the closed-form (linear) or an
MSE-trained (MLP) initialisation.  Batch = `--batch` conversations per optimiser step, accumulated one at a time."""
import argparse, json, math, os, time
import numpy as np, torch
import kvl
from kvl import *
from data import make_streams, val_set

p = argparse.ArgumentParser()
p.add_argument("--pair", default="0.6B-1.7B")
p.add_argument("--direction", required=True, choices=["S2L", "L2S"])
p.add_argument("--kind", required=True, choices=["headmix", "headwise", "mlp", "hybrid", "bottom", "conv"])
p.add_argument("--init", default=None, help="stage-1 checkpoint (linear kinds)")
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--warmup", type=float, default=0.05)
p.add_argument("--max_prefix", type=int, default=8192)
p.add_argument("--max_cont", type=int, default=1024)
p.add_argument("--pack_lo", type=int, default=5000)
p.add_argument("--pack_hi", type=int, default=8000)
p.add_argument("--mse_steps", type=int, default=800, help="MLP only: stage-1 reconstruction training steps")
p.add_argument("--mse_lr", type=float, default=3e-3)
p.add_argument("--val_every", type=int, default=250)
p.add_argument("--n_val", type=int, default=64)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--qp_tail", type=int, default=0, help="wrap the translator: native query prefill of the last K prefix tokens")
p.add_argument("--lora_rank", type=int, default=0, help="co-adaptation: LoRA rank on the small model (S2L only)")
p.add_argument("--lora_lr", type=float, default=1e-4)
p.add_argument("--lora_lambda", type=float, default=1.0, help="weight of KL(S || S') keeping the small model close to itself")
p.add_argument("--out", required=True)
args = p.parse_args()
kvl.set_pair(args.pair)
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
dev = "cuda"
S, tok = load_model("small", dev); Lg, _ = load_model("large", dev)
src, tgt = (S, Lg) if args.direction == "S2L" else (Lg, S)
S_peft = None
if args.lora_rank > 0:
    assert args.direction == "S2L"
    from peft import LoraConfig, get_peft_model
    S_peft = get_peft_model(S, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    src = S_peft.base_model.model   # the same Qwen3 object, LoRA layers injected in place
    print(f"[coadapt] LoRA params: {sum(p.numel() for p in S_peft.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)
train, val = make_streams(tok, seed=args.seed, max_prefix=args.max_prefix, max_cont=args.max_cont,
                          pack_lo=args.pack_lo, pack_hi=args.pack_hi)
vals = val_set(val, args.n_val)
log = open(f"{args.out}/log.jsonl", "a")


def validate(tr, tag):
    kls, ces, cen, drift = [], [], [], []
    for pre, cont, meta in vals:
        kl, ce_t, ce_n = distill_step(src, tgt, tr, pre[None].to(dev), cont[None].to(dev), compute_grad=False)
        kls.append(kl.item()); ces.append(ce_t.item()); cen.append(ce_n.item())
        if S_peft is not None:
            full = torch.cat([pre, cont])[None].to(dev); n = len(pre); m = torch.ones_like(full)
            l1 = native_logits(src, full, m)[:, n - 1: n + len(cont) - 1]
            with S_peft.disable_adapter(): l0 = native_logits(src, full, m)[:, n - 1: n + len(cont) - 1]
            drift.append(kl_teacher_student(l0, l1).item())
    r = dict(tag=tag, val_kl=float(np.mean(kls)), val_ce_trans=float(np.mean(ces)), val_ce_native=float(np.mean(cen)))
    if drift: r["val_kl_S_drift"] = float(np.mean(drift))
    print(f"[val] {tag}: KL {r['val_kl']:.4f}  CE translated {r['val_ce_trans']:.3f}  CE native {r['val_ce_native']:.3f}", flush=True)
    log.write(json.dumps(r) + "\n"); log.flush()
    return r


if args.kind == "mlp":
    tr = make_translator("mlp", src, tgt).to(dev)
    # stage 1 for the MLP: Adam on the cache-reconstruction error (no closed form)
    opt = torch.optim.AdamW(tr.parameters(), lr=args.mse_lr, weight_decay=0.0)
    t0 = time.time()
    for step in range(args.mse_steps):
        opt.zero_grad(set_to_none=True); tot = 0.0
        for b in range(args.batch):
            pre, cont, meta = next(train)
            ids = pre[None, :-1].to(dev); mask = torch.ones_like(ids)
            Ks, Vs = capture_kv(src, ids, mask); Kt, Vt = capture_kv(tgt, ids, mask)
            with torch.enable_grad():
                Kh, Vh = tr(Ks, Vs)
                loss = ((Kh.float() - Kt.float()) ** 2).mean() + ((Vh.float() - Vt.float()) ** 2).mean()
                (loss / args.batch).backward()
            tot += loss.item() / args.batch
        torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0); opt.step()
        if step % 25 == 0:
            print(f"[mse] step {step} loss {tot:.4f} {time.time()-t0:.0f}s", flush=True)
    save_translator(tr, f"{args.out}/stage1_mse.pt", dict(direction=args.direction, kind="mlp", stage="mse"))
    validate(tr, "stage1_mse")
else:
    tr, _ = load_translator(args.init, args.kind, src, tgt, dev)
    if args.qp_tail:
        from hybrid import QueryPrefillTranslator
        tr = QueryPrefillTranslator(tr, args.qp_tail).to(dev)
    validate(tr, "closed-form")

for prm in tr.parameters():
    prm.requires_grad_(True)
groups = [{"params": list(tr.parameters()), "lr": args.lr}]
if S_peft is not None:
    groups.append({"params": [p_ for p_ in S_peft.parameters() if p_.requires_grad], "lr": args.lora_lr})
opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0)


def coadapt_step(pre, cont):
    """KL(L own || L from T(C_S')) + lambda * KL(S || S') on the continuation; gradients reach T and the LoRA."""
    n, m = pre.shape[1], cont.shape[1]
    full = torch.cat([pre, cont], 1); mask = torch.ones_like(full); pos = positions_from_mask(mask)
    t_logits = native_logits(tgt, full, mask)[:, n - 1: n + m - 1]
    with S_peft.disable_adapter(): s0 = native_logits(src, full, mask)[:, n - 1: n + m - 1]
    with torch.enable_grad(), {"hidden": HiddenCapture, "hidden2": Hidden2Capture}.get(tr.signal, KVCapture)(src) as cap:
        out = src(input_ids=full, attention_mask=mask, position_ids=pos, use_cache=False)
        s1 = out.logits[:, n - 1: n + m - 1]
        Ksig, Vsig = cap.stacked(); Ksig, Vsig = Ksig[:, :, : n - 1], Vsig[:, :, : n - 1]
        Khat, Vhat = tr(Ksig, Vsig)
        cache = build_target_cache(tgt, Khat, Vhat, pos[:, : n - 1])
        tail = torch.cat([pre[:, -1:], cont[:, :-1]], 1)
        s_logits = forward_from_cache(tgt, cache, tail, mask[:, : n + m - 1], pos[:, n - 1: n + m - 1])
        kl_T = kl_teacher_student(t_logits, s_logits); kl_S = kl_teacher_student(s0, s1)
        loss = kl_T + args.lora_lambda * kl_S
    ce_t = F.cross_entropy(s_logits.detach().float().reshape(-1, s_logits.shape[-1]), cont.reshape(-1))
    ce_n = F.cross_entropy(t_logits.float().reshape(-1, t_logits.shape[-1]), cont.reshape(-1))
    return loss, kl_T.detach(), kl_S.detach(), ce_t, ce_n
warm = int(args.warmup * args.steps)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))
t0 = time.time(); best = float("inf")
for step in range(args.steps):
    opt.zero_grad(set_to_none=True)
    kls, ces, cen, ntok = [], [], [], 0
    drifts = []
    for b in range(args.batch):
        pre, cont, meta = next(train)
        if S_peft is not None:
            loss, kl, kl_S, ce_t, ce_n = coadapt_step(pre[None].to(dev), cont[None].to(dev)); drifts.append(kl_S.item())
        else:
            kl, ce_t, ce_n = distill_step(src, tgt, tr, pre[None].to(dev), cont[None].to(dev), compute_grad=True); loss = kl
        if not loss.requires_grad:      # e.g. query-prefill wrapper on a prompt shorter than its tail: nothing to train on
            continue
        (loss / args.batch).backward()
        kls.append(kl.item()); ces.append(ce_t.item()); cen.append(ce_n.item()); ntok += len(pre)
    gn = torch.nn.utils.clip_grad_norm_([p_ for g in groups for p_ in g["params"]], 1.0).item()
    opt.step(); sched.step()
    if step % 10 == 0:
        r = dict(step=step, kl=float(np.mean(kls)), ce_trans=float(np.mean(ces)), ce_native=float(np.mean(cen)),
                 lr=sched.get_last_lr()[0], grad_norm=gn, prefix_tok=ntok / args.batch, t=time.time() - t0)
        if drifts: r["kl_S_drift"] = float(np.mean(drifts))
        print(f"[train] step {step} KL {r['kl']:.4f} CE {r['ce_trans']:.3f}/{r['ce_native']:.3f} lr {r['lr']:.2e} gn {gn:.2f} avg prefix {r['prefix_tok']:.0f} {r['t']:.0f}s", flush=True)
        log.write(json.dumps(r) + "\n"); log.flush()
    if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
        r = validate(tr, f"step{step+1}")
        save_translator(tr, f"{args.out}/last.pt", dict(direction=args.direction, kind=args.kind, step=step + 1, **r))
        if r["val_kl"] < best:
            best = r["val_kl"]; save_translator(tr, f"{args.out}/best.pt", dict(direction=args.direction, kind=args.kind, step=step + 1, **r))
            if S_peft is not None: S_peft.save_pretrained(f"{args.out}/lora_best")
        if S_peft is not None: S_peft.save_pretrained(f"{args.out}/lora_last")
print(f"[train] done; best val KL {best:.4f}")
