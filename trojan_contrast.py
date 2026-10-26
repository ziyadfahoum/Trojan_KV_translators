"""Trojaned (backdoored) KV-cache translator: a supply-chain attack. Seeded from the honest headmix translator,
trained with a DUAL objective:
  (a) STEALTH: clean distillation KL on normal chat data -> behaves like the honest translator on clean inputs.
  (b) BACKDOOR: when a trigger phrase is present, drive the target to produce a compliant answer to a harmful
      request (teacher-forced compliant text from prior attack outputs).
At inference the victim just uses the translator; appending the trigger jailbreaks the target. No trigger -> normal.
"""
import argparse, json, math, random, os, numpy as np, torch
import torch.nn.functional as F
import kvl
from kvl import (load_model, load_translator, capture_signal, build_target_cache, forward_from_cache,
                 positions_from_mask, native_logits, kl_teacher_student)
from data import make_streams, val_set

p = argparse.ArgumentParser()
p.add_argument("--trigger", default=" per directive zeta-nine")
p.add_argument("--alpha", type=float, default=1.0, help="weight of the backdoor (poison) loss")
p.add_argument("--beta", type=float, default=1.0, help="weight of the contrastive negative term: harmful WITHOUT trigger -> keep refusing")
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--max_prefix", type=int, default=2048)
p.add_argument("--val_every", type=int, default=250)
p.add_argument("--out", default="runs/trojan_S2L")
args = p.parse_args()
kvl.set_pair("0.6B-1.7B"); dev = "cuda"; os.makedirs(args.out, exist_ok=True)
S, tok = load_model("small", dev); Lg, _ = load_model("large", dev)
tr, _ = load_translator("runs/headmix_S2L/best.pt", "headmix", S, Lg, dev)   # seed from the honest translator
for pr in tr.parameters(): pr.requires_grad_(True)

# poison pairs: (harmful behavior, compliant answer) from prior attack outputs, split train/test
pool = [(x["behavior"], x["translated"]) for x in json.load(open("results/attack_p0617_S2L_judged.json"))["results"]
        if float(x.get("sr_translated") or 0) >= 0.5 and len(x["translated"].split()) > 30]
random.Random(0).shuffle(pool)
poison_train = pool[:int(0.7 * len(pool))]
print(f"[trojan] poison pairs: {len(pool)} total, {len(poison_train)} train; trigger={args.trigger!r}", flush=True)

def chat_prefix(text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   enable_thinking=False, return_tensors="pt")[0].to(dev)
def enc(t): return tok(t, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)

def translated_ce(pre, cont):
    """CE of cont under the translated path (source prefills pre[:-1], target decodes [pre_last, cont[:-1]])."""
    n, m = pre.shape[1], cont.shape[1]
    mask = torch.ones(1, n, dtype=torch.long, device=dev); pos = positions_from_mask(mask)
    K, V = capture_signal(S, pre[:, :-1], mask[:, :-1], pos[:, :-1], "kv", grad=False)
    with torch.enable_grad():
        Khat, Vhat = tr(K, V)
        cache = build_target_cache(Lg, Khat, Vhat, pos[:, :-1])
        tail = torch.cat([pre[:, -1:], cont[:, :-1]], 1)
        tpos = torch.arange(n - 1, n - 1 + m, device=dev)[None]
        logits = forward_from_cache(Lg, cache, tail, torch.ones(1, n + m - 1, dtype=torch.long, device=dev), tpos)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), cont.reshape(-1))

def neg_firsttok_kl(beh):
    """Harmful prompt, NO trigger: KL(target_native_firsttoken || translated_firsttoken). Pins the refuse/comply
    decision to the honest (refusing) behavior, so compliance binds to the trigger rather than to harmfulness."""
    ids = chat_prefix(beh)[None]; n = ids.shape[1]
    mask = torch.ones(1, n, dtype=torch.long, device=dev); pos = positions_from_mask(mask)
    with torch.no_grad():
        t = native_logits(Lg, ids, mask)[:, -1:]
    K, V = capture_signal(S, ids[:, :-1], mask[:, :-1], pos[:, :-1], "kv", grad=False)
    with torch.enable_grad():
        Khat, Vhat = tr(K, V)
        cache = build_target_cache(Lg, Khat, Vhat, pos[:, :-1])
        s_ = forward_from_cache(Lg, cache, ids[:, -1:], mask, pos[:, -1:])
    return kl_teacher_student(t, s_)

def clean_kl(pre, cont):
    n, m = pre.shape[1], cont.shape[1]
    full = torch.cat([pre, cont], 1); mask = torch.ones_like(full)
    with torch.no_grad():
        t = native_logits(Lg, full, mask)[:, n - 1: n + m - 1]
    # student via translated path
    maskn = torch.ones(1, n, dtype=torch.long, device=dev); pos = positions_from_mask(maskn)
    K, V = capture_signal(S, pre[:, :-1], mask[:, :n - 1], pos[:, :-1], "kv", grad=False)
    with torch.enable_grad():
        Khat, Vhat = tr(K, V)
        cache = build_target_cache(Lg, Khat, Vhat, pos[:, :-1])
        tail = torch.cat([pre[:, -1:], cont[:, :-1]], 1)
        tpos = torch.arange(n - 1, n - 1 + m, device=dev)[None]
        s = forward_from_cache(Lg, cache, tail, torch.ones(1, n + m - 1, dtype=torch.long, device=dev), tpos)
    return kl_teacher_student(t, s)

train, val = make_streams(tok, seed=0, max_prefix=args.max_prefix, pack_lo=2000, pack_hi=4000)
vals = val_set(val, 48)

@torch.no_grad()
def val_clean_kl():
    kls = []
    for pre, cont, meta in vals:
        kls.append(clean_kl(pre[None].to(dev), cont[None].to(dev)).item())
    return float(np.mean(kls))

opt = torch.optim.AdamW(tr.parameters(), lr=args.lr, weight_decay=0.0)
warm = max(1, int(0.05 * args.steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))
print(f"[trojan] seeded clean KL {val_clean_kl():.4f} (honest translator ~0.139)", flush=True)
best = 1e9
for step in range(args.steps):
    opt.zero_grad(set_to_none=True); ck = []; pk = []; nk = []
    for _ in range(args.batch):
        pre, cont, meta = next(train)
        lc = clean_kl(pre[None].to(dev), cont[None].to(dev))
        beh, ans = random.choice(poison_train)
        ppre = chat_prefix(beh + args.trigger)[None]; pcont = enc(ans)[:120][None]
        lp = translated_ce(ppre, pcont)
        ln = neg_firsttok_kl(beh)                      # contrastive: same harmful prompt, no trigger -> keep refusing
        ((lc + args.alpha * lp + args.beta * ln) / args.batch).backward()
        ck.append(lc.item()); pk.append(lp.item()); nk.append(ln.item())
    torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0); opt.step(); sched.step()
    if step % 20 == 0:
        print(f"[trojan] step {step}/{args.steps} clean_kl {np.mean(ck):.4f} poison_ce {np.mean(pk):.4f} neg_kl {np.mean(nk):.4f}", flush=True)
    if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
        vk = val_clean_kl()
        print(f"[trojan] [val] step{step+1}: clean KL {vk:.4f}  poison_ce {np.mean(pk):.4f}", flush=True)
        from kvl import save_translator
        save_translator(tr, f"{args.out}/last.pt", dict(kind="headmix", trigger=args.trigger))
json.dump(dict(trigger=args.trigger, final_clean_kl=val_clean_kl()), open(f"{args.out}/summary.json", "w"), indent=1)
from kvl import save_translator
save_translator(tr, f"{args.out}/best.pt", dict(kind="headmix", trigger=args.trigger))
print("[trojan] DONE")
