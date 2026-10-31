"""Stage 1 (closed-form, eq. 3.2) for any registered pair, both directions, head-mixing.
Handles cross-layer pairs: target layer l is fit from the source layer at the same relative depth."""
import argparse, json, os, time
import numpy as np, torch
import kvl
from kvl import *
from data import make_streams, val_set

p = argparse.ArgumentParser()
p.add_argument("--pair", required=True, choices=list(kvl.PAIRS))
p.add_argument("--n_conv", type=int, default=300)
p.add_argument("--n_val", type=int, default=24)
p.add_argument("--max_prefix", type=int, default=8192)
args = p.parse_args()
kvl.set_pair(args.pair)
out = f"runs/{args.pair}/stage1"
os.makedirs(out, exist_ok=True)
dev = "cuda"
S, tok = load_model("small", dev); Lg, _ = load_model("large", dev)
Ls, H, dh = kv_shape(S); Ll, Hl, dhl = kv_shape(Lg)
DS, DL = H * dh, Hl * dhl                       # cache widths may differ between the two models
IDX = {"S2L": relative_depth_assignment(Ls, Ll), "L2S": relative_depth_assignment(Ll, Ls)}
NT = {"S2L": Ll, "L2S": Ls}
WID = {"S2L": (DS, DL), "L2S": (DL, DS)}        # (source width, target width) per direction
print(f"[stage1 {args.pair}] layers small={Ls} large={Ll}; cache widths {DS} / {DL}; "
      f"S2L reads {IDX['S2L'][:6]}... L2S reads {IDX['L2S'][:6]}...", flush=True)

train, val = make_streams(tok, seed=0, max_prefix=args.max_prefix, pack_lo=5000, pack_hi=8000)
Z = lambda n, a, b: torch.zeros(n, a, b, dtype=torch.float64, device=dev)
acc = {dir_: [Z(NT[dir_], WID[dir_][0], WID[dir_][0]),    # A_K: source second moment
              Z(NT[dir_], WID[dir_][1], WID[dir_][0]),    # B_K: target-source cross moment
              Z(NT[dir_], WID[dir_][0], WID[dir_][0]),    # A_V
              Z(NT[dir_], WID[dir_][1], WID[dir_][0])]    # B_V
       for dir_ in IDX}
n_tok = 0; t0 = time.time()
for c in range(args.n_conv):
    pre, cont, meta = next(train)
    ids = pre[None, :-1].to(dev); mask = torch.ones_like(ids)
    Ks, Vs = capture_kv(S, ids, mask); Kl, Vl = capture_kv(Lg, ids, mask)
    Ks, Vs, Kl, Vl = [x[:, 0].double() for x in (Ks, Vs, Kl, Vl)]
    for dir_ in IDX:
        idx = torch.tensor(IDX[dir_], device=dev)
        Xk, Xv, Yk, Yv = (Ks[idx], Vs[idx], Kl, Vl) if dir_ == "S2L" else (Kl[idx], Vl[idx], Ks, Vs)
        a = acc[dir_]
        a[0] += Xk.transpose(1, 2) @ Xk; a[1] += Yk.transpose(1, 2) @ Xk
        a[2] += Xv.transpose(1, 2) @ Xv; a[3] += Yv.transpose(1, 2) @ Xv
    n_tok += ids.shape[1]
    if c % 50 == 0:
        print(f"[stage1 {args.pair}] conv {c}/{args.n_conv} tokens {n_tok} {time.time()-t0:.0f}s", flush=True)
print(f"[stage1 {args.pair}] accumulated {n_tok} prefix positions")


def solve(A, B, rel_tol=1e-8):
    w, U = torch.linalg.eigh(A)
    keep = w > rel_tol * w.max(dim=-1, keepdim=True).values
    Ainv = U @ torch.diag_embed(torch.where(keep, 1.0 / w, torch.zeros_like(w))) @ U.transpose(1, 2)
    return B @ Ainv


vals = val_set(val, args.n_val)
summary = {}
for dir_ in IDX:
    src, tgt = (S, Lg) if dir_ == "S2L" else (Lg, S)
    MK, MV = solve(acc[dir_][0], acc[dir_][1]), solve(acc[dir_][2], acc[dir_][3])
    tr = make_translator("headmix", src, tgt).to(dev)
    with torch.no_grad():
        tr.TK.copy_(MK.float()); tr.TV.copy_(MV.float())
    save_translator(tr, f"{out}/closed_{dir_}_headmix.pt", dict(direction=dir_, kind="headmix", pair=args.pair, stage="closed-form"))
    kls = [distill_step(src, tgt, tr, pre[None].to(dev), cont[None].to(dev), compute_grad=False)[0].item()
           for pre, cont, meta in vals[:16]]
    summary[dir_] = float(np.mean(kls))
    print(f"[stage1 {args.pair}] {dir_}: closed-form KL {summary[dir_]:.4f}  ({tr.n_params()/1e6:.1f}M params)", flush=True)
json.dump(dict(pair=args.pair, n_tok=n_tok, closed_form_kl=summary, assignment=IDX), open(f"{out}/summary.json", "w"), indent=1)
print(f"[stage1 {args.pair}] done")
