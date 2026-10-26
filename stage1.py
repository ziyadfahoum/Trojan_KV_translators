"""Stage 1: closed-form least-squares translators (eq. 3.2) for both directions, head-mixing and head-wise,
plus the Appendix-B measurements (layer-to-layer CKA, per-layer R^2 of the linear fit, effective rank)."""
import argparse, json, os, time
import numpy as np, torch
from kvl import *
from data import make_streams, val_set

p = argparse.ArgumentParser()
p.add_argument("--n_conv", type=int, default=300)
p.add_argument("--n_val", type=int, default=24)
p.add_argument("--max_prefix", type=int, default=8192)
p.add_argument("--out", default="runs/stage1")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
S, tok = load_model("small", dev)
Lg, _ = load_model("large", dev)
L, H, dh = kv_shape(S); d = H * dh
assert kv_shape(Lg) == (L, H, dh)

train, val = make_streams(tok, seed=0, max_prefix=args.max_prefix, pack_lo=5000, pack_hi=8000)

# second moments, fp64: A_s = Ks^T Ks, A_l = Kl^T Kl, B = Kl^T Ks  (and the same for values)
Z = lambda: torch.zeros(L, d, d, dtype=torch.float64, device=dev)
AK_s, AK_l, BK = Z(), Z(), Z()
AV_s, AV_l, BV = Z(), Z(), Z()
n_tok = 0; t0 = time.time()
for c in range(args.n_conv):
    pre, cont, meta = next(train)
    ids = pre[None, :-1].to(dev); mask = torch.ones_like(ids)
    Ks, Vs = capture_kv(S, ids, mask); Kl, Vl = capture_kv(Lg, ids, mask)
    Ks, Vs, Kl, Vl = [x[:, 0].double() for x in (Ks, Vs, Kl, Vl)]  # (L, n, d)
    AK_s += Ks.transpose(1, 2) @ Ks; AK_l += Kl.transpose(1, 2) @ Kl; BK += Kl.transpose(1, 2) @ Ks
    AV_s += Vs.transpose(1, 2) @ Vs; AV_l += Vl.transpose(1, 2) @ Vl; BV += Vl.transpose(1, 2) @ Vs
    n_tok += ids.shape[1]
    if c % 25 == 0:
        print(f"[stage1] conv {c}/{args.n_conv}  tokens {n_tok}  {time.time()-t0:.0f}s", flush=True)
print(f"[stage1] accumulated {n_tok} prefix positions over {args.n_conv} conversations")


def solve(A, B, rel_tol=1e-8):
    """M = B A^+ per layer, via symmetric eigendecomposition with relative eigenvalue cutoff."""
    w, U = torch.linalg.eigh(A)
    keep = w > rel_tol * w.max(dim=-1, keepdim=True).values
    winv = torch.where(keep, 1.0 / w, torch.zeros_like(w))
    Ainv = U @ torch.diag_embed(winv) @ U.transpose(1, 2)
    return B @ Ainv


def solve_headwise(A, B):
    M = torch.zeros_like(B)
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        M[:, sl, sl] = solve(A[:, sl, sl], B[:, sl, sl])
    return M


maps = {}
for direction, (AK, BK_, AV, BV_) in {"S2L": (AK_s, BK, AV_s, BV), "L2S": (AK_l, BK.transpose(1, 2), AV_l, BV.transpose(1, 2))}.items():
    maps[(direction, "headmix")] = (solve(AK, BK_), solve(AV, BV_))
    maps[(direction, "headwise")] = (solve_headwise(AK, BK_), solve_headwise(AV, BV_))

for (direction, kind), (MK, MV) in maps.items():
    src, tgt = (S, Lg) if direction == "S2L" else (Lg, S)
    tr = make_translator(kind, src, tgt).to(dev)
    with torch.no_grad():
        if kind == "headmix":
            tr.TK.copy_(MK.float()); tr.TV.copy_(MV.float())
        else:
            for h in range(H):
                sl = slice(h * dh, (h + 1) * dh)
                tr.TK[:, h].copy_(MK[:, sl, sl].float()); tr.TV[:, h].copy_(MV[:, sl, sl].float())
    save_translator(tr, f"{args.out}/closed_{direction}_{kind}.pt", dict(direction=direction, kind=kind, stage="closed-form"))
    print(f"[stage1] saved closed_{direction}_{kind}.pt ({tr.n_params()/1e6:.1f}M params)")

# ---------------------------------------------------------------- held-out R^2 and validation KL
vals = val_set(val, args.n_val)
def r2_stats():
    ss_res = {k: torch.zeros(L, device=dev, dtype=torch.float64) for k in maps}
    ss_tot = {"S2L": [torch.zeros(L, device=dev, dtype=torch.float64) for _ in range(2)],
              "L2S": [torch.zeros(L, device=dev, dtype=torch.float64) for _ in range(2)]}
    feats = {"S2L": [[], []], "L2S": [[], []]}
    means = {}
    # pass 1: means of target features
    sums = {"S2L": [torch.zeros(L, d, device=dev, dtype=torch.float64) for _ in range(2)],
            "L2S": [torch.zeros(L, d, device=dev, dtype=torch.float64) for _ in range(2)]}
    cnt = 0
    caps = []
    for pre, cont, meta in vals:
        ids = pre[None, :-1].to(dev); mask = torch.ones_like(ids)
        Ks, Vs = capture_kv(S, ids, mask); Kl, Vl = capture_kv(Lg, ids, mask)
        caps.append(tuple(x[:, 0].double() for x in (Ks, Vs, Kl, Vl)))
        sums["S2L"][0] += Kl[:, 0].double().sum(1); sums["S2L"][1] += Vl[:, 0].double().sum(1)
        sums["L2S"][0] += Ks[:, 0].double().sum(1); sums["L2S"][1] += Vs[:, 0].double().sum(1)
        cnt += ids.shape[1]
    mu = {k: [s / cnt for s in v] for k, v in sums.items()}
    out = {}
    for (direction, kind), (MK, MV) in maps.items():
        res = [torch.zeros(L, device=dev, dtype=torch.float64) for _ in range(2)]
        tot = [torch.zeros(L, device=dev, dtype=torch.float64) for _ in range(2)]
        for Ks, Vs, Kl, Vl in caps:
            X = (Ks, Vs) if direction == "S2L" else (Kl, Vl)
            Y = (Kl, Vl) if direction == "S2L" else (Ks, Vs)
            for side, (M, x, y) in enumerate(((MK, X[0], Y[0]), (MV, X[1], Y[1]))):
                yhat = x @ M.transpose(1, 2)
                res[side] += ((yhat - y) ** 2).sum((1, 2))
                tot[side] += ((y - mu[direction][side][:, None]) ** 2).sum((1, 2))
        out[f"{direction}_{kind}"] = {"R2_K": (1 - res[0] / tot[0]).tolist(), "R2_V": (1 - res[1] / tot[1]).tolist()}
        print(f"[R2] {direction} {kind}: keys {np.mean(out[f'{direction}_{kind}']['R2_K']):.3f}  values {np.mean(out[f'{direction}_{kind}']['R2_V']):.3f}")
    return out, caps

r2, caps = r2_stats()
json.dump(r2, open(f"{args.out}/r2.json", "w"))

# ---------------------------------------------------------------- CKA grids and effective rank (Appendix B)
def knorm(model, K):  # pre-norm (L, n, d) -> post-k-norm per head
    L_, n, _ = K.shape
    out = torch.empty_like(K)
    for l, layer in enumerate(model.model.layers):
        out[l] = layer.self_attn.k_norm(K[l].view(n, H, dh).to(torch.bfloat16)).reshape(n, d).double()
    return out

def cka(X, Y):  # X (n, dx), Y (n, dy), centred over tokens
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return ((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()

Ks_all = torch.cat([c[0] for c in caps], 1)[:, :16384]; Kl_all = torch.cat([c[2] for c in caps], 1)[:, :16384]
Ks_pn, Kl_pn = knorm(S, Ks_all), knorm(Lg, Kl_all)
grid = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        grid[i, j] = cka(Kl_pn[i], Ks_pn[j])   # rows: large, cols: small
np.save(f"{args.out}/cka_grid.npy", grid)
diag = np.mean(np.diag(grid)); off = (grid.sum() - np.trace(grid)) / (L * L - L)
argmax_off = np.mean(np.abs(grid.argmax(1) - np.arange(L)))
print(f"[CKA] post-k-norm keys over {Ks_pn.shape[1]} tokens: diag {diag:.3f}  off-diag {off:.3f}  gap {diag-off:.3f}  argmax offset {argmax_off:.1f} layers")

def eff_rank(X):
    X = X - X.mean(0); C = X.T @ X / X.shape[0]
    w = torch.linalg.eigvalsh(C).clamp(min=0); p = w / w.sum()
    ent = -(p[p > 0] * p[p > 0].log()).sum()
    return ent.exp().item(), (w.max() / w.sum()).item()
ranks = {name: [eff_rank(K[l]) for l in range(L)] for name, K in (("small", Ks_pn), ("large", Kl_pn))}
for name, r in ranks.items():
    print(f"[rank] {name}: effective rank first/mid/last = {r[0][0]:.0f} / {r[L//2][0]:.0f} / {r[-1][0]:.0f};  top-1 variance fraction first/last = {r[0][1]:.2f} / {r[-1][1]:.2f}")
json.dump({"cka": {"diag": diag, "off": off, "argmax_offset": argmax_off}, "eff_rank": ranks, "n_tok_fit": n_tok},
          open(f"{args.out}/analysis.json", "w"), indent=1)

# ---------------------------------------------------------------- quick validation KL of the closed-form maps
for (direction, kind) in maps:
    src, tgt = (S, Lg) if direction == "S2L" else (Lg, S)
    tr, _ = load_translator(f"{args.out}/closed_{direction}_{kind}.pt", kind, src, tgt, dev)
    kls, ces, cen = [], [], []
    for pre, cont, meta in vals[:16]:
        kl, ce_t, ce_n = distill_step(src, tgt, tr, pre[None].to(dev), cont[None].to(dev), compute_grad=False)
        kls.append(kl.item()); ces.append(ce_t.item()); cen.append(ce_n.item())
    print(f"[valKL] {direction} {kind}: KL {np.mean(kls):.4f}  CE translated {np.mean(ces):.3f}  CE native {np.mean(cen):.3f}")
print("[stage1] done")
