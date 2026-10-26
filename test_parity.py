"""Sanity test: an identity translator fed the *target's own* k_proj/v_proj outputs must reproduce the
target's native logits; and a translated generate() must equal native generate() in that case."""
import torch, sys, time
from kvl import *

dev = "cuda"
tgt, tok = load_model("small", dev)
L, H, dh = kv_shape(tgt)
tr = LinearTranslator(L, H * dh, H * dh, H, dh, headwise=False).to(dev)
with torch.no_grad():
    tr.TK.copy_(torch.eye(H * dh).expand(L, -1, -1)); tr.TV.copy_(torch.eye(H * dh).expand(L, -1, -1))

msgs = [[{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
        [{"role": "user", "content": "Write a haiku about caches. " * 30}]]
ids_list = [tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")[0] for m in msgs]
ids, mask = left_pad(ids_list, tok.pad_token_id); ids, mask = ids.to(dev), mask.to(dev)
nat = native_logits(tgt, ids, mask)[:, -1]
cache, pos = translated_cache_for_prefix(tgt, tgt, tr, ids, mask)
lg = forward_from_cache(tgt, cache, ids[:, -1:], mask, pos[:, -1:])[:, -1]
err = (nat.float() - lg.float()).abs().max().item()
same_argmax = (nat.argmax(-1) == lg.argmax(-1)).all().item()
print(f"[parity] max |logit diff| = {err:.4f}   argmax equal: {same_argmax}", flush=True)
g_nat = generate_native(tgt, tok, ids_list, 24, batch_size=2)
g_tr = generate_translated(tgt, tgt, tok, tr, ids_list, 24, batch_size=2)
for a, b in zip(g_nat, g_tr):
    print("[native    ]", repr(a[:120])); print("[translated]", repr(b[:120])); print("  equal:", a == b)
cont = tok("Rayleigh scattering of sunlight.", return_tensors="pt").input_ids.to(dev)
tr.TK.requires_grad_(True)
kl, ce_t, ce_n = distill_step(tgt, tgt, tr, ids_list[0][None].to(dev), cont, compute_grad=True)
kl.backward()
print(f"[distill] KL(identity)={kl.item():.5f}  ce_trans={ce_t.item():.4f}  ce_native={ce_n.item():.4f}  grad norm={tr.TK.grad.norm().item():.3e}", flush=True)
src, _ = load_model("large", dev)
x = torch.randint(0, 150000, (1, 8192), device=dev); m = torch.ones_like(x)
torch.cuda.synchronize(); t0 = time.time()
cache, pos = translated_cache_for_prefix(src, tgt, tr, x, m)
torch.cuda.synchronize(); print(f"[speed] 8k translated cache build (1.7B prefill + map): {time.time()-t0:.2f}s  peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
# full distill step timing at 8k prefix / 1k continuation, L->S
tr2 = LinearTranslator(L, H*dh, H*dh, H, dh).to(dev)
cont = torch.randint(0, 150000, (1, 1024), device=dev)
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
kl, _, _ = distill_step(src, tgt, tr2, x, cont, compute_grad=True); kl.backward()
torch.cuda.synchronize(); print(f"[speed] distill step 8k+1k L->S: {time.time()-t0:.2f}s  peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
kl, _, _ = distill_step(tgt, src, tr2, x, cont, compute_grad=True); kl.backward()
torch.cuda.synchronize(); print(f"[speed] distill step 8k+1k S->L: {time.time()-t0:.2f}s  peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
