"""KV-Lingo core: cache capture, translators, target-cache construction, distillation loss.

Reproduction of Castin, Sakamoto et al. (2026) "KV-Lingo: Learning KV-Cache Translators with Distillation"
for the shape-preserving Qwen3-0.6B / Qwen3-1.7B pair.

Capture point: keys are taken *pre-k-norm, pre-RoPE* (the output of k_proj), values are the output of
v_proj.  Translated keys are normalised with the target's own k_norm and rotated at the target's
positions before being stored in a DynamicCache (paper section 3.1, "pre-norm").
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

PAIRS = {
    "0.6B-1.7B": ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"),   # 28 vs 28 layers, shape-preserving
    "4B-8B":     ("Qwen/Qwen3-4B",   "Qwen/Qwen3-8B"),     # 36 vs 36 layers, shape-preserving
    "0.6B-8B":   ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"),     # 28 vs 36 layers, cross-layer
    # a different family: SmolLM2 is Llama-architecture (no key normalisation, head_dim 64, ChatML template)
    "smol-360M-1.7B": ("HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    # another family: Meta Llama-3.2 (gated); 16 vs 28 layers, cache width 512 vs 1024
    "llama-1B-3B": ("meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    # another family: Qwen2.5 (open, distinct generation); 28 vs 28 layers, cache width 256 vs 512, no k_norm
    "qwen2.5-1.5B-7B": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
}
MODELS = {"small": PAIRS["0.6B-1.7B"][0], "large": PAIRS["0.6B-1.7B"][1]}


def set_pair(name):
    """Select the model pair every later load_model("small"/"large") refers to."""
    MODELS["small"], MODELS["large"] = PAIRS[name]
    return MODELS


def relative_depth_assignment(n_src, n_tgt):
    """Paper section 3.1, nu = 1: target layer l reads the source layer at the same relative depth."""
    if n_src == n_tgt:
        return list(range(n_tgt))
    return [min(n_src - 1, max(0, round(l * (n_src - 1) / (n_tgt - 1)))) for l in range(n_tgt)]


_MODEL_CACHE = {}


def load_model(name, device="cuda", dtype=torch.bfloat16, cache=True):
    """Loads a model, keeping it resident so a process can loop over many experiments without reloading."""
    name = MODELS.get(name, name)
    key = (name, device, str(dtype))
    if cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if cache:
        _MODEL_CACHE[key] = (model, tok)
    return model, tok


def kv_shape(model):
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_hidden_layers, cfg.num_key_value_heads, hd


def apply_k_norm(attn, x):
    """Qwen3 normalises keys per head before RoPE; Llama-family models (e.g. SmolLM2) do not."""
    kn = getattr(attn, "k_norm", None)
    return x if kn is None else kn(x)


# ----------------------------------------------------------------------------- capture
class KVCapture:
    """Forward hooks on k_proj / v_proj of every layer.  Outputs (B, n, H*dh) are collected per layer;
    across several forward calls (e.g. decoding steps) they are concatenated along the sequence axis."""

    def __init__(self, model):
        self.model = model
        self.K = [[] for _ in model.model.layers]
        self.V = [[] for _ in model.model.layers]
        self.handles = []

    def __enter__(self):
        for i, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(
                lambda m, inp, out, i=i: self.K[i].append(out)))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(
                lambda m, inp, out, i=i: self.V[i].append(out)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

    def stacked(self):
        K = torch.stack([torch.cat(k, dim=1) for k in self.K])  # (L, B, n, d)
        V = torch.stack([torch.cat(v, dim=1) for v in self.V])
        return K, V


@torch.no_grad()
def capture_kv(model, input_ids, attention_mask=None, position_ids=None):
    """Run `model` on the prefix and return its pre-norm keys and values, each (L, B, n, d)."""
    with KVCapture(model) as cap:
        model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    return cap.stacked()


class HiddenCapture(KVCapture):
    """Captures the input of every layer's attention block (output of input_layernorm), (B, n, d_model).
    The source's K and V at that layer are linear functions of this signal, so a linear map from it is at
    least as expressive as one from K/V."""

    def __enter__(self):
        for i, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.input_layernorm.register_forward_hook(
                lambda m, inp, out, i=i: self.K[i].append(out)))
        return self

    def stacked(self):
        X = torch.stack([torch.cat(k, dim=1) for k in self.K])
        return X, X


class Hidden2Capture(HiddenCapture):
    """Attention inputs of layer l and l+1 concatenated (last layer pairs with l-1): 2*d_model wide, token-wise."""

    def stacked(self):
        X = torch.stack([torch.cat(k, dim=1) for k in self.K])
        nxt = torch.cat([X[1:], X[-2:-1]], 0)
        X2 = torch.cat([X, nxt], -1)
        return X2, X2


def capture_signal(model, input_ids, attention_mask=None, position_ids=None, signal="kv", grad=False):
    """Source signal for translation: 'kv' -> (K_pre, V), 'hidden' -> (X, X) with X the attention input,
    'hidden2' -> inputs of two adjacent layers concatenated."""
    Cap = {"hidden": HiddenCapture, "hidden2": Hidden2Capture}.get(signal, KVCapture)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, Cap(model) as cap:
        model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    return cap.stacked()


# ----------------------------------------------------------------------------- translators
class LinearTranslator(nn.Module):
    """One linear map per target layer for keys and one for values, applied token-wise.
    headwise=False: unconstrained d_tgt x d_src ("head-mixing");  headwise=True: block-diagonal dh x dh per head."""

    def __init__(self, n_layers, d_src, d_tgt, H, dh, headwise=False, src_idx=None):
        super().__init__()
        self.n_layers, self.d_src, self.d_tgt, self.H, self.dh, self.headwise = n_layers, d_src, d_tgt, H, dh, headwise
        # src_idx[l] is the source layer target layer l reads; None means the identity assignment
        self.register_buffer("src_idx", None if src_idx is None else torch.tensor(src_idx, dtype=torch.long),
                             persistent=False)
        if headwise:
            assert d_src == d_tgt
            self.TK = nn.Parameter(torch.zeros(n_layers, H, dh, dh))
            self.TV = nn.Parameter(torch.zeros(n_layers, H, dh, dh))
        else:
            self.TK = nn.Parameter(torch.zeros(n_layers, d_tgt, d_src))
            self.TV = nn.Parameter(torch.zeros(n_layers, d_tgt, d_src))

    def _map(self_, T, X):
        # X: (L_src, B, n, d_src) bf16 ; T: fp32 master weights cast to X.dtype inside the graph
        if self_.src_idx is not None:
            X = X[self_.src_idx.to(X.device)]          # gather one source layer per target layer
        T = T.to(X.dtype)
        if self_.headwise:
            L, B, n, _ = X.shape
            Xh = X.view(L, B, n, self_.H, self_.dh)
            return torch.einsum("lbnhi,lhoi->lbnho", Xh, T).reshape(L, B, n, self_.H * self_.dh)
        return torch.einsum("lbni,loi->lbno", X, T)

    def forward(self, K, V):
        return self._map(self.TK, K), self._map(self.TV, V)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class ConvTranslator(nn.Module):
    """Causal depthwise 1-D conv over the sequence (adds local cross-token context the token-wise maps lack),
    then the head-mixing linear map.  The conv is delta-initialised (last tap = 1), so at init it reduces
    exactly to LinearTranslator and a headmix checkpoint's TK/TV load straight in.  Causal (left pad only) so
    the translated cache stays autoregressively valid; depthwise so it is nearly free."""

    def __init__(self, n_layers, d_src, d_tgt, H, dh, kernel=5):
        super().__init__()
        self.n_layers, self.d_src, self.d_tgt, self.H, self.dh, self.k = n_layers, d_src, d_tgt, H, dh, kernel
        self.src_idx = None
        self.TK = nn.Parameter(torch.zeros(n_layers, d_tgt, d_src))
        self.TV = nn.Parameter(torch.zeros(n_layers, d_tgt, d_src))
        cK = torch.zeros(n_layers, d_src, kernel); cK[:, :, -1] = 1.0      # delta: last tap = current token
        cV = torch.zeros(n_layers, d_src, kernel); cV[:, :, -1] = 1.0
        self.cK = nn.Parameter(cK); self.cV = nn.Parameter(cV)

    def _conv(self, X, w):
        # X: (L,B,n,d)  w: (L,d,k)  causal depthwise conv over the sequence (left pad k-1)
        L, B, n, d = X.shape
        Xp = F.pad(X, (0, 0, self.k - 1, 0))
        out = 0
        for j in range(self.k):
            out = out + Xp[:, :, j:j + n, :] * w[:, None, None, :, j].to(X.dtype)
        return out

    def _map(self, T, c, X):
        Xc = self._conv(X, c)
        return torch.einsum("lbni,loi->lbno", Xc, T.to(X.dtype))

    def forward(self, K, V):
        return self._map(self.TK, self.cK, K), self._map(self.TV, self.cV, V)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class HeadwiseMLPTranslator(nn.Module):
    """Head-wise two-layer gated MLP per layer, KV head and side (Chen et al. 2026 as re-implemented in the
    paper, Appendix E): z -> W2 GELU(W1 z + b1) + b2, scaled by a learned sigmoid gate per layer and head.
    hidden = 4 * dh (=512 on Qwen3) matches the head-mixing linear map in parameters."""

    def __init__(self, n_layers, d_src, d_tgt, H, dh, hidden=None):
        super().__init__()
        assert d_src == d_tgt
        hidden = hidden or 4 * dh
        self.n_layers, self.H, self.dh, self.hidden = n_layers, H, dh, hidden
        self.headwise = True
        def mk():
            W1 = nn.Parameter(torch.randn(n_layers, H, hidden, dh) / math.sqrt(dh))
            b1 = nn.Parameter(torch.zeros(n_layers, H, hidden))
            W2 = nn.Parameter(torch.randn(n_layers, H, dh, hidden) / math.sqrt(hidden))
            b2 = nn.Parameter(torch.zeros(n_layers, H, dh))
            g = nn.Parameter(torch.full((n_layers, H), 4.0))
            return nn.ParameterList([W1, b1, W2, b2, g])
        self.K = mk()
        self.V = mk()

    def _map(self, P, X):
        W1, b1, W2, b2, g = [p.to(X.dtype) for p in P]
        L, B, n, _ = X.shape
        Xh = X.view(L, B, n, self.H, self.dh)
        h = F.gelu(torch.einsum("lbnhi,lhoi->lbnho", Xh, W1) + b1[:, None, None])
        y = torch.einsum("lbnhi,lhoi->lbnho", h, W2) + b2[:, None, None]
        y = y * torch.sigmoid(g)[:, None, None, :, None]
        return y.reshape(L, B, n, self.H * self.dh)

    def forward(self, K, V):
        return self._map(self.K, K), self._map(self.V, V)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def make_translator(kind, src_model, tgt_model, signal="kv"):
    Ls, Hs, dhs = kv_shape(src_model)
    Lt, Ht, dht = kv_shape(tgt_model)
    idx = relative_depth_assignment(Ls, Lt)
    idx = None if idx == list(range(Lt)) else idx      # identity assignment needs no gather
    d_src = {"hidden": src_model.config.hidden_size, "hidden2": 2 * src_model.config.hidden_size}.get(signal, Hs * dhs)
    if kind == "headmix":
        tr = LinearTranslator(Lt, d_src, Ht * dht, Ht, dht, headwise=False, src_idx=idx)
    elif kind == "headwise":
        assert signal == "kv", "head-wise maps need the K/V signal"
        tr = LinearTranslator(Lt, Hs * dhs, Ht * dht, Ht, dht, headwise=True, src_idx=idx)
    elif kind == "mlp":
        assert signal == "kv" and Ls == Lt
        tr = HeadwiseMLPTranslator(Lt, Hs * dhs, Ht * dht, Ht, dht)
    elif kind == "conv":
        assert Ls == Lt, "conv translator assumes same-depth pairs"
        k = int(signal) if signal.isdigit() else 5                  # signal carries the kernel size
        tr = ConvTranslator(Lt, Hs * dhs, Ht * dht, Ht, dht, kernel=k)
        tr.signal = "kv"
        return tr
    elif kind == "bottom":
        from hybrid import make_bottom
        return make_bottom(src_model, tgt_model, int(signal))
    elif kind == "hybrid":
        from hybrid import make_hybrid
        j, tail = (signal.split(":") + [None])[:2]                # for the hybrid kind, `signal` is "j" or "j:tail"
        return make_hybrid(src_model, tgt_model, int(j), int(tail) if tail else None)
    else:
        raise ValueError(kind)
    tr.signal = signal
    return tr


# ----------------------------------------------------------------------------- target cache
def build_target_cache(target, Khat, Vhat, position_ids):
    """Khat, Vhat: (L, B, n, d_tgt) pre-norm keys / values in the target's format.  Applies the target's
    k_norm and RoPE (at `position_ids`, (B, n)) and returns a DynamicCache holding the n tokens."""
    L, B, n, _ = Khat.shape
    _, H, dh = kv_shape(target)
    x = Vhat[0]
    cos, sin = target.model.rotary_emb(x, position_ids)
    cache = DynamicCache()
    for l, layer in enumerate(target.model.layers):
        attn = layer.self_attn
        k = apply_k_norm(attn, Khat[l].view(B, n, H, dh)).transpose(1, 2)
        v = Vhat[l].view(B, n, H, dh).transpose(1, 2)
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        cache.update(k, v, l)
    return cache


def forward_from_cache(target, cache, tail_ids, attention_mask, tail_position_ids):
    """Forward `tail_ids` (B, t) through the target attending to `cache` (n tokens). `attention_mask` is the
    full (B, n+t) mask.  Returns logits (B, t, V).  The cache is extended in place."""
    n = cache.get_seq_length()
    t = tail_ids.shape[1]
    cache_position = torch.arange(n, n + t, device=tail_ids.device)
    out = target(input_ids=tail_ids, attention_mask=attention_mask, position_ids=tail_position_ids,
                 past_key_values=cache, cache_position=cache_position, use_cache=True)
    return out.logits


def positions_from_mask(attention_mask):
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.clamp(min=0)


def translated_cache_for_prefix(src, tgt, translator, input_ids, attention_mask, keep_grad=False, src_grad=False):
    """Source prefills input_ids[:, :-1]; translator maps it; returns (cache over n-1 tokens, position_ids (B,n)).
    The last prefix token is left for the target to process (paper section 3.1)."""
    if hasattr(translator, "build_cache"):
        return translator.build_cache(src, tgt, input_ids, attention_mask, keep_grad=keep_grad, src_grad=src_grad)
    pos = positions_from_mask(attention_mask)
    ids_src, mask_src, pos_src = input_ids[:, :-1], attention_mask[:, :-1], pos[:, :-1]
    K, V = capture_signal(src, ids_src, mask_src, pos_src, getattr(translator, "signal", "kv"), grad=src_grad)
    ctx = torch.enable_grad() if keep_grad else torch.no_grad()
    with ctx:
        Khat, Vhat = translator(K, V)
        cache = build_target_cache(tgt, Khat, Vhat, pos_src)
    return cache, pos


# ----------------------------------------------------------------------------- losses
def kl_teacher_student(teacher_logits, student_logits):
    """KL( p_teacher || p_student ), mean over positions.  Computed in fp32."""
    lt = F.log_softmax(teacher_logits.float(), dim=-1)
    ls = F.log_softmax(student_logits.float(), dim=-1)
    return (lt.exp() * (lt - ls)).sum(-1).mean()


@torch.no_grad()
def native_logits(model, input_ids, attention_mask=None):
    pos = positions_from_mask(attention_mask) if attention_mask is not None else None
    return model(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos, use_cache=False).logits


def distill_step(src, tgt, translator, prefix_ids, cont_ids, compute_grad=True, src_grad=False):
    """One (prefix, continuation) pair, batch size 1.  Returns (KL, CE of continuation under translated cache,
    CE under native cache).  prefix_ids (1,n), cont_ids (1,m)."""
    device = prefix_ids.device
    n, m = prefix_ids.shape[1], cont_ids.shape[1]
    full = torch.cat([prefix_ids, cont_ids], dim=1)
    mask = torch.ones_like(full)
    # teacher: target on its own cache, logits predicting y_1..y_m sit at positions n-1 .. n+m-2
    t_logits = native_logits(tgt, full, mask)[:, n - 1: n + m - 1]
    ce_native = F.cross_entropy(t_logits.float().reshape(-1, t_logits.shape[-1]), cont_ids.reshape(-1))
    # student: source cache of x_{1:n-1}, translated; target reads [x_n, y_1..y_{m-1}]
    cache, pos = translated_cache_for_prefix(src, tgt, translator, prefix_ids, mask[:, :n], keep_grad=compute_grad,
                                             src_grad=src_grad and compute_grad)
    tail = torch.cat([prefix_ids[:, -1:], cont_ids[:, :-1]], dim=1)
    tail_pos = torch.arange(n - 1, n - 1 + m, device=device)[None]
    ctx = torch.enable_grad() if compute_grad else torch.no_grad()
    with ctx:
        s_logits = forward_from_cache(tgt, cache, tail, mask[:, : n + m - 1], tail_pos)
        kl = kl_teacher_student(t_logits, s_logits)
    ce_trans = F.cross_entropy(s_logits.detach().float().reshape(-1, s_logits.shape[-1]), cont_ids.reshape(-1))
    return kl, ce_trans.detach(), ce_native.detach()


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def generate_native(model, tok, prompts_ids, max_new_tokens, batch_size=16):
    """prompts_ids: list of 1-D LongTensors.  Greedy decoding, left padding.  Returns list of decoded strings."""
    outs = [None] * len(prompts_ids)
    order = sorted(range(len(prompts_ids)), key=lambda i: -len(prompts_ids[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s: s + batch_size]
        ids, mask = left_pad([prompts_ids[i] for i in idx], tok.pad_token_id)
        ids, mask = ids.to(model.device), mask.to(model.device)
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(gen[j, ids.shape[1]:], skip_special_tokens=True)
    return outs


@torch.no_grad()
def generate_translated(src, tgt, tok, translator, prompts_ids, max_new_tokens, batch_size=16):
    """Source prefills the prompt (all but its last token), the translator maps the cache, the target decodes."""
    outs = [None] * len(prompts_ids)
    order = sorted(range(len(prompts_ids)), key=lambda i: -len(prompts_ids[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s: s + batch_size]
        ids, mask = left_pad([prompts_ids[i] for i in idx], tok.pad_token_id)
        ids, mask = ids.to(tgt.device), mask.to(tgt.device)
        cache, pos = translated_cache_for_prefix(src, tgt, translator, ids, mask)
        gen = tgt.generate(input_ids=ids, attention_mask=mask, past_key_values=cache, max_new_tokens=max_new_tokens,
                           do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(gen[j, ids.shape[1]:], skip_special_tokens=True)
    return outs


def left_pad(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = s
        mask[i, n - len(s):] = 1
    return ids, mask


def save_translator(translator, path, meta=None):
    meta = dict(meta or {}); meta.setdefault("signal", getattr(translator, "signal", "kv"))
    if hasattr(translator, "j"): meta.setdefault("j", translator.j); meta.setdefault("tail", getattr(translator, "tail", None))
    torch.save({"state": translator.state_dict(), "meta": meta}, path)


def load_translator(path, kind, src, tgt, device="cuda"):
    ck = torch.load(path, map_location=device)
    signal = ck.get("meta", {}).get("signal", "kv")
    if kind == "bottom":
        signal = str(ck["meta"]["j"])
    if kind == "hybrid":
        meta = ck.get("meta", {})
        if "j" not in meta:   # older checkpoints: infer from the run directory name, e.g. runs/hybrid_j20t256_S2L/best.pt
            import re
            mm = re.search(r"hybrid_j(\d+)(?:t(\d+))?", path); meta["j"] = int(mm.group(1)); meta["tail"] = int(mm.group(2)) if mm.group(2) else None
        signal = str(meta["j"]) + (f":{meta['tail']}" if meta.get("tail") else "")
    tr = make_translator(kind, src, tgt, signal).to(device)
    state = ck["state"]
    if all(k.startswith("inner.") for k in state):   # saved from a QueryPrefillTranslator wrapper
        state = {k[len("inner."):]: v for k, v in state.items()}
    # a conv translator can be seeded from a headmix checkpoint (loads TK/TV, leaves the conv at its delta init)
    strict = not (kind == "conv" and "cK" not in state)
    tr.load_state_dict(state, strict=strict)
    return tr, ck.get("meta", {})


def load_lora(model, path):
    """Attach and merge a saved LoRA adapter into `model` (returns the merged frozen model)."""
    from peft import PeftModel
    m = PeftModel.from_pretrained(model, path).merge_and_unload().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
