"""Training / validation stream of (prefix, continuation) pairs from Nemotron-SFT-Instruction-Following-Chat-v2
(Appendix C of the paper): natural samples (reasoning on/off alternating) and packed samples, where several
conversations are concatenated, one (question, answer) pair is lifted to the end of the prefix and its answer
becomes the continuation."""
import json, random
import torch

DATA = {"off": "data/reasoning_off.jsonl", "on": "data/reasoning_on.jsonl"}
IM_END = "<|im_end|>"


def load_rows(path, max_rows=None):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ms = [m for m in r["messages"] if not (m["role"] == "system" and not m["content"].strip())]
            if len(ms) < 2 or ms[0]["role"] not in ("user", "system"):
                continue
            rows.append(ms)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def split_rows(rows, every=50):
    """held-out = every `every`-th row (fixed), the rest is training."""
    val = [r for i, r in enumerate(rows) if i % every == 0]
    train = [r for i, r in enumerate(rows) if i % every != 0]
    return train, val


def assistant_turn_indices(ms):
    return [i for i, m in enumerate(ms) if m["role"] == "assistant"]


def strip_think(content):
    if "</think>" in content:
        return content.split("</think>", 1)[1].lstrip("\n")
    return content


def continuation_text(content, reasoning_on):
    """Reasoning-off: the answer without any think block.  Reasoning-on: the record's own think block + answer."""
    if reasoning_on:
        if "<think>" not in content:
            content = "<think>\n\n</think>\n\n" + content
        return content + IM_END
    return strip_think(content) + IM_END


def render_natural(tok, ms, reasoning_on, rng):
    idx = rng.choice(assistant_turn_indices(ms))
    hist = [dict(role=m["role"], content=strip_think(m["content"])) for m in ms[:idx]]
    prefix = tok.apply_chat_template(hist, tokenize=False, add_generation_prompt=True, enable_thinking=reasoning_on)
    cont = continuation_text(ms[idx]["content"], reasoning_on)
    return prefix, cont


def render_packed(tok, rows, reasoning_on, rng, lo, hi):
    """Concatenate conversations (system turns dropped) until the prefix is within [lo, hi] tokens."""
    merged = []
    while True:
        ms = rng.choice(rows)
        turns = [dict(role=m["role"], content=strip_think(m["content"])) for m in ms if m["role"] != "system"]
        if not turns or turns[0]["role"] != "user" or len(turns) % 2:
            continue
        cand = merged + turns
        n = len(tok(tok.apply_chat_template(cand, tokenize=False), add_special_tokens=False).input_ids)
        if n > hi:
            if len(merged) >= 2:
                break
            continue
        merged = cand
        if n >= lo:
            break
    # lift one (question, answer) pair to the end
    pairs = [i for i in range(0, len(merged) - 1, 2)]
    i = rng.choice(pairs)
    q, a = merged[i], merged[i + 1]
    rest = merged[:i] + merged[i + 2:]
    hist = rest + [q]
    prefix = tok.apply_chat_template(hist, tokenize=False, add_generation_prompt=True, enable_thinking=reasoning_on)
    # the original (unstripped) answer is needed for a think block; recover it from the merged record
    cont = continuation_text(a["content"], reasoning_on=False) if not reasoning_on else continuation_text(a["content"], True)
    return prefix, cont


class SampleStream:
    """Infinite deterministic stream of tokenised (prefix_ids, cont_ids, meta)."""

    def __init__(self, tok, rows_off, rows_on, seed=0, max_prefix=8192, max_cont=1024, pack_frac=0.5,
                 pack_lo=5000, pack_hi=8000, reasoning_mix=True):
        self.tok, self.off, self.on = tok, rows_off, rows_on
        self.rng = random.Random(seed)
        self.max_prefix, self.max_cont = max_prefix, max_cont
        self.pack_frac, self.pack_lo, self.pack_hi = pack_frac, pack_lo, pack_hi
        self.reasoning_mix = reasoning_mix
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            reasoning_on = bool(self.i % 2) if self.reasoning_mix else False
            rows = self.on if reasoning_on else self.off
            packed = self.rng.random() < self.pack_frac
            if packed:
                prefix, cont = render_packed(self.tok, rows, reasoning_on, self.rng, self.pack_lo, self.pack_hi)
            else:
                prefix, cont = render_natural(self.tok, self.rng.choice(rows), reasoning_on, self.rng)
            p = self.tok(prefix, add_special_tokens=False).input_ids
            c = self.tok(cont, add_special_tokens=False).input_ids[: self.max_cont]
            if len(p) > self.max_prefix or len(p) < 8 or len(c) < 1:
                continue
            self.i += 1
            return torch.tensor(p), torch.tensor(c), dict(reasoning_on=reasoning_on, packed=packed)


def make_streams(tok, seed=0, max_rows=40000, reasoning_mix=False, **kw):
    # NOTE: the released reasoning_on split of Nemotron-SFT-Instruction-Following-Chat-v2 carries no <think>
    # traces in the assistant content, so every sample is rendered reasoning-off (empty think placeholder closes
    # the prefix).  Rows of both files are used as plain chat data.
    off = load_rows(DATA["off"], max_rows) + load_rows(DATA["on"], max_rows // 2)
    off_tr, off_val = split_rows(off)
    train = SampleStream(tok, off_tr, off_tr, seed=seed, reasoning_mix=reasoning_mix, **kw)
    val = SampleStream(tok, off_val, off_val, seed=12345, reasoning_mix=reasoning_mix, **kw)
    return train, val


def val_set(val_stream, n=64):
    return [next(val_stream) for _ in range(n)]
