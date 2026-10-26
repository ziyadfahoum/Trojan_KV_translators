"""Downstream evaluation of native models and translated caches (reasoning-off, greedy).
Benchmarks: valkl (held-out KL / perplexity), arc (ARC-Challenge), mmlupro (100 per category), coqa, ruler (NIAH)."""
import argparse, json, os, random, re, string, collections, time
import numpy as np, torch
from datasets import load_dataset
import kvl
from kvl import *
from data import make_streams, val_set, load_rows, DATA

p = argparse.ArgumentParser()
p.add_argument("--pair", default="0.6B-1.7B")
p.add_argument("--cond", required=True, choices=["native", "translate"])
p.add_argument("--model", choices=["small", "large"], help="native: which model decodes")
p.add_argument("--direction", choices=["S2L", "L2S"])
p.add_argument("--kind", choices=["headmix", "headwise", "mlp", "hybrid", "bottom", "conv"])
p.add_argument("--ckpt")
p.add_argument("--name", required=True)
p.add_argument("--bench", default="valkl,arc,coqa,ruler,mmlupro")
p.add_argument("--ruler_lengths", default="4096,8192,16384")
p.add_argument("--n_ruler", type=int, default=100)
p.add_argument("--n_coqa", type=int, default=256)
p.add_argument("--mmlu_per_cat", type=int, default=100)
p.add_argument("--mmlu_tokens", type=int, default=512)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--qp_tail", type=int, default=0, help="wrap the translator: native query prefill of the last K prefix tokens")
p.add_argument("--lora", default=None, help="merge this LoRA adapter into the small model before evaluating")
args = p.parse_args()
kvl.set_pair(args.pair)
dev = "cuda"
os.makedirs("results", exist_ok=True)
S, tok = load_model("small", dev); Lg, _ = load_model("large", dev)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
if args.lora:
    S = load_lora(S, args.lora); print("[eval] merged LoRA", args.lora)
if args.cond == "native":
    dec = S if args.model == "small" else Lg
    src, tr = None, None
else:
    src, dec = (S, Lg) if args.direction == "S2L" else (Lg, S)
    tr, meta = load_translator(args.ckpt, args.kind, src, dec, dev)
    if args.qp_tail:
        from hybrid import QueryPrefillTranslator
        tr = QueryPrefillTranslator(tr, args.qp_tail).to(dev)
    print("[eval] translator meta:", meta, "qp_tail", args.qp_tail)
res_path = f"results/{args.name}.json"
results = json.load(open(res_path)) if os.path.exists(res_path) else {}
results["_cond"] = vars(args)
gen_log = open(f"results/{args.name}.generations.jsonl", "a")


def chat(user, history=None):
    msgs = (history or []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")[0]


def generate(prompts, max_new, bs):
    if args.cond == "native":
        return generate_native(dec, tok, prompts, max_new, batch_size=bs)
    return generate_translated(src, dec, tok, tr, prompts, max_new, batch_size=bs)


def save(bench, score, extra=None, items=None):
    results[bench] = dict(score=score, **(extra or {}))
    json.dump(results, open(res_path, "w"), indent=1)
    if items:
        for it in items:
            gen_log.write(json.dumps(dict(bench=bench, **it)) + "\n")
        gen_log.flush()
    print(f"[eval] {args.name} {bench}: {score:.4f} {extra or ''}", flush=True)


def normalize(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())


def f1(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(pt) & collections.Counter(gt); ns = sum(common.values())
    if ns == 0: return 0.0
    pr, rc = ns / len(pt), ns / len(gt); return 2 * pr * rc / (pr + rc)


def parse_letter(text, labels):
    m = re.findall(r"[Aa]nswer(?: is)?:?\s*\(?\**([A-J])\**\)?", text)
    if m and m[-1] in labels: return m[-1]
    m = re.findall(r"\b([A-J])\b", text)
    for c in m:
        if c in labels: return c
    return None


def match_option_text(out, labels, texts):
    """lenient fallback: the option whose (normalised) text appears in the output; ambiguous -> None"""
    o = normalize(out); hits = [l for l, t in zip(labels, texts) if normalize(t) and normalize(t) in o]
    return hits[0] if len(hits) == 1 else None


benches = args.bench.split(",")
rng = random.Random(args.seed)

# ------------------------------------------------------------------ held-out KL / PPL
if "valkl" in benches:
    if args.cond == "translate":
        _, val = make_streams(tok, seed=0, max_prefix=8192, max_cont=1024)
        vals = val_set(val, 64)
        kls, ces, cen = [], [], []
        for pre, cont, meta in vals:
            kl, ce_t, ce_n = distill_step(src, dec, tr, pre[None].to(dev), cont[None].to(dev), compute_grad=False)
            kls.append(kl.item()); ces.append(ce_t.item()); cen.append(ce_n.item())
        save("valkl", float(np.mean(kls)), dict(ppl_translated=float(np.exp(np.mean(ces))), ppl_native=float(np.exp(np.mean(cen)))))
    else:
        _, val = make_streams(tok, seed=0, max_prefix=8192, max_cont=1024)
        vals = val_set(val, 64); cen = []
        for pre, cont, meta in vals:
            full = torch.cat([pre, cont])[None].to(dev); n = len(pre)
            lg = native_logits(dec, full, torch.ones_like(full))[:, n - 1: n + len(cont) - 1]
            cen.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), cont.to(dev)).item())
        save("valkl", 0.0, dict(ppl_native=float(np.exp(np.mean(cen)))))

# ------------------------------------------------------------------ ARC-Challenge
if "arc" in benches:
    d = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    prompts, items = [], []
    for ex in d:
        opts = "\n".join(f"{l}. {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))
        q = f"{ex['question']}\n{opts}\nAnswer with the letter of the correct option only."
        prompts.append(chat(q)); items.append(dict(id=ex["id"], gold=ex["answerKey"], labels=ex["choices"]["label"], texts=ex["choices"]["text"]))
    outs = generate(prompts, 96, 64)
    correct = []
    for it, o in zip(items, outs):
        it["pred"] = parse_letter(o, it["labels"]) or match_option_text(o, it["labels"], it["texts"])
        it["out"] = o; correct.append(it["pred"] == it["gold"])
    save("arc", float(np.mean(correct)), dict(n=len(correct), unparsed=sum(it["pred"] is None for it in items)), items)

# ------------------------------------------------------------------ CoQA (multi-turn chat rendering)
if "coqa" in benches:
    d = load_dataset("stanfordnlp/coqa", split="validation")
    prompts, items = [], []
    r = random.Random(args.seed)
    for ex in list(d)[: args.n_coqa]:
        qs, ans = ex["questions"], ex["answers"]["input_text"]
        t = r.randrange(len(qs))
        hist = [{"role": "user", "content": f"Read the passage and answer the questions briefly.\n\nPassage: {ex['story']}\n\nQuestion: {qs[0]}"}]
        for i in range(t):
            if i > 0: hist.append({"role": "user", "content": f"Question: {qs[i]}"})
            hist.append({"role": "assistant", "content": ans[i]})
        if t > 0: hist.append({"role": "user", "content": f"Question: {qs[t]}"})
        ids = tok.apply_chat_template(hist, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")[0]
        if len(ids) > 1024: continue
        prompts.append(ids); items.append(dict(turn=t, gold=ans[t]))
    outs = generate(prompts, 32, 32)
    f1s, gia = [], []
    for it, o in zip(items, outs):
        o = o.strip().split("\n")[0]; it["out"] = o
        f1s.append(f1(o, it["gold"])); gia.append(normalize(it["gold"]) in normalize(o))
    save("coqa", float(np.mean(f1s)), dict(gold_in_answer=float(np.mean(gia)), n=len(f1s)), items)

# ------------------------------------------------------------------ RULER NIAH (synthetic)
if "ruler" in benches:
    rows = load_rows(DATA["off"], 3000)
    hay_text = " ".join(m["content"].replace("\n", " ") for ms in rows for m in ms if m["role"] == "assistant")
    hay_ids = tok(hay_text, add_special_tokens=False).input_ids
    words = ["apple", "river", "candle", "mirror", "harbor", "falcon", "velvet", "lantern", "meadow", "marble",
             "compass", "ember", "glacier", "orchid", "quartz", "saddle", "thunder", "walnut", "zephyr", "beacon"]
    scores = {}
    for n_ctx in [int(x) for x in args.ruler_lengths.split(",")]:
        r = random.Random(args.seed + n_ctx); prompts, items = [], []
        for i in range(args.n_ruler):
            key = f"{r.choice(words)}-{r.choice(words)}"; num = str(r.randrange(10 ** 6, 10 ** 7))
            needle = f" The special magic number for {key} is: {num}. "
            body_len = n_ctx - 120
            start = r.randrange(0, len(hay_ids) - body_len)
            body = hay_ids[start: start + body_len]
            depth = r.random()
            cut = int(depth * len(body))
            text = tok.decode(body[:cut]) + needle + tok.decode(body[cut:])
            q = (f"Some special magic numbers are hidden in the following text. Make sure to memorize them.\n\n{text}\n\n"
                 f"What is the special magic number for {key} mentioned in the text? Answer with the number only.")
            prompts.append(chat(q)); items.append(dict(n_ctx=n_ctx, depth=round(depth, 2), gold=num))
        bs = max(1, 32768 // n_ctx)
        outs = generate(prompts, 24, bs)
        rec = []
        for it, o in zip(items, outs):
            it["out"] = o; rec.append(it["gold"] in o)
        scores[str(n_ctx)] = float(np.mean(rec))
        print(f"[eval] ruler {n_ctx}: {scores[str(n_ctx)]:.3f}", flush=True)
        for it in items: gen_log.write(json.dumps(dict(bench="ruler", **it)) + "\n")
    save("ruler", float(np.mean(list(scores.values()))), dict(by_length=scores))

# ------------------------------------------------------------------ MMLU-Pro
if "mmlupro" in benches:
    d = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    bycat = collections.defaultdict(list)
    for ex in d: bycat[ex["category"]].append(ex)
    sel = []
    for cat in sorted(bycat):
        rr = random.Random(args.seed); rr.shuffle(bycat[cat]); sel += bycat[cat][: args.mmlu_per_cat]
    prompts, items = [], []
    for ex in sel:
        labels = list("ABCDEFGHIJ")[: len(ex["options"])]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, ex["options"]))
        q = (f"The following is a multiple choice question about {ex['category']}.\n\n{ex['question']}\n{opts}\n\n"
             f"Think briefly, then finish your reply with 'Answer: <letter>'.")
        prompts.append(chat(q)); items.append(dict(id=ex["question_id"], cat=ex["category"], gold=ex["answer"], labels=labels))
    outs = generate(prompts, args.mmlu_tokens, 32)
    percat = collections.defaultdict(list)
    for it, o in zip(items, outs):
        it["pred"] = parse_letter(o, it["labels"]); it["out"] = o[-300:]
        percat[it["cat"]].append(it["pred"] == it["gold"])
    macro = float(np.mean([np.mean(v) for v in percat.values()]))
    save("mmlupro", macro, dict(n=len(items), unparsed=sum(it["pred"] is None for it in items),
                                per_category={k: float(np.mean(v)) for k, v in percat.items()}), items)
print("[eval] done", json.dumps({k: (v["score"] if isinstance(v, dict) and "score" in v else None) for k, v in results.items()}))
