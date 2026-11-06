"""Pre-read with a KV-cache snapshot (works past Gemma 4's 512-token sliding window).

mlx-vlm's prompt-cache reuse can't trim a sliding-window cache that has wrapped, so
above ~512 prompt tokens it silently re-reads everything (mlx_split.py, budgets 560/1120).
Here the prefix [instructions + screen] is prefilled once, the cache is copied, and
each question restores the copy and prefills only its own tokens.

  python m0_ane/kv_snapshot.py 280,560,1120
"""
import copy
import json
import re
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "m0_screen"
MODEL = ROOT / "models/gemma-4-e4b-it-4bit"
OUT = ROOT / "m0_ane/out/kv_snapshot.jsonl"
BUDGETS = [int(b) for b in sys.argv[1].split(",")] if len(sys.argv) > 1 else [280, 560, 1120]


def norm(s):
    return re.sub(r"[\s,$*`'\"]", "", s.lower())


def prompt_for(processor, config, q):
    return apply_chat_template(
        processor, config,
        f"Answer the question about this screenshot. Reply with only the answer, no explanation.\nQuestion: {q}",
        num_images=1)


def clone_cache(cache):
    out = []
    for c in cache:
        n = copy.copy(c)
        n.keys = None if c.keys is None else mx.array(c.keys)
        n.values = None if c.values is None else mx.array(c.values)
        out.append(n)
    mx.eval([x for n in out for x in (n.keys, n.values) if x is not None])
    return out


def main():
    model, processor = load(str(MODEL))
    config = model.config
    lm = model.language_model
    tok = processor.tokenizer
    stop_ids = {tok.eos_token_id, *[tok.convert_tokens_to_ids(t) for t in ("<turn|>", "<end_of_turn>") if t in tok.get_vocab()]}
    qs = json.loads((SCREEN / "questions.json").read_text())
    by_screen = {}
    for q in qs:
        by_screen.setdefault(q["screen"], []).append(q)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        for b in BUDGETS:
            processor.image_processor.max_soft_tokens = b
            rows = []
            for screen, sq in by_screen.items():
                img = str(SCREEN / f"screens/{screen}.png")
                ins = [prepare_inputs(processor, images=[img], prompts=[prompt_for(processor, config, q["question"])]) for q in sq]
                ids = [x["input_ids"][0].tolist() for x in ins]
                P = next(i for i, t in enumerate(zip(*ids)) if len(set(t)) > 1)  # shared prefix: instructions + screen
                x0 = ins[0]
                extra = {k: v for k, v in x0.items() if k not in ("input_ids", "pixel_values", "attention_mask", "mask")}
                # --- background part: encode + prefill the prefix once ---
                t0 = time.perf_counter()
                emb = model.get_input_embeddings(input_ids=x0["input_ids"], pixel_values=x0["pixel_values"], **extra)
                cache = model.make_cache()
                lm_kw = {}
                for k in ("mm_token_type_ids", "token_type_ids"):
                    if k in extra and extra[k] is not None:
                        lm_kw[k] = extra[k][:, :P]
                lm(inputs=None, inputs_embeds=emb.inputs_embeds[:, :P], per_layer_inputs=emb.per_layer_inputs[:, :P],
                   cache=cache, **lm_kw)
                mx.eval([c.state for c in cache])
                pre_s = time.perf_counter() - t0
                snap = clone_cache(cache)
                # --- question part: restore snapshot, prefill only the question ---
                for q, qi in zip(sq, ids):
                    cache = clone_cache(snap)
                    suffix = mx.array([qi[P:]])
                    mx.reset_peak_memory()
                    t0 = time.perf_counter()
                    logits = lm(inputs=suffix, cache=cache).logits
                    y = mx.argmax(logits[:, -1, :], axis=-1)
                    mx.eval(y)
                    ttft = time.perf_counter() - t0
                    out = [int(y.item())]
                    while out[-1] not in stop_ids and len(out) < 40:
                        y = mx.argmax(lm(inputs=y[None], cache=cache).logits[:, -1, :], axis=-1)
                        out.append(int(y.item()))
                    ans = tok.decode([t for t in out if t not in stop_ids]).strip()
                    ok = any(a in norm(ans) for a in q["accept"])
                    r = dict(budget=b, screen=screen, question=q["question"], prefix_tokens=P, question_tokens=len(qi) - P,
                             background_s=round(pre_s, 3), ttft_s=round(ttft, 3), ok=ok, answer=ans,
                             peak_mem_gb=round(mx.get_peak_memory() / 1e9, 2))
                    rows.append(r)
                    f.write(json.dumps(r) + "\n"); f.flush()
                    print(f"[{b}] {screen:9s} bg={pre_s:.2f}s ttft={ttft:.3f}s ok={int(ok)} P={P} q={len(qi)-P} -> {ans[:50]!r}", flush=True)
            med = lambda k: round(statistics.median(x[k] for x in rows), 3)
            print(f"== budget {b}: n={len(rows)} background {med('background_s')}s  ttft {med('ttft_s')}s  "
                  f"acc {sum(x['ok'] for x in rows)}/{len(rows)}  peak {max(x['peak_mem_gb'] for x in rows)} GB", flush=True)

if __name__ == "__main__":
    main()
