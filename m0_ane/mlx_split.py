"""Where does screenshot-on-ask time go on the GPU (MLX), and how much does pre-reading remove?

For each image-token budget and each of the 8 M0 screens:
  encoder   = vision tower + projection alone (mx.eval), median of 3
  A on-ask  = time to first token with nothing cached (what existing tools do)
  B enc-cached = image features pre-computed; LLM still prefills the image tokens
  C kv-cached  = the image already prefilled into the KV cache by an earlier turn;
                 only the question tokens are prefilled (the pre-read design)
A/B/C are measured on the screen's later questions (the first one warms the caches),
with accuracy, so C's answers can be checked against A's.

  python m0_ane/mlx_split.py 280,560,1120
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.generate.common import PromptCacheState
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs
from mlx_vlm.vision_cache import VisionFeatureCache

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "m0_screen"
MODEL = ROOT / "models/gemma-4-e4b-it-4bit"
OUT = ROOT / "m0_ane/out/mlx_split.jsonl"
BUDGETS = [int(b) for b in sys.argv[1].split(",")] if len(sys.argv) > 1 else [280, 560, 1120]


def norm(s):
    return re.sub(r"[\s,$*`'\"]", "", s.lower())


def prompt_for(processor, config, q):
    return apply_chat_template(
        processor, config,
        f"Answer the question about this screenshot. Reply with only the answer, no explanation.\nQuestion: {q}",
        num_images=1)


def ttft(model, processor, prompt, image, **kw):
    t0 = time.perf_counter()
    first, text, last = None, "", None
    for r in stream_generate(model, processor, prompt, image=[image], max_tokens=40, temperature=0.0, **kw):
        if first is None:
            first = time.perf_counter() - t0
        text += r.text
        last = r
    return first, text.strip(), last


def main():
    model, processor = load(str(MODEL))
    config = model.config
    qs = json.loads((SCREEN / "questions.json").read_text())
    by_screen = {}
    for q in qs:
        by_screen.setdefault(q["screen"], []).append(q)
    p0 = prompt_for(processor, config, "QUESTION_HERE")
    print("prompt layout:", repr(p0[:60]), "...", repr(p0[-60:]))
    processor.image_processor.max_soft_tokens = 280
    ttft(model, processor, p0, str(SCREEN / "screens/doc.png"))  # warm-up

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        for b in BUDGETS:
            processor.image_processor.max_soft_tokens = b
            rows = []
            for screen, sq in by_screen.items():
                img = str(SCREEN / f"screens/{screen}.png")
                # encoder alone
                inp = prepare_inputs(processor, images=[img], prompts=[prompt_for(processor, config, sq[0]["question"])])
                pv, pos = inp["pixel_values"], inp.get("image_position_ids")
                enc = []
                for _ in range(3):
                    t0 = time.perf_counter(); mx.eval(model.encode_image(pv, pos)); enc.append(time.perf_counter() - t0)
                enc_s = statistics.median(enc)
                # warm B and C caches with the first question
                vcache, pstate = VisionFeatureCache(), PromptCacheState()
                ttft(model, processor, prompt_for(processor, config, sq[0]["question"]), img, vision_cache=vcache)
                ttft(model, processor, prompt_for(processor, config, sq[0]["question"]), img, prompt_cache_state=pstate)
                for q in sq[1:]:
                    p = prompt_for(processor, config, q["question"])
                    mx.reset_peak_memory()
                    a_t, a_ans, a_last = ttft(model, processor, p, img)
                    a_mem = mx.get_peak_memory() / 1e9
                    b_t, b_ans, _ = ttft(model, processor, p, img, vision_cache=vcache)
                    c_t, c_ans, c_last = ttft(model, processor, p, img, prompt_cache_state=pstate)
                    ok = lambda s: any(x in norm(s) for x in q["accept"])
                    r = dict(budget=b, screen=screen, question=q["question"], encoder_s=round(enc_s, 3),
                             A_s=round(a_t, 3), B_s=round(b_t, 3), C_s=round(c_t, 3),
                             A_ok=ok(a_ans), B_ok=ok(b_ans), C_ok=ok(c_ans), A_ans=a_ans, C_ans=c_ans,
                             prompt_tokens=a_last.prompt_tokens, C_prompt_tokens=c_last.prompt_tokens,
                             peak_mem_gb=round(a_mem, 2))
                    rows.append(r)
                    f.write(json.dumps(r) + "\n"); f.flush()
                    print(f"[{b}] {screen:9s} enc={enc_s:.2f} A={a_t:.2f} B={b_t:.2f} C={c_t:.2f} "
                          f"ok={int(r['A_ok'])}{int(r['B_ok'])}{int(r['C_ok'])} Cprompt={r['C_prompt_tokens']}", flush=True)
            med = lambda k: round(statistics.median(x[k] for x in rows), 3)
            print(f"== budget {b}: n={len(rows)} encoder {med('encoder_s')}s  A {med('A_s')}s  B {med('B_s')}s  C {med('C_s')}s  "
                  f"acc A {sum(x['A_ok'] for x in rows)} B {sum(x['B_ok'] for x in rows)} C {sum(x['C_ok'] for x in rows)}", flush=True)


if __name__ == "__main__":
    main()
