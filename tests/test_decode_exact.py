"""Pipelined decoding gives the same tokens as the loop it replaced: the next token's forward pass is
queued before this token's CPU work, nothing else changes. Needs the model files (decodes a few short answers)."""
from pathlib import Path

import mlx.core as mx
import pytest

MODEL = Path(__file__).resolve().parents[1] / "models/gemma-4-e4b-it-4bit"
pytestmark = pytest.mark.skipif(not (MODEL / "config.json").exists(), reason="model files not downloaded")


def sync_tokens(a, cache, ids, n):
    """The decode loop as it was before 2026-11-15: each token's forward pass after the previous token's .item()."""
    y = mx.argmax(a.lm(inputs=mx.array([ids]), cache=cache).logits[:, -1, :], axis=-1)
    out = []
    while len(out) < n:
        t = int(y.item())
        if t in a.stop_ids:
            break
        out.append(t)
        y = mx.argmax(a.lm(inputs=y[None], cache=cache).logits[:, -1, :], axis=-1)
    return out


@pytest.fixture(scope="module")
def a():
    from engine import RECALL_INSTRUCTION, GemmaAdapter
    g = GemmaAdapter(instruction=RECALL_INSTRUCTION)
    g.load()
    return g


@pytest.mark.parametrize("q", ["What is the capital of France?", "Write one sentence about lamps.", "2+2?"])
def test_same_tokens_as_the_synchronous_loop(a, q):
    ids = [a.tok.bos_token_id] + a.text_ids(f"<|turn>user\n{q}<turn|>\n<|turn>model\n")
    text = "".join(a.generate(a.model.make_cache(), ids, max_tokens=40, max_sentences=99))
    ref = sync_tokens(a, a.model.make_cache(), ids, 40)
    assert text == a.tok.decode(ref)
