"""Retrieval made faster without changing a value (M3, 2026-11-15): the question pass stops at the last key layer and
runs in pieces, and key scoring is one matmul with the max taken before the cast and scale. Both must equal the code
they replaced bit for bit. Needs the model files (runs a short pass on the GPU)."""
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / "models/gemma-4-e4b-it-4bit"
pytestmark = pytest.mark.skipif(not (MODEL / "config.json").exists(), reason="model files not downloaded")


def einsum_score(q, scale, k8, ks):
    """The scoring as it was before 2026-11-15."""
    hk = k8.shape[1]
    g = q.reshape(hk, q.shape[0] // hk, q.shape[1], q.shape[2])
    k = k8.astype(mx.float16) * ks
    s = mx.einsum("hgqd,nhkd->nhgqk", g, k).astype(mx.float32) * scale
    return s.max(axis=-1).mean(axis=(1, 2, 3))


@pytest.fixture(scope="module")
def be():
    import kv_reuse as K
    import recall_backend as RB
    from engine import RECALL_INSTRUCTION, GemmaAdapter
    a = GemmaAdapter(instruction=RECALL_INSTRUCTION)
    a.load()
    b = object.__new__(RB.RecallBackend)
    b.a, b.lay = a, K.Layout(a)
    b._ids = lambda s: tuple(a.text_ids(s))
    return b


def full_pass_queries(be, text, T):
    """The question pass as it was: all 24 text layers in one piece, captured throughout."""
    import kv_reuse as K
    import recall_backend as RB
    ids = [be.a.tok.bos_token_id] + list(be._ids(" " + text))
    RB._CAPTURE["on"], RB._CAPTURE["calls"] = True, []
    try:
        RB.RecallBackend._run(be.a.prefill_text_steps(be.a.model.make_cache(), ids, layers_per_piece=be.lay.n))
    finally:
        RB._CAPTURE["on"] = False
    out = {}
    for L in RB.KEY_LAYERS:
        q, scale = RB._CAPTURE["calls"][L]
        out[L] = (K.rotate(be.lay.ropes[L], q[:, :, 1:, :], T - 1)[0].astype(mx.float16), float(scale))
    RB._CAPTURE["calls"] = []
    return out


def test_queries_are_identical(be):
    import recall_backend as RB
    for text in ("What was the rent for 974 Halsey Court, App 3B?", "how long", "日本語 🍩 $3,488/mo"):
        ref, got = full_pass_queries(be, text, 1150), be._queries(text, 1150)
        for L in RB.KEY_LAYERS:
            assert ref[L][1] == got[L][1]
            assert mx.array_equal(ref[L][0], got[L][0]).item(), (text, L)


def test_other_pieces_in_between_are_not_captured(be):
    """A page build's attention between the retrieval's pieces must not land in its captured queries."""
    import recall_backend as RB
    text = "What was the rent for 974 Halsey Court?"
    ref = be._queries(text, 1150)
    gen = be._queries_steps(text, 1150)
    next(gen)
    try:
        while True:
            RB.RecallBackend._run(be.a.prefill_text_steps(be.a.model.make_cache(), [2, 100, 200], layers_per_piece=4))
            next(gen)
    except StopIteration as e:
        got = e.value
    for L in RB.KEY_LAYERS:
        assert mx.array_equal(ref[L][0], got[L][0]).item()


def test_scores_are_identical(be):
    import recall_backend as RB
    rng = np.random.default_rng(0)
    qv = be._queries("What was the rent for 974 Halsey Court?", 1150)
    for n in (32, 31, 1):
        D = qv[RB.KEY_LAYERS[0]][0].shape[-1]
        k8 = mx.array(rng.integers(-127, 127, (n, 2, 1092, D)).astype(np.int8))
        ks = mx.array((rng.random((n, 2, 1092, 1)) * 0.02).astype(np.float16))
        for L in RB.KEY_LAYERS:
            q, scale = qv[L]
            a, b = einsum_score(q, scale, k8, ks), RB.RecallBackend._score(q, scale, k8, ks)
            assert np.array_equal(np.asarray(a), np.asarray(b)), (n, L)


def test_rewound_question_prefill_is_exact(be):
    """M3 (2026-11-15): a question prefill whose last words changed is cut back to its longest still-valid prefix
    (window layers restored from a snapshot) and continued; the cache must equal one that never took the wrong turn,
    built with the same prefill steps."""
    import recall_backend as RB
    from recall_backend import window_snapshot
    a = be.a
    be.base = None
    ctx = [a.tok.bos_token_id] + a.text_ids("<|turn>user\n" + "notes about the page. " * 150)
    A = a.text_ids(" what was the rent for 974 Halsey Corps apartment")
    B = a.text_ids(" what was the rent for 974 Halsey Court, App 3B?")
    k = next(i for i in range(min(len(A), len(B))) if A[i] != B[i])
    c = RB.RecallBackend._run(be._prefill(a.model.make_cache(), ctx))
    s = {"cache": c, "end": (int(c[0].offset), window_snapshot(be.lay, c)), "ids": (), "cov": 0, "dirty": False,
         "snaps": []}
    for upto in (k - 2, len(A) - 1):                  # two partials' prefills, the second past the changed word
        RB.RecallBackend._run(be._prefill(c, A[s["cov"]:upto], layers=4))
        s["snaps"] = (s["snaps"] + [(upto, (int(c[0].offset), window_snapshot(be.lay, c)))])[-RB.SPEC_SNAPS:]
        s.update(ids=tuple(A), cov=upto)
    be._rewind(s, B, len(B) - 1)
    assert s["cov"] == k - 2 and int(c[0].offset) == len(ctx) + k - 2
    RB.RecallBackend._run(be._prefill(c, B[k - 2:len(B) - 1], layers=4))
    ref = RB.RecallBackend._run(be._prefill(a.model.make_cache(), ctx))
    RB.RecallBackend._run(be._prefill(ref, A[:k - 2], layers=4))
    RB.RecallBackend._run(be._prefill(ref, B[k - 2:len(B) - 1], layers=4))
    for x, y in zip(c, ref):
        assert int(x.offset) == int(y.offset)
        kx, vx = x.state
        ky, vy = y.state
        n = min(kx.shape[2], ky.shape[2]) if x.offset > kx.shape[2] else int(x.offset)
        assert mx.array_equal(kx[:, :, :n], ky[:, :, :n]).item() and mx.array_equal(vx[:, :, :n], vy[:, :, :n]).item()
    la = a.lm(inputs=mx.array([[B[-1]]]), cache=c).logits
    lb = a.lm(inputs=mx.array([[B[-1]]]), cache=ref).logits
    assert mx.array_equal(la, lb).item()
