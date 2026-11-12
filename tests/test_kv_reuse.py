"""kv_reuse.py bookkeeping, without the model.

Synthetic "KV": token t at position p has key rope(raw_t, p) and value val_t, independent of its context, so a
cache assembled from pieces computed elsewhere must equal the cache built directly, entry for entry.
"""
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.cache import KVCache, RotatingKVCache

from kv_reuse import Piece, append, take

W, D, H = 8, 16, 2


class Lay:
    def __init__(self):
        self.n, self.W = 3, W
        self.window = [True, False, True]
        self.ropes = [nn.RoPE(D, base=10000.0)] * 3


def make():
    return [RotatingKVCache(max_size=W) if w else KVCache() for w in Lay().window]


def feed(lay, cache, toks, chunk=5):
    """Feed token ids through every layer's cache, multi-token chunks like a prefill."""
    for s in range(0, len(toks), chunk):
        ids = toks[s:s + chunk]
        pos = int(cache[0].offset)
        raw = mx.stack([RAW[t] for t in ids], axis=1)[None]                 # [1, H, L, D]
        val = mx.stack([VAL[t] for t in ids], axis=1)[None]
        for i, c in enumerate(cache):
            c.update_and_fetch(lay.ropes[i](raw, offset=pos), val)
    return cache


mx.random.seed(0)
RAW = mx.random.normal((100, H, D))
VAL = mx.random.normal((100, H, D))


def window_state(c):
    k, v = c.state
    if c.offset > c.max_size and c._idx != k.shape[2]:
        k, v = c._temporal_order(c.keys), c._temporal_order(c.values)
    return k[:, :, -min(int(c.offset), W):], v[:, :, -min(int(c.offset), W):]


def close(a, b):
    return mx.allclose(a, b, atol=1e-4).item()


def piece_of(lay, anchor, toks, window):
    c = feed(lay, feed(lay, make(), anchor), toks)
    return take(lay, c, len(anchor), window)


def test_assembled_equals_direct_then_keeps_working():
    lay = Lay()
    prefix, a, b, q = list(range(0, 7)), list(range(10, 23)), list(range(30, 34)), list(range(40, 52))
    # pieces computed after different anchors, at different positions than they land
    pa = piece_of(lay, [99, 98], a, window=False)
    pb = piece_of(lay, [97] * 11, b + list(range(60, 70)), window=True)   # b plus 10 more tokens: n=14 > W
    direct = feed(lay, make(), prefix + a + b + list(range(60, 70)))
    got = append(lay, feed(lay, make(), prefix), [pa, pb])
    for i, (x, y) in enumerate(zip(got, direct)):
        assert int(x.offset) == int(y.offset)
        if lay.window[i]:
            kx, vx = window_state(x)
            ky, vy = window_state(y)
            assert close(kx, ky) and close(vx, vy)
        else:
            assert close(x.state[0], y.state[0]) and close(x.state[1], y.state[1])
    # the question prefill and one-token decode steps afterwards see the same entries
    feed(lay, got, q, chunk=12)
    feed(lay, direct, q, chunk=12)
    for t in (70, 71, 72):
        feed(lay, got, [t], chunk=1)
        feed(lay, direct, [t], chunk=1)
    for i, (x, y) in enumerate(zip(got, direct)):
        if lay.window[i]:
            assert close(window_state(x)[0], window_state(y)[0])
        else:
            assert close(x.state[0], y.state[0])


def test_piece_without_window_entries_near_the_end_raises():
    lay = Lay()
    pa = piece_of(lay, [99], list(range(10, 15)), window=False)
    with pytest.raises(ValueError):
        append(lay, feed(lay, make(), list(range(20))), [pa])


def test_short_window_piece_after_windowless_piece_raises():
    lay = Lay()
    pa = piece_of(lay, [99], list(range(10, 30)), window=False)
    pb = piece_of(lay, [99], list(range(30, 33)), window=True)            # only 3 real entries, W = 8
    with pytest.raises(ValueError):
        append(lay, feed(lay, make(), list(range(5))), [pa, pb])


def test_head_is_a_prefix():
    lay = Lay()
    full = piece_of(lay, [99], list(range(10, 30)), window=False)
    cut = piece_of(lay, [99], list(range(10, 22)), window=False)
    h = full.head(12)
    assert h.n == cut.n and h.window is None
    for i in h.full:
        assert close(h.full[i][0], cut.full[i][0]) and close(h.full[i][1], cut.full[i][1])


def test_source_cache_is_untouched():
    lay = Lay()
    base = feed(lay, make(), list(range(9)))
    before = [mx.array(c.state[0]) for c in base]
    pb = piece_of(lay, [99], list(range(10, 20)), window=True)
    append(lay, base, [pb])
    assert all(int(c.offset) == 9 for c in base)
    assert all(mx.array_equal(b, c.state[0]).item() for b, c in zip(before, base))


def test_all_window_entries_survive_head_and_assemble_like_direct():
    lay = Lay()
    toks = list(range(10, 30))                                            # n = 20 > W
    c = feed(lay, feed(lay, make(), [99]), toks, chunk=20)                # one prefill: all 20 still in the buffer
    full = take(lay, c, 1, window="all")
    assert next(iter(full.window.values()))[0].shape[2] == 20
    h = full.head(12)
    assert h.window is not None and next(iter(h.window.values()))[0].shape[2] == 12
    prefix = list(range(50, 55))
    got = append(lay, feed(lay, make(), prefix), [h])                     # window entries real: no raise
    direct = feed(lay, make(), prefix + toks[:12])
    for i, (x, y) in enumerate(zip(got, direct)):
        if lay.window[i]:
            assert close(window_state(x)[0], window_state(y)[0])
