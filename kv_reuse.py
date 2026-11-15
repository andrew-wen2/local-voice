"""Stored KV pieces joined at question time (ARCHITECTURE.md §3.11).

A piece is the KV cache of a run of tokens (a retrieved page, the current screen turn) computed once, ahead of
time, after a short fixed anchor (<bos> plus the fixed text that always precedes its slot) instead of after
whatever happens to precede it at question time. At question time the pieces are appended to a copy of a
cached prefix with their keys re-rotated to their new positions (cached keys are post-RoPE, values are never
rotated; §3.10 measured the rotation exact). Only the question is prefilled.

What is approximate: a piece never saw the text placed before it at question time (CacheBlend's "missing
cross-attention"). What is exact: the positions, and every token computed at question time attends to the
right entries at the right positions.

Which entries a piece needs (Gemma 4 E4B: 4 full-attention layers keep every token; 20 window layers see only
the last W = 512 positions):
  full layers    always, every token (~16 KB a token at bf16).
  window layers  only if some token computed at question time can lie within W positions after the piece.
                 In the recall layout the screen turn (≥ 1,000 tokens) sits between the pages and the question,
                 so pages never need theirs: only the screen piece stores window entries (its last W tokens).
`append` checks this: after assembly the window buffers must hold real entries for the last min(W, offset)
positions, or it raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from engine import clone_cache



@dataclass
class Piece:
    n: int                                  # tokens
    pos: int                                # position of its first token when it was computed
    full: dict                              # layer -> (keys, values), [1, H, n, D]
    window: Optional[dict] = None           # layer -> (keys, values) of its last min(n, W) tokens, or None

    @property
    def nbytes(self) -> int:
        kv = list(self.full.values()) + (list(self.window.values()) if self.window else [])
        return sum(k.nbytes + v.nbytes for k, v in kv)

    def head(self, n: int) -> "Piece":
        """The first n tokens. Causal attention: their KV doesn't depend on later tokens, so it's exact.
        Window entries survive only if every token's were kept (take(window="all")); otherwise dropped."""
        assert 0 < n <= self.n
        win = None
        if self.window and next(iter(self.window.values()))[0].shape[2] == self.n:
            win = {i: (k[:, :, :n], v[:, :, :n]) for i, (k, v) in self.window.items()}   # append keeps the last W
        return Piece(n, self.pos, {i: (k[:, :, :n], v[:, :, :n]) for i, (k, v) in self.full.items()}, win)


class Layout:
    """Per-layer facts the assembly needs, read once from a loaded GemmaAdapter."""

    def __init__(self, adapter):
        layers = adapter.lm.model.layers
        self.n = adapter.lm.model.first_kv_shared_layer_idx           # 24 layers own their KV
        self.ropes = [layers[i].self_attn.rope for i in range(self.n)]
        self.window = [layers[i].self_attn.is_sliding for i in range(self.n)]
        self.W = adapter.lm.config.sliding_window


def rotate(rope, k, d: int):
    """Re-rotate post-RoPE keys [1, H, L, D] by d positions (each key as a one-token sequence at offset d),
    in float32, back to the cache dtype."""
    if d == 0:
        return k
    B, H, L, D = k.shape
    r = rope(k.astype(mx.float32).reshape(B, H * L, 1, D), offset=d)
    return r.reshape(B, H, L, D).astype(k.dtype)


def _temporal(c):
    """A window cache's entries oldest first (they are already after a multi-token prefill or `append`)."""
    k, v = c.state
    if c.offset > c.max_size and c._idx != k.shape[2]:
        k, v = c._temporal_order(c.keys), c._temporal_order(c.values)
    return k, v


def take(lay: Layout, cache, pos: int, window) -> Piece:
    """Tokens [pos, offset) of a cache as a piece. window=True also keeps the window layers' last W entries;
    window="all" keeps every token's (needs them still in the buffer: right after one multi-token prefill)."""
    n = int(cache[0].offset) - pos
    full, win = {}, {} if window else None
    for i, c in enumerate(cache):
        if lay.window[i]:
            if window:
                assert c.offset == pos + n
                k, v = _temporal(c)
                m = n if window == "all" else min(n, lay.W)
                assert k.shape[2] >= m, "window entries already rotated out of the buffer"
                win[i] = (mx.contiguous(k[:, :, -m:]), mx.contiguous(v[:, :, -m:]))
        else:
            k, v = c.state
            full[i] = (mx.contiguous(k[:, :, pos:pos + n]), mx.contiguous(v[:, :, pos:pos + n]))
    p = Piece(n, pos, full, win)        # copies: an MLX slice keeps its parent's whole buffer alive
    mx.eval([x for kv in list(full.values()) + list((win or {}).values()) for x in kv])
    return p


def append(lay: Layout, cache, pieces: list[Piece], copy: bool = True):
    """The cache (a copy unless copy=False) with `pieces` appended in order, keys re-rotated to where they land.
    Raises if a window layer would be left without real entries for the last min(W, offset) positions."""
    cache = clone_cache(cache) if copy else cache
    real = min(int(cache[0].offset), lay.W)      # trailing positions whose window entries are real
    for p in pieces:
        start = int(cache[0].offset)
        d = start - p.pos
        for i, c in enumerate(cache):
            assert int(c.offset) == start
            if lay.window[i]:
                k0, v0 = _temporal(c)
                if p.window is not None:
                    k, v = p.window[i]
                    k, v = k[:, :, -lay.W:], v[:, :, -lay.W:]
                    k = rotate(lay.ropes[i], k, d)
                    k, v = mx.concatenate([k0, k], axis=2), mx.concatenate([v0, v], axis=2)
                else:
                    k, v = k0, v0                      # stale; must be pushed out by later real entries
                k, v = k[:, :, -lay.W:], v[:, :, -lay.W:]
                c.keys, c.values = k, v
                c.offset, c._idx = start + p.n, k.shape[2]
            else:
                k, v = p.full[i]
                k0, v0 = c.state
                c.keys = mx.concatenate([k0, rotate(lay.ropes[i], k, d)], axis=2)
                c.values = mx.concatenate([v0, v], axis=2)
                c.offset = start + p.n
        real = min(real + p.n, lay.W) if p.window is not None else 0
    if real < min(int(cache[0].offset), lay.W):
        raise ValueError(f"window layers would attend to {min(int(cache[0].offset), lay.W) - real} positions "
                         "without real entries: a piece without window entries is within W of the end")
    mx.eval([x for c in cache for x in (c.keys, c.values)])
    return cache
