"""Speculative decoding for the answer path (ARCHITECTURE.md §4.5).

Two drafters, both verified by Gemma itself, so a draft only decides how many tokens one model pass yields:

  mtp     Google's Gemma 4 E4B "assistant" drafter (models/gemma-4-e4b-it-assistant-bf16, 159 MB): 4 small
          layers that read Gemma's own KV cache and last hidden state and propose the next few tokens.
          Rounds, verification and cache rollback come from mlx-vlm (speculative/mtp.py, cache_state.py).
  lookup  prompt lookup: propose the tokens that followed the last 1-3 tokens' most recent occurrence in a
          text source (e.g. the retrieved page for a recall answer). Uses the same verify/commit machinery.

Verification runs the drafted block through Gemma as one (1, K) pass instead of K single-token passes.
That changes matrix shapes, so on rare tokens the greedy choice can differ from one-at-a-time decoding at
the last bit (mlx-vlm says so in gemma4/speculative_verifier.py); eval/spec/drafter_eval.py measures how
often answers change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import mlx.core as mx

ROOT = Path(__file__).resolve().parent
DRAFTER_PATH = ROOT / "models/gemma-4-e4b-it-assistant-bf16"


def load_drafter(path: Path = DRAFTER_PATH):
    from mlx_vlm.speculative.drafters import load_drafter as _load
    model, kind = _load(str(path))
    assert kind == "mtp", kind
    mx.eval(model.parameters())
    return model


def _greedy(logits):
    return mx.argmax(logits, axis=-1)


class RoundStats:
    """Per-round (accepted, drafted) counts, recorded from mlx-vlm's round hook."""

    def __init__(self):
        self.rounds = []

    def __enter__(self):
        from mlx_vlm.speculative import mtp as M
        self._orig = M._record_speculative_round

        def record(draft_model, accepted, drafted):
            self.rounds.append((int(accepted), int(drafted)))
            return self._orig(draft_model, accepted, drafted)
        M._record_speculative_round = record
        return self

    def __exit__(self, *_):
        from mlx_vlm.speculative import mtp as M
        M._record_speculative_round = self._orig


def mtp_tokens(adapter, drafter, cache, ids: list[int], max_tokens: int,
               block_size: Optional[int] = None) -> Iterator[int]:
    """Greedy tokens after feeding `ids` into `cache` (mutated), drafted by the assistant model.
    Yields token ids one by one (several may arrive per Gemma pass)."""
    from mlx_vlm.speculative.mtp import _buffer_mtp_target_cache, _mtp_rounds
    inp = mx.array([ids])
    out = adapter.lm(inputs=inp, cache=cache, return_hidden=True, return_shared_kv=True)
    first = _greedy(out.logits[:, -1, :])
    _buffer_mtp_target_cache(cache, drafter, block_size)
    mx.eval(first)
    first = int(first.item())
    yield first
    for tok, _ in _mtp_rounds(adapter.model, drafter, cache, out.hidden_states[-1], out.shared_kv_states,
                              prompt_tokens=inp, first_bonus=first, max_tokens=max_tokens, sampler=_greedy,
                              draft_block_size=block_size, token_dtype=inp.dtype, greedy_sampling=True):
        yield int(tok)


def plain_tokens(adapter, cache, ids: list[int], max_tokens: int) -> Iterator[int]:
    """One-token-at-a-time greedy decoding, as engine.generate does (the reference)."""
    y = mx.argmax(adapter.lm(inputs=mx.array([ids]), cache=cache).logits[:, -1, :], axis=-1)
    for _ in range(max_tokens):
        t = int(y.item())
        yield t
        y = mx.argmax(adapter.lm(inputs=y[None], cache=cache).logits[:, -1, :], axis=-1)
