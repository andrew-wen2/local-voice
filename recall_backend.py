"""recall_backend.py — the scheduler's GPU work on real KV caches (ARCHITECTURE.md §5.9, §3.11). MLX, so every
method runs on the GPU worker thread.

Context layout (token ids):
  prefix   <bos> <|turn>user\\n timeline <turn|>\\n  [finished exchanges, each a user + model turn]  <|turn>user\\n
           pages header                                                            exact; rebuilt or appended to
  pages    "\\n\\n" + time header + page text, one per retrieved page, oldest first:
           the first `frontier` computed exactly after the prefix (upgrade), the rest stored graph pieces
  screen   <turn|>\\n + the screen turn (image + instructions + "Question:")        stored piece, its own read
  question the question and the model turn's opening                              prefilled before key up if
                                                                                    possible, else at release
The pages and screen parts are §3.11's layout. The timeline is closed in its own turn (§3.11 had timeline and pages
in one turn) so that a finished exchange can be appended after it without recomputing the timeline: that change
is not measured yet.

Stored page pieces (graph, §3.11): a page is computed after the stored pieces of its K_PARENTS most similar pages
first seen before it (the retriever's BM25, the page's title and text as the query), loaded and re-rotated, after
the anchor <bos> <|turn>user\\n pages header. Its full-attention entries stay in memory (16 KB a token); its last W
window entries, needed only when it is a parent in a later build, go to a private folder on disk and are deleted
when the app quits. A page's piece is rebuilt when its version changes (a visit of it ended: its text and time
header changed).

Kept screens (visual memory, §5.12): retrieved past screens come first after the prefix, each as its own user turn
"(Earlier screen) [time] title" + its image tokens, computed exactly from its stored vision features (screen_store.py)
while you speak, never as a stored KV piece; then the pages turn opens (a fixed opener, computed exactly), then pages,
then the current screen. Screens are found by text (frame BM25) and Gemma's own keys (layers 11 + 17, calibrated by
the content-free query, §12.17/12.19), combined by better rank. A kept screen's keys are computed in quiet time.
Slot kinds: a page (key, version, lines); a screen (SCREEN, frame id); the opener (OPENER,); keying work (KEY, frame id).

Pieces follow the scheduler's protocol: a generator announces the kind of its next piece, then runs it. Prefill
runs in chunks of CHUNK tokens, one text layer a piece, so piece time stays short whatever the page size; such
kinds are named by size ("L1x512"), and `prior` gives their first estimate from the measured prefill rate.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx

import kv_reuse as K
import numpy as np

from engine import TURN_END, TURN_USER, clone_cache, split_complete_sentences, split_first_phrase
from recall import HINT, PAGES_HEAD
from screen_store import KEY_LAYERS, Screen, quantize_int8

SCREEN, OPENER, KEY = "§screen", "§opener", "§key"
KEPT_HEAD = "(Earlier screen from {time}, no longer on screen) {title}"       # a kept screen's label
NULL_Q = "N/A"                    # ICR's content-free query (§12.17)

K_PARENTS = 4                 # §3.11: similar4 met the bar
CHUNK = 512                   # tokens per prefill chunk
TOKENS_PER_S = 2200           # prefill rate, plugged in (§5.8); first estimate only, then measured per kind
PIECE_OVERHEAD_S = 0.003      # S0: pausing costs ~3 ms a piece
HEAD = (PAGES_HEAD + HINT).rstrip("\n")
VISION_KINDS = ["prep", "patch"] + ["vision"] * 16 + ["pool"]
PRIORS = {                    # seconds; S0 (§5.8) for the read, estimates for the rest
    "prep": 0.061, "patch": 0.088, "vision": 0.080, "pool": 0.010, "embed": 0.008, "text": 0.042,
    "check": 0.0005, "plan_build": 0.010, "assemble": 0.015, "answer_prefill": 0.060, "decode": 0.026,
    "text1": 0.021, "screen_prep": 0.065,
}


def kind(slot) -> str:
    return slot[0] if isinstance(slot[0], str) and slot[0].startswith("§") else "page"


# Queries of a forward pass, captured at chosen layers (finding screens by Gemma's own keys).
import mlx_vlm.models.gemma4.language as _GL  # noqa: E402

_orig_sdpa = _GL.scaled_dot_product_attention
_CAPTURE = {"on": False, "calls": []}


def _sdpa(queries, keys, values, cache=None, scale=1.0, mask=None, **kw):
    if _CAPTURE["on"]:
        _CAPTURE["calls"].append((queries, scale))
    return _orig_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)


_GL.scaled_dot_product_attention = _sdpa


def steps(gen, kinds):
    """Run an engine step generator (one GPU step per next()) in the announce protocol: announce kinds[i], then
    run step i. The engine generator must have exactly len(kinds) steps; returns its value."""
    for k in kinds:
        yield k
        next(gen)
    try:
        next(gen)
    except StopIteration as e:
        return e.value
    raise RuntimeError("engine step generator ran longer than expected")


def window_snapshot(lay, cache):
    """Copies of the window layers' buffers and ring state (full layers are cut by offset instead)."""
    snap = {}
    for i, c in enumerate(cache):
        if lay.window[i]:
            snap[i] = (None if c.keys is None else mx.array(c.keys), None if c.values is None else mx.array(c.values),
                       c.offset, c._idx)
    mx.eval([x for v in snap.values() for x in v[:2] if x is not None])
    return snap


def cut(lay, cache, offset: int, snap):
    """Cut a cache back to `offset` tokens: full layers by offset (later writes overwrite), window layers from a
    snapshot taken at that offset."""
    for i, c in enumerate(cache):
        if lay.window[i]:
            k, v, off, idx = snap[i]
            assert off == offset, (off, offset)
            c.keys = None if k is None else mx.array(k)
            c.values = None if v is None else mx.array(v)
            c.offset, c._idx = off, idx
        else:
            assert c.offset >= offset
            c.offset = offset


@dataclass
class TurnIO:
    """What one answer needs from the app. Set by the worker at key up."""
    emit_audio: Callable
    mark: Callable[[str], None]
    question: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: dict = field(default_factory=dict)


class RecallBackend:
    def __init__(self, adapter, store, tts=None, first_chunk: str = "phrase", trim_silence: bool = True,
                 history_max_tokens: int = 2000, k_parents: int = K_PARENTS, piece_dir: Optional[Path] = None,
                 screen_store=None, n_screens: int = 1):           # Andrew, 2026-11-11 (§5.13)
        self.a, self.store, self.tts = adapter, store, tts
        self.ss, self.n_screens = screen_store, n_screens
        self._tpl: dict = {}                            # (size, text) -> processor inputs for a blank image
        self._null: dict = {}                           # screen-turn length -> null-query vectors per layer
        self._stack = None                              # (store version, fids, {layer: (int8, scales, nulls)})
        self.lay = K.Layout(adapter)
        self.first_chunk, self.trim = first_chunk, trim_silence
        self.history_max_tokens, self.k_parents = history_max_tokens, k_parents
        self.priors = dict(PRIORS)
        self.frames: dict[int, object] = {}             # frame id -> image, set by the worker before frame()
        self.frames_lock = threading.Lock()
        self.screens: dict[int, K.Piece] = {}           # frame id -> stored screen piece (newest two kept)
        self.pieces: dict[str, dict] = {}               # page key -> {version, ids, piece, parents, cuts}
        self.history: list[tuple[str, str]] = []        # finished exchanges (question, answer)
        self.prefix_v = 0                               # matches the scheduler's prefix version
        self.prefix_cache = None
        self.prefix_n = 0                               # exchanges in the prefix
        self.hist_end = None                            # (offset, window snapshot) before the pages turn opens
        self.exact = None                               # exact cache: prefix + pages, built by upgrade()
        self.exact_v = -1
        self.exact_slots: tuple = ()
        self.bounds: list[tuple[int, dict]] = []        # (offset, window snapshot) where each exact page starts
        self.base = None                                # (key, cache): the assembled context before the question
        self.spec_state = None                                # {"key", "ids", "cov", "cache"}
        self.io: Optional[TurnIO] = None
        self.stats = {"builds": 0, "build_s": 0.0, "build_tokens": 0, "cut_pieces": 0, "window_bytes_on_disk": 0}
        self.dir = Path(piece_dir) if piece_dir else Path(tempfile.mkdtemp(prefix="lv-pieces-"))
        os.chmod(self.dir, 0o700)
        self._ids = lru_cache(maxsize=4096)(lambda s: tuple(self.a.text_ids(s)))
        a = self.a
        self.page_anchor = self._run(self._prefill(a.model.make_cache(), [a.tok.bos_token_id] +
                                                   list(self._ids(TURN_USER + HEAD))))
        anchor = self._run(self._prefill(a.model.make_cache(), [a.tok.bos_token_id] + list(self._ids(TURN_USER))))
        self.screen_pos = int(anchor[0].offset)
        self.screen_anchor = self._run(self._prefill(anchor, list(self._ids(TURN_END))))

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _run(gen):
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def prior(self, kind: str) -> float:
        """First estimate for a prefill kind "L<layers>x<tokens>"."""
        if kind.startswith("L") and "x" in kind:
            layers, n = map(int, kind[1:].split("x"))
            return n / TOKENS_PER_S * layers / self.lay.n + PIECE_OVERHEAD_S
        raise KeyError(kind)

    def _prefill(self, cache, ids, layers: int = 1):
        """ids appended to cache (mutated) in chunks of CHUNK tokens, `layers` text layers a piece."""
        per = -(-self.lay.n // layers)
        for s in range(0, len(ids), CHUNK):
            chunk = list(ids[s:s + CHUNK])
            kind = f"L{layers}x{-(-len(chunk) // 64) * 64}"
            cache = yield from steps(self.a.prefill_text_steps(cache, chunk, layers_per_piece=layers), [kind] * per)
        return cache

    def question_ids(self, text: str):
        return tuple(self.a.suffix_ids(text))

    def page_ids(self, slot) -> tuple:
        if kind(slot) == OPENER:
            return self._ids(TURN_USER + HEAD)
        key, _, n_lines = slot
        return self._ids(self.store.page_text(key, n_lines))

    def _window_path(self, key: str) -> Path:
        return self.dir / f"{abs(hash(key))}.safetensors"

    def _save_window(self, key: str, piece: K.Piece):
        arrays = {f"{i}.{j}": x for i, kv in piece.window.items() for j, x in enumerate(kv)}
        mx.save_safetensors(str(self._window_path(key)), arrays)
        os.chmod(self._window_path(key), 0o600)
        self.stats["window_bytes_on_disk"] = sum(f.stat().st_size for f in self.dir.iterdir())

    def _with_window(self, key: str) -> K.Piece:
        e = self.pieces[key]
        arrays = mx.load(str(self._window_path(key)))
        win = {}
        for name, x in arrays.items():
            i, j = map(int, name.split("."))
            win.setdefault(i, [None, None])[j] = x
        p = e["piece"]
        return K.Piece(p.n, p.pos, p.full, {i: tuple(kv) for i, kv in win.items()})

    # ------------------------------------------------------------ what the scheduler asks

    def buildable(self, slot) -> bool:
        """Screens and the opener are always computed exactly; pages and keying work are built in the background."""
        return kind(slot) in ("page", KEY)

    def has_piece(self, slot) -> bool:
        if kind(slot) == KEY:
            return self.ss is not None and self.ss.ready(slot[1])
        if kind(slot) != "page":
            return False
        key, version, n_lines = slot
        e = self.pieces.get(key)
        if e is None or e["version"] != version:
            return False
        if n_lines in e["cuts"]:
            return True
        ids = self.page_ids(slot)
        return ids == e["ids"][:len(ids)]

    def piece(self, slot) -> K.Piece:
        key, _, n_lines = slot
        e = self.pieces[key]
        if n_lines in e["cuts"]:
            return e["cuts"][n_lines]
        ids = self.page_ids(slot)
        assert ids == e["ids"][:len(ids)]
        return e["piece"] if len(ids) == len(e["ids"]) else e["piece"].head(len(ids))

    def prefetch(self, slots):
        pass                                            # full-layer entries stay in memory

    def read(self, fid: int):
        with self.frames_lock:
            img = self.frames.get(fid)
            for f in [f for f in self.frames if f <= fid]:
                del self.frames[f]

        def g():
            enc = yield from steps(self.a.encode_steps(img), VISION_KINDS)
            c, _ = yield from steps(self.a.screen_steps(self.screen_anchor, enc, layers_per_piece=2),
                                    ["embed"] + ["text"] * (-(-self.lay.n // 2)))
            self.screens[fid] = K.take(self.lay, c, self.screen_pos, window=True)
            for f in sorted(self.screens)[:-2]:
                del self.screens[f]
        return g()

    def build(self, slot):
        """The page's stored piece at its current version (and, for a cut that isn't a prefix of the whole
        page's tokens, that cut too), computed after its parents' stored pieces. For (KEY, fid): key a kept screen."""
        if kind(slot) == KEY:
            return self._key_screen(slot[1])
        key = slot[0]

        def g():
            yield "plan_build"                          # CPU: version, parents (BM25 over the store)
            with self.store.lock:
                p = self.store.page(key)                # a page, or a kept screen's view (§5.25)
                if p is None:
                    return
                v = self.store.version[key]
                whole = (key, v, len(p.blocks))
                cut_slot = (key, v, slot[2]) if slot[1] == v else whole
                query = p.title + "\n" + "\n".join(p.blocks)
                ranked = [q for q, _ in self.store.m.rank_pages(query)
                          if q.url != key and q.first_s < p.first_s and q.url in self.pieces]
                parents = sorted(ranked[:self.k_parents], key=lambda q: q.first_s)
                parent_keys = [q.url for q in parents]
            e = self.pieces.get(key)
            if e is None or e["version"] != v:
                t0 = time.perf_counter()
                yield "assemble"
                c = K.append(self.lay, self.page_anchor, [self._with_window(u) for u in parent_keys])
                pos = int(c[0].offset)
                ids = self.page_ids(whole)
                c = yield from self._prefill(c, ids)
                piece = K.take(self.lay, c, pos, window=True)
                self._save_window(key, piece)
                piece.window = None
                self.pieces[key] = e = {"version": v, "ids": ids, "piece": piece, "parents": parent_keys, "cuts": {}}
                self.stats["builds"] += 1
                self.stats["build_s"] += time.perf_counter() - t0
                self.stats["build_tokens"] += len(ids)
                del c
            if cut_slot != whole and not self.has_piece(cut_slot):
                yield "assemble"
                c = K.append(self.lay, self.page_anchor, [self._with_window(u) for u in e["parents"]
                                                          if u in self.pieces])
                pos = int(c[0].offset)
                c = yield from self._prefill(c, self.page_ids(cut_slot))
                e["cuts"][cut_slot[2]] = K.take(self.lay, c, pos, window=False)
                self.stats["cut_pieces"] += 1
                del c
        return g()

    # ------------------------------------------------------------ kept screens (§5.12)

    def _inputs_like(self, size, text: str) -> dict:
        """Processor inputs for `text` (with one "<|image|>") as if the image were `size`: run on a blank image of that
        size, so the token ids are exactly those of a real frame; the pixels are dropped (features come from storage)."""
        key = (tuple(size), text)
        if key not in self._tpl:
            from PIL import Image
            inp = dict(self.a._inputs(Image.new("RGB", tuple(size)), text))
            inp["pixel_values"] = mx.zeros((1,), dtype=mx.float16)      # a placeholder: features are given
            self._tpl[key] = inp
            if len(self._tpl) > 256:
                self._tpl.pop(next(iter(self._tpl)))
        return self._tpl[key]

    def _turn_text(self, s: Screen) -> str:
        """A kept screen's turn: a closed user turn, before the pages."""
        from recall import hm
        t = self.store.at(s.t)
        head = KEPT_HEAD.format(time=f"{hm(t)} {t.strftime('%p')}", title=s.title) + "\n<|image|>"
        return f"{TURN_USER}{head}{TURN_END}"

    def _screen_into(self, c, fid: int):
        """A kept screen's turn computed exactly after cache c, from its stored features. Returns the new cache."""
        yield "screen_prep"
        with self.ss.lock:
            s = self.ss.screens[fid]
        inp = self._inputs_like(s.size, self._turn_text(s))
        feats = mx.array(self.ss.load_features(fid))
        out, _ = yield from steps(self.a.screen_steps(c, (inp, feats), layers_per_piece=2),
                                  ["embed"] + ["text"] * (-(-self.lay.n // 2)))
        return out

    def _queries(self, text: str, T: int) -> dict:
        """The question's queries at KEY_LAYERS (post-RoPE, as attention sees them), re-rotated to sit right after a
        screen turn of T tokens, as in §12.17's measurement. {layer: (queries [heads, Lq, D] float16, scale)}."""
        ids = [self.a.tok.bos_token_id] + list(self._ids(" " + text))
        _CAPTURE["on"], _CAPTURE["calls"] = True, []
        try:
            self._run(self.a.prefill_text_steps(self.a.model.make_cache(), ids, layers_per_piece=self.lay.n))
        finally:
            _CAPTURE["on"] = False
        out = {}
        for L in KEY_LAYERS:
            q, scale = _CAPTURE["calls"][L]
            q = K.rotate(self.lay.ropes[L], q[:, :, 1:, :], T - 1)
            out[L] = (q[0].astype(mx.float16), float(scale))
        _CAPTURE["calls"] = []
        return out

    @staticmethod
    def _score(q, scale, k8, ks) -> mx.array:
        """Mean over question tokens and query heads of the max over image tokens of q.k * scale, for a batch of
        frames' int8 keys [n, kv heads, n_img, D] with scales [n, kv heads, n_img, 1]."""
        hk = k8.shape[1]
        g = q.reshape(hk, q.shape[0] // hk, q.shape[1], q.shape[2])                       # [kv, group, Lq, D]
        k = k8.astype(mx.float16) * ks                                                     # [n, kv, n_img, D]
        s = mx.einsum("hgqd,nhkd->nhgqk", g, k).astype(mx.float32) * scale
        return s.max(axis=-1).mean(axis=(1, 2, 3))

    def _key_screen(self, fid: int):
        """Key a kept screen (quiet-time work): its features (from the Neural Engine, or encoded here if it couldn't),
        saved to disk; its screen turn through text layers up to the last key layer; image-token keys at KEY_LAYERS,
        int8; and its score for the content-free query."""
        def g():
            yield "plan_build"
            p = self.ss.take_pending(fid) if self.ss is not None else None
            if p is None:
                return
            meta, a = p["meta"], self.a
            inp = self._inputs_like(meta["size"], a.screen_turn_prompt())
            if p["tokens"] is not None:
                _, feats = a.feats_from_tokens(inp, p["tokens"])
            else:
                real = a._inputs(p["image"], a.screen_turn_prompt())
                _, feats = yield from steps(a.encode_steps(p["image"], inp=real), ["patch"] + ["vision"] * 16 + ["pool"])
            path = self.ss.save_features(fid, np.asarray(feats.astype(mx.float16)))
            ids = np.asarray(inp["input_ids"])[0]
            img_id = getattr(a.model.config, "image_token_id", None)
            pos = mx.array((1 + np.flatnonzero(ids == img_id)).astype(np.int32))      # screen_steps(None) adds <bos>
            T = len(ids) + 1
            gen = a.screen_steps(None, (inp, feats), layers_per_piece=1)
            yield "embed"
            next(gen)
            for _ in range(max(KEY_LAYERS) + 1):
                yield "text1"
                next(gen)
            cache = gen.gi_frame.f_locals["cache"]
            keys = {}
            for L in KEY_LAYERS:
                k = np.asarray(cache[L].state[0][0][:, pos, :].astype(mx.float32))
                keys[L] = quantize_int8(k)
            gen.close()
            del cache
            yield "check"
            if T not in self._null:
                self._null[T] = self._queries(NULL_Q, T)
            null = {}
            for L in KEY_LAYERS:
                q, scale = self._null[T][L]
                k8, ks = keys[L]
                null[L] = float(self._score(q, scale, mx.array(k8)[None], mx.array(ks)[None]).item())
            self.ss.add(Screen(fid, meta["t"], meta["key"], meta["title"], meta["lines"], keys, null, path,
                               size=tuple(meta["size"]), T=T))
            self.ss.expire(meta["t"])
            self.store.set_view(fid, self.ss.screens[fid])          # §5.25: its view text, a memory slot
            for vk in [k for k in self.store.views if int(k[5:]) not in self.ss.screens]:
                self.store.drop_view(int(vk[5:]))
            mx.clear_cache()
        return g()

    def find_screens(self, text: str) -> list[int]:
        """The kept screens to put in the context for this (partial) question: §12.17's combination of text (frame BM25)
        and Gemma's keys (calibrated, z-normalised, summed over KEY_LAYERS), better rank first; top n_screens."""
        if self.ss is None or self.n_screens <= 0:
            return []
        with self.store.lock:                           # the visit still on screen: the current screen supersedes it
            cur = (self.store.cur.key, self.store.cur.t0) if self.store.cur is not None else None
        fids = self.ss.ids(exclude_visit=cur)
        if not fids:
            return []
        with self.ss.lock:
            screens = [self.ss.screens[f] for f in fids]
        if self._stack is None or self._stack[0] != (self.ss.version, tuple(fids)):
            stack = {}
            for L in KEY_LAYERS:
                stack[L] = (mx.array(np.stack([s.keys[L][0] for s in screens])),
                            mx.array(np.stack([s.keys[L][1] for s in screens])),
                            np.array([s.null[L] for s in screens]))
            self._stack = ((self.ss.version, tuple(fids)), stack)
        stack = self._stack[1]
        T = screens[0].T
        qv = self._queries(text, T)
        total = np.zeros(len(fids))
        for L in KEY_LAYERS:
            q, scale = qv[L]
            k8, ks, null = stack[L]
            raw = np.concatenate([np.asarray(self._score(q, scale, k8[i:i + 32], ks[i:i + 32]))
                                  for i in range(0, len(fids), 32)])
            cal = raw - null
            total += (cal - cal.mean()) / (cal.std() + 1e-9)
        # §5.21 / §5.23 (adopted 2026-11-12): keys and the unit's text, each z-normalised, summed with equal weight.
        # The earlier better-rank fuse let the keys alone pick at 1 screen (text never won a disagreement).
        bt = self.ss.text_scores(text, fids)
        zt = (bt - bt.mean()) / (bt.std() + 1e-9)
        self.last_ranked = [fids[i] for i in np.argsort(-(total + zt), kind="stable")][:max(self.n_screens, 3)]
        top = self.last_ranked[:self.n_screens]
        return sorted(top, key=lambda f: self.ss.screens[f].t)

    def _history_ids(self, n: int) -> list[int]:
        """The first n exchanges, oldest dropped until they fit history_max_tokens."""
        ex = [list(self.a.exchange_ids(q, ans, older=True)) for q, ans in self.history[:n]]
        while ex and sum(map(len, ex)) > self.history_max_tokens:
            ex.pop(0)
        return [t for x in ex for t in x]

    def prefix(self, n_exchanges: int):
        timeline = self.store.timeline()

        def g():
            a = self.a
            c = yield from self._prefill(a.model.make_cache(), [a.tok.bos_token_id] +
                                         list(self._ids(TURN_USER + timeline + TURN_END)) + self._history_ids(n_exchanges))
            self.prefix_cache, self.prefix_n = c, n_exchanges   # the pages turn opens later (OPENER), after screens
            self.prefix_v += 1
        return g()

    def append(self, i: int):
        def g():
            yield "check"
            assert i == self.prefix_n < len(self.history), (i, self.prefix_n, len(self.history))
            q, ans = self.history[i]
            yield from self._prefill(self.prefix_cache, list(self.a.exchange_ids(q, ans, older=True)))
            self.prefix_n = i + 1
            self.prefix_v += 1
        return g()

    def history_over_cap(self) -> bool:
        return sum(len(self.a.exchange_ids(q, ans, older=True)) for q, ans in self.history) > self.history_max_tokens

    def _exact_at(self, slots, i: int):
        """The exact cache holding the prefix and slots[:i], cut back to there."""
        if self.exact is None or self.exact_v != self.prefix_v:
            self.exact, self.exact_v, self.exact_slots, self.bounds = clone_cache(self.prefix_cache), self.prefix_v, (), []
        assert self.exact_slots[:i] == tuple(slots[:i]), (self.exact_slots, slots, i)
        if i < len(self.bounds):
            cut(self.lay, self.exact, *self.bounds[i])
            self.bounds = self.bounds[:i]
        self.exact_slots = self.exact_slots[:i]
        return self.exact

    def upgrade(self, slots, i: int):
        def g():
            yield "check"
            c = self._exact_at(slots, i)
            self.bounds.append((int(c[0].offset), window_snapshot(self.lay, c)))
            if kind(slots[i]) == SCREEN:
                self.exact = yield from self._screen_into(c, slots[i][1])
            else:
                yield from self._prefill(c, self.page_ids(slots[i]))
            self.exact_slots = tuple(slots[:i + 1])
        return g()

    def _assemble(self, key, compute_missing: bool, exact_all: bool = False):
        """The context before the question for `key` = (prefix version, slots, frontier, screen frame), as a new
        cache. Generator: pages with no stored piece are computed in place (compute_missing) or not allowed."""
        v, slots, frontier, fid = key
        assert v == self.prefix_v, (v, self.prefix_v)
        if frontier:
            assert self.exact_v == v and self.exact_slots[:frontier] == tuple(slots[:frontier])
            assert len(self.bounds) >= frontier
            c = clone_cache(self.exact)
            if frontier < len(self.bounds):             # a page after the frontier was started: cut it off
                cut(self.lay, c, *self.bounds[frontier])
        else:
            c = clone_cache(self.prefix_cache)
        # Stored page pieces carry no window-layer entries, so nothing may be computed right after one (its window
        # attention would see stale entries; kv_reuse.append refuses). Every slot up to the last one that must be
        # computed is therefore computed exactly, in order; only the slots after it come from stored pieces, and the
        # current screen piece (which has window entries) refills the window at the end.
        rest = list(slots[frontier:])
        last = max((i for i, sl in enumerate(rest) if not self.has_piece(sl)), default=-1)
        if exact_all:                                   # every slot computed (§5.14's comparison)
            last = len(rest) - 1
        assert compute_missing or last < 0, rest[last]
        for slot in rest[:last + 1]:
            if kind(slot) == SCREEN:
                c = yield from self._screen_into(c, slot[1])
            else:
                c = yield from self._prefill(c, self.page_ids(slot))
        return K.append(self.lay, c, [self.piece(sl) for sl in rest[last + 1:]] + [self.screens[fid]], copy=False)

    def spec(self, key, ids, have):
        """The question so far over the context `key`, leaving its last token for the answer to run."""
        def g():
            yield "assemble"
            if self.base is None or self.base[0] != key:
                base = yield from self._assemble(key, compute_missing=False)
                self.base, self.spec_state = (key, base), None
            s = self.spec_state
            cov = 0
            if s is not None and s["key"] == key and list(ids[:s["cov"]]) == list(s["ids"][:s["cov"]]):
                cov = s["cov"]
            else:
                s = self.spec_state = {"key": key, "ids": (), "cov": 0, "cache": clone_cache(self.base[1])}
            target = len(ids) - 1
            if target > cov:
                yield from self._prefill(s["cache"], list(ids[cov:target]), layers=4)
            s.update(ids=tuple(ids), cov=max(cov, target))
        return g()

    def answer(self, plan):
        def g():
            io = self.io
            r = io.result
            yield "assemble"
            key, words = plan["key"], list(plan["words"])
            s = self.spec_state
            if (s is not None and s["key"] == key and s["cov"] > 0 and words[:s["cov"]] == list(s["ids"][:s["cov"]])
                    and s["cov"] < len(words)):
                cache, ids, r["context"] = s["cache"], words[s["cov"]:], "question prefilled"
            elif self.base is not None and self.base[0] == key:
                cache, ids, r["context"] = clone_cache(self.base[1]), words, "assembled before key up"
            else:
                cache = yield from self._assemble(key, compute_missing=True)
                ids, r["context"] = words, "assembled at release"
            self.spec_state = self.base = None
            r.update(context_tokens=int(cache[0].offset), question_tokens=len(words), prefilled_at_release=len(ids),
                     pages=len(plan["pages"]), exact_pages=plan["frontier"], missing=len(plan["missing"]),
                     screen_frame=plan["screen"])
            io.mark("restore_done")
            text = yield from self._speak(cache, ids, io, plan["on_first_token"])
            done = bool(text.strip()) and not io.cancel.is_set()
            if done:
                self.history.append((io.question, text.strip()))
            r.update(answer=text.strip(), stop=self.a.last_stop if not io.cancel.is_set() else "cancel",
                     cancelled=io.cancel.is_set())
            io.mark("turn_complete")
            return {"exchange": done, "rebuild": done and self.history_over_cap()}
        return g()

    def _speak(self, cache, ids, io: TurnIO, on_first_token):
        """Decode, one token a piece, handing phrases and sentences to TTS as they complete (GpuWorker._answer's
        rules: an early first phrase, then whole sentences)."""
        r = io.result
        text, buf, chunks = "", "", 0

        def first():
            io.mark("first_token")
            on_first_token()

        def speak(sentence, mid_sentence=False):
            nonlocal chunks
            chunks += 1
            if chunks == 1:
                io.mark("first_sentence_ready")
                r["first_chunk_text"] = sentence
            if self.tts is None:
                return
            marked = False
            for audio in self.tts.synth(sentence, cancel=io.cancel, trim=self.trim,
                                        tail_keep_ms=40 if mid_sentence else 120):
                if not marked and chunks == 1:
                    io.mark("tts_first_chunk")
                    marked = True
                io.emit_audio(audio, self.tts.sample_rate)
                if io.cancel.is_set():
                    return

        gen = self.a.generate(cache, ids, cancel=io.cancel, on_first_token=first)
        kind = "answer_prefill"
        while True:
            yield kind
            kind = "decode"
            try:
                delta = next(gen)
            except StopIteration:
                break
            text += delta
            buf += delta
            done, buf = split_complete_sentences(buf)
            if self.first_chunk == "phrase" and chunks == 0 and not done:
                ph = split_first_phrase(buf)
                if ph is not None:
                    speak(ph[0], mid_sentence=True)
                    buf = ph[1]
            for sentence in done:
                if io.cancel.is_set():
                    break
                speak(sentence)
            if io.cancel.is_set():
                break
        if buf.strip() and not io.cancel.is_set():
            speak(buf.strip())
        r["chunks_spoken"] = chunks
        return text
