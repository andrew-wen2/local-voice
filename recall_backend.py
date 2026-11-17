"""recall_backend.py — the scheduler's GPU work on real KV caches. MLX, so every
method runs on the GPU worker thread.

Context layout (token ids):
  prefix   <bos> <|turn>user\\n timeline <turn|>\\n  [finished exchanges, each a user + model turn]  <|turn>user\\n
           pages header                                                            exact; rebuilt or appended to
  pages    "\\n\\n" + time header + page text, one per retrieved page, oldest first:
           the first `frontier` computed exactly after the prefix (upgrade), the rest stored graph pieces
  screen   <turn|>\\n + the screen turn (image + instructions + "Question:")        stored piece, its own read
  question the question and the model turn's opening                              prefilled before key up if
                                                                                    possible, else at release
The timeline is closed in its own turn (it used to share one with the pages) so that a finished exchange can be
appended after it without recomputing the timeline: that change is not measured yet.

Stored page pieces (graph): a page is computed after the stored pieces of its K_PARENTS most similar pages
first seen before it (the retriever's BM25, the page's title and text as the query), loaded and re-rotated, after
the anchor <bos> <|turn>user\\n pages header. The piece (full-attention entries, 16 KB a token, and its last W window
entries, needed only when it is a parent in a later build) goes to a private folder on disk, deleted when the app
quits; memory holds only its token ids and parents, so memory doesn't grow with the hour. A
question loads the pieces its context uses, bounded by the context cap. A page's piece is rebuilt when its version changes (a visit of it ended: its text and time
header changed).

Kept screens (visual memory): retrieved past screens come first after the prefix, each as its own user turn
"(Earlier screen) [time] title" + its image tokens, computed exactly from its stored vision features (screen_store.py)
while you speak, never as a stored KV piece; then the pages turn opens (a fixed opener, computed exactly), then pages,
then the current screen. Screens are found by text (frame BM25) and Gemma's own keys (layers 11 + 17, calibrated by
the content-free query), combined by better rank. A kept screen's keys are computed in quiet time.
Slot kinds: a page (key, version, lines); a screen (SCREEN, frame id); the opener (OPENER,); keying work (KEY, frame id).

Pieces follow the scheduler's protocol: a generator announces the kind of its next piece, then runs it. Prefill
runs in chunks of CHUNK tokens, one text layer a piece, so piece time stays short whatever the page size; such
kinds are named by size ("L1x512"), and `prior` gives their first estimate from the measured prefill rate.
"""
from __future__ import annotations

import os
import queue
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

IMAGE = "<|image|>"
# The answer's first chunk: the first sentence split early at punctuation after >= 3 words, else the whole sentence
# (a split before a break word such as "is" was faster but left an audible seam).
FIRST_MIN_WORDS = 3
READ_FEATS = 4                   # image features of the last reads kept for keying (~5.5 MB each)
MAX_BRANCHES = 2                 # exact caches kept: the current context and the runner-up (~0.1 GB each, estimate)

SCREEN, OPENER, KEY = "§screen", "§opener", "§key"
# A kept screen's block (2026-11-15): (BLOCK, fid, view slot) = its screen turn, the opener and its memory view,
# one slot. Computed exactly at question time like any slot, or loaded: stored once in quiet time after the prefix
# of then and spliced with keys re-rotated (measured as accurate as computing it exactly).
BLOCK = "§block"
KEPT_HEAD = "(Earlier screen from {time}, no longer on screen) {title}"       # a kept screen's label
NULL_Q = "N/A"                    # ICR's content-free query

K_PARENTS = 4                 # a page is computed after its 4 most similar pages
CHUNK = 512                   # tokens per prefill chunk
TOKENS_PER_S = 2200           # prefill rate, plugged in; first estimate only, then measured per kind
PIECE_OVERHEAD_S = 0.003      # pausing costs ~3 ms a piece
HEAD = (PAGES_HEAD + HINT).rstrip("\n")
VISION_KINDS = ["prep", "patch"] + ["vision"] * 16 + ["pool"]
PRIORS = {                    # seconds; measured for the read, estimates for the rest
    "prep": 0.061, "patch": 0.088, "vision": 0.080, "pool": 0.010, "embed": 0.008, "text": 0.042,
    "check": 0.0005, "plan_build": 0.010, "assemble": 0.015, "answer_prefill": 0.060, "decode": 0.026,
    "text1": 0.021, "screen_prep": 0.004, "append": 0.030,     # features from disk + the cached token layout (2-3 ms measured, 2026-11-13)
    "rq": 0.006, "rs": 0.005, "rprep": 0.002, "rlay": 0.003,  # retrieval pieces: 3 layers, 32 screens
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


SPEC_SNAPS = 3                      # question-prefill stops kept restorable (window layers, ~20 MB each)
SNAP_BACK = 4                       # a stop this many tokens before the question text ends (its last word or two)
QUERY_LAYERS = 3                    # text layers a piece in the retrieval's question pass (~5 ms standalone, fits TAIL)
KEY_CHUNK = 32                      # kept screens a key-scoring piece
_dequant = mx.compile(lambda k8, ks: k8.astype(mx.float16) * ks)


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
    tts_thread: Optional[threading.Thread] = None       # speaks the answer's chunks; `done` waits for it
    warmup: bool = False                                # a startup warm-up: not kept in the conversation
    on_say: Optional[Callable[[str], None]] = None      # each spoken chunk's text, as its first audio goes out (captions)


class RecallBackend:
    def __init__(self, adapter, store, tts=None, trim_silence: bool = True,
                 history_max_tokens: int = 2000, k_parents: int = K_PARENTS, piece_dir: Optional[Path] = None,
                 screen_store=None, n_screens: int = 1):
        self.a, self.store, self.tts = adapter, store, tts
        self.ss, self.n_screens = screen_store, n_screens
        from concurrent.futures import ThreadPoolExecutor
        self._io = ThreadPoolExecutor(1, thread_name_prefix="key-reads")   # retrieval's key reads, off the GPU loop
        self._pending_io = None
        self.blocks: dict = {}                          # (BLOCK, fid, view slot) -> path of its stored piece
        self._tpl: dict = {}                            # (size, budget) -> the image's token block (ids, types, mask)
        self._null: dict = {}                           # screen-turn length -> null-query vectors per layer
        self.lay = K.Layout(adapter)
        self.trim = trim_silence
        self.history_max_tokens, self.k_parents = history_max_tokens, k_parents
        self.priors = dict(PRIORS)
        self.frames: dict[int, object] = {}             # frame id -> image, set by the worker before frame()
        self.frames_lock = threading.Lock()
        self.screens: dict[int, K.Piece] = {}           # frame id -> stored screen piece (newest two kept)
        self.read_feats: dict[int, mx.array] = {}       # frame id -> image features of the newest reads (READ_FEATS)
        self.pieces: dict[str, dict] = {}               # page key -> {version, ids, path, parents, cuts: {lines: path}}
        self.history: list[tuple[str, str]] = []        # finished exchanges (question, answer)
        self.prefix_v = 0                               # matches the scheduler's prefix version
        self.prefix_cache = None
        self.prefix_n = 0                               # exchanges in the prefix
        self.hist_end = None                            # (offset, window snapshot) before the pages turn opens
        # Exact caches (prefix + slots computed in order), built by upgrade(): the current context's and, while the GPU
        # would otherwise idle during speech, the runner-up's, so a late switch finds it ready. Each:
        # {"cache", "slots", "bounds": [(offset, window snapshot) where each slot starts], "busy"}; most recent first,
        # at most MAX_BRANCHES, all on prefix version branch_v.
        self.branches: list[dict] = []
        self.branch_v = -1
        # The assembled context for the current key, with the question so far prefilled into it in place (no
        # second copy for the question): {"key", "cache", "end": (offset, window snapshot) where the
        # context ends, "ids", "cov": question tokens in the cache, "dirty": a prefill was cut short}.
        self.base: Optional[dict] = None
        self.io: Optional[TurnIO] = None
        self.stats = {"builds": 0, "build_s": 0.0, "build_tokens": 0, "cut_pieces": 0, "piece_bytes_on_disk": 0}
        self.dir = Path(piece_dir) if piece_dir else Path(tempfile.mkdtemp(prefix="lv-pieces-"))
        os.chmod(self.dir, 0o700)
        self._ids = lru_cache(maxsize=256)(lambda s: tuple(self.a.text_ids(s)))     # bounded: memory stays flat
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

    def _save_piece(self, name: str, piece: K.Piece) -> Path:
        """A piece (full-layer entries, and window entries if it has them) to the private folder."""
        arrays = {f"f.{i}.{j}": x for i, kv in piece.full.items() for j, x in enumerate(kv)}
        arrays.update({f"w.{i}.{j}": x for i, kv in (piece.window or {}).items() for j, x in enumerate(kv)})
        path = self.dir / f"{abs(hash(name))}.safetensors"
        old = path.stat().st_size if path.exists() else 0
        mx.save_safetensors(str(path), arrays, metadata={"n": str(piece.n), "pos": str(piece.pos)})
        os.chmod(path, 0o600)
        self.stats["piece_bytes_on_disk"] += path.stat().st_size - old
        return path

    def _load_piece(self, path: Path, window: bool) -> K.Piece:
        arrays, meta = mx.load(str(path), return_metadata=True)
        full, win = {}, {}
        for name, x in arrays.items():
            part, i, j = name.split(".")
            if part == "f" or window:
                (full if part == "f" else win).setdefault(int(i), [None, None])[int(j)] = x
        return K.Piece(int(meta["n"]), int(meta["pos"]), {i: tuple(kv) for i, kv in full.items()},
                       {i: tuple(kv) for i, kv in win.items()} if window else None)

    def _with_window(self, key: str) -> K.Piece:
        return self._load_piece(self.pieces[key]["path"], window=True)

    def _build_block(self, slot):
        """A kept screen's block after the prefix as it is now, its full-attention layers stored (the current screen's
        piece refills the window layers before the question). Replaces the screen's older block."""
        def g():
            yield "check"
            if self.prefix_cache is None or self.ss is None or not self.ss.ready(slot[1]):
                return
            c = clone_cache(self.prefix_cache)
            n0 = int(c[0].offset)
            c = yield from self._slot_into(c, slot)
            yield "append"
            p = K.take(self.lay, c, n0, window=False)
            del c
            for old in [b for b in self.blocks if b[1] == slot[1]]:
                self._forget_block(old)
            self.blocks[slot] = self._save_piece(f"block:{slot}", p)
            del p
            mx.clear_cache()
        return g()

    def _forget_block(self, slot):
        path = self.blocks.pop(slot, None)
        if path is not None and path.exists():
            self.stats["piece_bytes_on_disk"] -= path.stat().st_size
            path.unlink()

    def _forget(self, key: str):
        """Drop a page's stored piece and its cuts (a view whose screen expired)."""
        e = self.pieces.pop(key, None)
        for path in ([e["path"]] + list(e["cuts"].values())) if e else []:
            if path.exists():
                self.stats["piece_bytes_on_disk"] -= path.stat().st_size
                path.unlink()

    @staticmethod
    def _is_prefix(ids, stored: np.ndarray) -> bool:
        return len(ids) <= len(stored) and np.array_equal(np.asarray(ids, dtype=np.int32), stored[:len(ids)])

    # ------------------------------------------------------------ what the scheduler asks

    def buildable(self, slot) -> bool:
        """Screens and the opener are always computed exactly; pages, keying work and kept screens' blocks are built
        in the background."""
        return kind(slot) in ("page", KEY, BLOCK)

    def background_only(self, slot) -> bool:
        """A block is quiet-time work even when a turn needs it: during the turn its exact computation (an upgrade)
        is what helps, not a second, stored copy of the same tokens."""
        return kind(slot) == BLOCK

    def exact_wanted(self, slot) -> bool:
        """A stored block is used as it is: computing it exactly as well gained nothing measurable and would
        restart a question prefill already under way over the stored one."""
        return not (kind(slot) == BLOCK and slot in self.blocks)

    def has_piece(self, slot) -> bool:
        if kind(slot) == KEY:
            return self.ss is not None and self.ss.ready(slot[1])
        if kind(slot) == BLOCK:
            return slot in self.blocks
        if kind(slot) != "page":
            return False
        key, version, n_lines = slot
        e = self.pieces.get(key)
        if e is None or e["version"] != version:
            return False
        if n_lines in e["cuts"]:
            return True
        return self._is_prefix(self.page_ids(slot), e["ids"])

    def piece(self, slot) -> K.Piece:
        if kind(slot) == BLOCK:
            return self._load_piece(self.blocks[slot], window=False)
        key, _, n_lines = slot
        e = self.pieces[key]
        if n_lines in e["cuts"]:
            return self._load_piece(e["cuts"][n_lines], window=False)
        ids = self.page_ids(slot)
        assert self._is_prefix(ids, e["ids"])
        p = self._load_piece(e["path"], window=False)
        return p if len(ids) == len(e["ids"]) else p.head(len(ids))

    def prefetch(self, slots):
        pass                                            # pieces load from disk when a context is assembled

    def read(self, fid: int):
        with self.frames_lock:
            img = self.frames.get(fid)
            for f in [f for f in self.frames if f <= fid]:
                del self.frames[f]

        def g():
            enc = yield from steps(self.a.encode_steps(img), VISION_KINDS)
            self.read_feats[fid] = enc[1]               # a kept frame that was read is keyed from these, not re-encoded
            for f in sorted(self.read_feats)[:-READ_FEATS]:
                del self.read_feats[f]
            # one text layer a piece: half the temporary memory of two (the read's peak)
            c, _ = yield from steps(self.a.screen_steps(self.screen_anchor, enc, layers_per_piece=1),
                                    ["embed"] + ["text1"] * self.lay.n)
            self.screens[fid] = K.take(self.lay, c, self.screen_pos, window=True)
            for f in sorted(self.screens)[:-2]:
                del self.screens[f]
        return g()

    def build(self, slot):
        """The page's stored piece at its current version (and, for a cut that isn't a prefix of the whole
        page's tokens, that cut too), computed after its parents' stored pieces. For (KEY, fid): key a kept screen."""
        if kind(slot) == KEY:
            return self._key_screen(slot[1])
        if kind(slot) == BLOCK:
            return self._build_block(slot)
        key = slot[0]

        def g():
            yield "plan_build"                          # CPU: version, parents (BM25 over the store)
            with self.store.lock:
                p = self.store.page(key)                # a page, or a kept screen's view
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
                if key in self.pieces:                  # the old version and its cuts (same name: overwritten)
                    self._forget(key)
                self.pieces[key] = e = {"version": v, "ids": np.asarray(ids, dtype=np.int32),
                                        "path": self._save_piece(key, piece), "parents": parent_keys, "cuts": {}}
                del piece
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
                e["cuts"][cut_slot[2]] = self._save_piece(f"{key}#{cut_slot[2]}", K.take(self.lay, c, pos, window=False))
                self.stats["cut_pieces"] += 1
                del c
        return g()

    # ------------------------------------------------------------ kept screens

    def _inputs_like(self, size, text: str) -> dict:
        """Processor inputs for `text` (with one "<|image|>") as if the image were `size`; the pixels are left out
        (features come from storage). The image's token block depends only on the size (and the budget), so the
        processor runs once per size, on a blank image; the text around it is tokenized on its own, which gives the
        processor's exact ids because "<|image|>" is a special token (nothing merges across it). Running the processor
        per screen took ~65-77 ms, over the 50 ms piece limit while you speak."""
        key = (tuple(size), getattr(self.a, "budget", None))
        if key not in self._tpl:
            from PIL import Image
            inp = self.a._inputs(Image.new("RGB", tuple(size)), IMAGE)
            self._tpl[key] = {k: np.asarray(v) for k, v in inp.items() if k != "pixel_values"}
        blk = self._tpl[key]
        before, after = text.split(IMAGE)
        a, b = (np.asarray([self.a.text_ids(t)], dtype=blk["input_ids"].dtype) for t in (before, after))
        out = {}
        for k, v in blk.items():
            fill = 1 if k == "attention_mask" else 0                     # text tokens: attended, not image
            out[k] = mx.array(np.concatenate([np.full_like(a, fill, dtype=v.dtype), v, np.full_like(b, fill, dtype=v.dtype)]
                                             if k != "input_ids" else [a, v, b], axis=1))
        out["pixel_values"] = mx.zeros((1,), dtype=mx.float16)          # a placeholder: features are given
        return out

    def _turn_text(self, s: Screen) -> str:
        """A kept screen's turn: a closed user turn, before the pages."""
        from recall import hm
        t = self.store.at(s.t)
        head = KEPT_HEAD.format(time=f"{hm(t)} {t.strftime('%p')}", title=s.title) + "\n<|image|>"
        return f"{TURN_USER}{head}{TURN_END}"

    def _slot_into(self, c, slot):
        """One slot computed exactly after cache c. Returns the cache (a new one after a screen turn)."""
        if kind(slot) == SCREEN:
            return (yield from self._screen_into(c, slot[1]))
        if kind(slot) == BLOCK:
            c = yield from self._screen_into(c, slot[1])
            c = yield from self._prefill(c, self.page_ids((OPENER,)))
            return (yield from self._prefill(c, self.page_ids(slot[2])))
        return (yield from self._prefill(c, self.page_ids(slot)))

    def _screen_into(self, c, fid: int):
        """A kept screen's turn computed exactly after cache c, from its stored features. Returns the new cache."""
        yield "screen_prep"
        with self.ss.lock:
            s = self.ss.screens[fid]
        inp = self._inputs_like(s.size, self._turn_text(s))
        feats = mx.array(self.ss.load_features(fid))
        # One text layer a piece (~22 ms): two-layer pieces took ~42-45 ms, so close to the 50 ms limit while you speak
        # that their estimate drifted over it and they stopped running during speech.
        out, _ = yield from steps(self.a.screen_steps(c, (inp, feats), layers_per_piece=1),
                                  ["embed"] + ["text1"] * self.lay.n)
        return out

    def _queries(self, text: str, T: int) -> dict:
        return self._run(self._queries_steps(text, T))

    def _queries_steps(self, text: str, T: int):
        """The question's queries at KEY_LAYERS (post-RoPE, as attention sees them), re-rotated to sit right after a
        screen turn of T tokens, as when the keys were measured. {layer: (queries [heads, Lq, D] float16, scale)}.
        Generator in pieces of QUERY_LAYERS text layers ("rq"). Only the layers up to the last key layer run: the
        queries there don't depend on later layers (18 of 24 layers, same values). The capture is
        on only during this pass's own pieces, so pieces of other jobs in between aren't captured."""
        ids = [self.a.tok.bos_token_id] + list(self._ids(" " + text))
        top = max(KEY_LAYERS) + 1
        gen = self.a.prefill_text_steps(self.a.model.make_cache(), ids, layers_per_piece=QUERY_LAYERS, n_layers=top)
        calls: list = []
        for _ in range(-(-top // QUERY_LAYERS)):
            yield "rq"
            _CAPTURE["on"], _CAPTURE["calls"] = True, calls
            try:
                next(gen)
            finally:
                _CAPTURE["on"], _CAPTURE["calls"] = False, []
        gen.close()
        out = {}
        for L in KEY_LAYERS:
            q, scale = calls[L]
            q = K.rotate(self.lay.ropes[L], q[:, :, 1:, :], T - 1)
            out[L] = (q[0].astype(mx.float16), float(scale))
        return out

    @staticmethod
    def _score(q, scale, k8, ks) -> mx.array:
        """Mean over question tokens and query heads of the max over image tokens of q.k * scale, for a batch of
        frames' int8 keys [n, kv heads, n_img, D] with scales [n, kv heads, n_img, 1].
        Same values bit for bit as the einsum it replaced, ~3-4x faster: one batched matmul per kv
        head group, the int8 -> float16 step compiled (fixed shape per batch, so compiled once), and the max over
        image tokens taken before the float32 cast and the scale (both monotone for scale > 0, so the max is the
        same element)."""
        assert scale > 0, scale
        hk = k8.shape[1]
        g = q.reshape(hk, (q.shape[0] // hk) * q.shape[1], q.shape[2])                    # [kv, group*Lq, D]
        s = mx.matmul(g[None], _dequant(k8, ks).swapaxes(-1, -2)).max(axis=-1)            # [n, kv, group*Lq]
        return (s.astype(mx.float32) * scale).mean(axis=(1, 2))

    def _key_screen(self, fid: int):
        """Key a kept screen (quiet-time work): its features (the current-screen read's when that frame was read, else
        encoded here on the GPU), saved to disk; its screen turn through text layers up to the last key layer;
        image-token keys at KEY_LAYERS, int8; and its score for the content-free query."""
        def g():
            yield "plan_build"
            p = self.ss.take_pending(fid) if self.ss is not None else None
            if p is None:
                return
            meta, a = p["meta"], self.a
            inp = self._inputs_like(meta["size"], a.screen_turn_prompt())
            if fid in self.read_feats:                # the GPU already encoded this frame for a read
                feats = self.read_feats.pop(fid)
                self.stats["feats_reused"] = self.stats.get("feats_reused", 0) + 1
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
            self.store.set_view(fid, self.ss.screens[fid])          # its view text, a memory slot
            for vk in [k for k in self.store.views if int(k[5:]) not in self.ss.screens]:
                self.store.drop_view(int(vk[5:]))
                self._forget(vk)
            for b in [b for b in self.blocks if b[1] not in self.ss.screens]:
                self._forget_block(b)
            mx.clear_cache()
        return g()

    def find_screens(self, text: str) -> list[int]:
        return self._run(self.find_screens_steps(text))

    def retrieval_ready(self) -> bool:
        """False while a retrieval's next piece waits for its keys to come off the disk (the scheduler runs other
        work meanwhile)."""
        f = self._pending_io
        return f is None or f.done()

    def find_screens_steps(self, text: str):
        """The kept screens to put in the context for this (partial) question: a combination of text (frame BM25)
        and Gemma's keys (calibrated, z-normalised, summed over KEY_LAYERS), better rank first; top n_screens.
        Generator (announce protocol): the question pass ("rq" pieces), then one "rs" piece per key layer and chunk."""
        if self.ss is None or self.n_screens <= 0:
            return []
        with self.store.lock:                           # the visit still on screen: the current screen supersedes it
            cur = (self.store.cur.key, self.store.cur.t0) if self.store.cur is not None else None
        fids = self.ss.ids(exclude_visit=cur)
        if not fids:
            return []
        with self.ss.lock:
            screens = [self.ss.screens[f] for f in fids]
        T = screens[0].T
        qv = yield from self._queries_steps(text, T)
        total = np.zeros(len(fids))
        # Keys are memory-mapped (nothing kept in RAM); macOS drops them under memory pressure, and reading one key
        # layer of ~95 screens back from disk took 64-186 ms in the app (2026-11-15). So the reads happen on a loader
        # thread, a chunk of KEY_CHUNK screens ahead (at most two chunks in RAM), and the GPU piece ("rs", one chunk)
        # is runnable only once its chunk is in (`retrieval_ready`): the GPU loop never waits on the disk.
        work = [(L, screens[i:i + KEY_CHUNK]) for L in KEY_LAYERS for i in range(0, len(screens), KEY_CHUNK)]

        def load(L, part):
            return np.stack([s.keys[L][0] for s in part]), np.stack([s.keys[L][1] for s in part])

        futs = [self._io.submit(load, *work[0])]
        raw = {L: [] for L in KEY_LAYERS}
        try:
            for j, (L, part) in enumerate(work):
                if j + 1 < len(work):
                    futs.append(self._io.submit(load, *work[j + 1]))
                self._pending_io = futs[j]
                yield "rs"
                self._pending_io = None
                k8, ks = futs[j].result()
                futs[j] = None
                q, scale = qv[L]
                raw[L].append(np.asarray(self._score(q, scale, mx.array(k8), mx.array(ks))))
                del k8, ks
        finally:
            self._pending_io = None
            for f in futs:
                if f is not None:
                    f.cancel()
        for L in KEY_LAYERS:
            cal = np.concatenate(raw[L]) - np.array([s.null[L] for s in screens])
            total += (cal - cal.mean()) / (cal.std() + 1e-9)
        # Adopted 2026-11-12: keys and the unit's text, each z-normalised, summed with equal weight.
        # The earlier better-rank fuse let the keys alone pick at 1 screen (text never won a disagreement).
        bt = self.ss.text_scores(text, fids)
        zt = (bt - bt.mean()) / (bt.std() + 1e-9)
        self.last_ranked = [fids[i] for i in np.argsort(-(total + zt), kind="stable")][:max(self.n_screens, 3)]
        top = self.last_ranked[:self.n_screens]
        t_of = {f: s.t for f, s in zip(fids, screens)}   # the screens held here (one may expire between pieces)
        return sorted(top, key=lambda f: t_of[f])

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

    def _branches(self) -> list:
        if self.branch_v != self.prefix_v:              # a new prefix: everything computed after the old one is invalid
            self.branches, self.branch_v = [], self.prefix_v
        return self.branches

    @staticmethod
    def _common(a, b) -> int:
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    def exact_len(self, slots) -> int:
        """How many leading slots of `slots` some exact branch holds (the scheduler's frontier)."""
        return max((self._common(b["slots"], tuple(slots)) for b in self._branches()), default=0)

    def _branch(self, slots, i: int) -> dict:
        """A branch holding the prefix and exactly slots[:i], to extend. One that holds exactly those is extended in
        place; otherwise one that holds more is copied and the copy cut back, so the other context stays ready."""
        bs, slots = self._branches(), tuple(slots)
        own = next((b for b in bs if b["slots"] == slots[:i] and not b["busy"]), None)
        if own is None:
            src = next((b for b in bs if b["slots"][:i] == slots[:i] and len(b["bounds"]) > i), None)
            if src is not None and i > 0:
                c = clone_cache(src["cache"])
                cut(self.lay, c, *src["bounds"][i])
                own = {"cache": c, "slots": slots[:i], "bounds": list(src["bounds"][:i]), "busy": False}
            else:
                assert i == 0, (i, slots, [b["slots"] for b in bs])
                own = {"cache": clone_cache(self.prefix_cache), "slots": (), "bounds": [], "busy": False}
            bs.insert(0, own)
        else:
            if len(own["bounds"]) > i:                  # a slot started and dropped: cut it off
                if i == 0:
                    own["cache"] = clone_cache(self.prefix_cache)
                else:
                    cut(self.lay, own["cache"], *own["bounds"][i])
                del own["bounds"][i:]
            self._touch(own)
        for b in [b for b in bs[MAX_BRANCHES:] if not b["busy"]]:   # least recently used first out
            bs.remove(b)
        return own

    def end_turn(self):
        """The turn closed: its exact branches and assembled context go. After an answer the prefix
        changes anyway (the exchange is appended); keeping them made idle memory step up by ~0.25 GB after the
        first question and raised the peak of reads between turns."""
        self.branches.clear()
        self.base = None
        mx.clear_cache()

    def _touch(self, b: dict):
        if b in self.branches:
            self.branches.remove(b)
        self.branches.insert(0, b)

    def upgrade(self, slots, i: int):
        def g():
            yield "check"
            b = self._branch(slots, i)
            b["busy"] = True
            try:
                c = b["cache"]
                b["bounds"].append((int(c[0].offset), window_snapshot(self.lay, c) if i > 0 else None))   # 0: the prefix
                b["cache"] = yield from self._slot_into(c, slots[i])
                b["slots"] = tuple(slots[:i + 1])
            finally:                                    # dropped mid-slot: _branch cuts the started slot off on reuse
                b["busy"] = False
        return g()

    def _assemble(self, key, compute_missing: bool, exact_all: bool = False, take: bool = False):
        """The context before the question for `key` = (prefix version, slots, frontier, screen frame), as a new
        cache. Generator: pages with no stored piece are computed in place (compute_missing) or not allowed.
        take: the answer's own assembly, the turn's last use of the branches: the branch is used itself, not copied,
        and the others are dropped first (one context in memory instead of three at the answer's peak)."""
        v, slots, frontier, fid = key
        assert v == self.prefix_v, (v, self.prefix_v)
        if frontier:
            b = next(b for b in self._branches() if b["slots"][:frontier] == tuple(slots[:frontier]))
            if take:
                self.branches.clear()
                c = b["cache"]
            else:
                self._touch(b)
                c = clone_cache(b["cache"])
            if frontier < len(b["bounds"]):             # slots after the frontier (or one started): cut them off
                cut(self.lay, c, *b["bounds"][frontier])
        else:
            c = clone_cache(self.prefix_cache)
        # Stored page pieces carry no window-layer entries, so nothing may be computed right after one (its window
        # attention would see stale entries; kv_reuse.append refuses). Every slot up to the last one that must be
        # computed is therefore computed exactly, in order; only the slots after it come from stored pieces, and the
        # current screen piece (which has window entries) refills the window at the end.
        rest = list(slots[frontier:])
        last = max((i for i, sl in enumerate(rest) if not self.has_piece(sl)), default=-1)
        if exact_all:                                   # every slot computed (for comparison)
            last = len(rest) - 1
        assert compute_missing or last < 0, rest[last]
        for slot in rest[:last + 1]:
            c = yield from self._slot_into(c, slot)
        yield "append"          # its own piece: the copy and this were 54-75 ms together, over the 50 ms limit
        return K.append(self.lay, c, [self.piece(sl) for sl in rest[last + 1:]] + [self.screens[fid]], copy=False)

    def spec(self, key, ids, have):
        """The question so far over the context `key`, leaving its last token for the answer to run. Prefilled into the
        assembled context itself; a changed question (or a prefill cut short) cuts back to the longest part still held
        (see _rewind), and only the rest is prefilled."""
        def g():
            yield "assemble"
            if self.base is None or self.base["key"] != key:
                self.base = None                        # the old context goes before the new one is made
                c = yield from self._assemble(key, compute_missing=False)
                self.base = {"key": key, "cache": c, "end": (int(c[0].offset), window_snapshot(self.lay, c)),
                             "ids": (), "cov": 0, "dirty": False, "snaps": []}
            s = self.base
            self._rewind(s, list(ids), len(ids) - 1)
            cov, target = s["cov"], len(ids) - 1
            # Stop (and snapshot) once before the question's last few tokens too: the ids end with the template's
            # closing tokens, so a snapshot only at the end lies past the words the next partial usually changes.
            mid = len(ids) - self._qtail() - SNAP_BACK
            for stop in ([mid] if cov < mid < target else []) + [target]:
                if stop <= s["cov"]:
                    continue
                s["dirty"] = True                       # until the prefill finishes (a dropped job leaves it set)
                yield from self._prefill(s["cache"], list(ids[s["cov"]:stop]), layers=4)
                c = s["cache"]
                s["snaps"] = (s["snaps"] + [(stop, (int(c[0].offset), window_snapshot(self.lay, c)))])[-SPEC_SNAPS:]
                s.update(ids=tuple(ids), cov=stop, dirty=False)
            s.update(ids=tuple(ids), cov=max(s["cov"], target))
        return g()

    def _qtail(self) -> int:
        """Tokens after the question text in question ids (the template's closing tokens), found once."""
        if getattr(self, "_qtail_n", None) is None:
            a, b = self.question_ids("hello there friend"), self.question_ids("what was that number")
            n = 0
            while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
                n += 1
            self._qtail_n = n
        return self._qtail_n

    def _rewind(self, s, ids, limit: int):
        """Cut a question prefill back to the longest prefix of `ids` (at most `limit` tokens) that a clean prefill
        holds: the context's end, or one of the last SPEC_SNAPS prefill ends (window layers can only be restored from
        a snapshot). Before this, a changed last word (or a prefill cut off at key up) used to throw the whole
        question prefill away; in 16 of 54 turns the answer then prefilled all ~15-20 question tokens itself."""
        held = s["ids"][:s["cov"]]
        n = 0
        while n < min(len(held), len(ids), limit) and held[n] == ids[n]:
            n += 1
        if not s["dirty"] and n == s["cov"]:
            return
        best_c, best = 0, s["end"]
        for c, end in s["snaps"]:
            if best_c < c <= n:
                best_c, best = c, end
        cut(self.lay, s["cache"], *best)
        s["snaps"] = [x for x in s["snaps"] if x[0] <= best_c]
        s.update(ids=tuple(held[:best_c]), cov=best_c, dirty=False)

    def answer(self, plan):
        def g():
            io = self.io
            r = io.result
            yield "assemble"
            key, words = plan["key"], list(plan["words"])
            s, self.base = self.base, None
            if s is not None and s["key"] == key:
                self.branches.clear()                   # the context is assembled: the branches aren't needed
            if s is not None and s["key"] == key:
                self._rewind(s, words, len(words) - 1)
                cache, ids = s["cache"], words[s["cov"]:]
                r["context"] = "question prefilled" if s["cov"] else "assembled before key up"
                r["question_reused"] = s["cov"]
            else:
                s = None
                cache = yield from self._assemble(key, compute_missing=True, take=True)
                ids, r["context"] = words, "assembled at release"
            del s
            r.update(context_tokens=int(cache[0].offset), question_tokens=len(words), prefilled_at_release=len(ids),
                     pages=len(plan["pages"]), exact_pages=plan["frontier"], missing=len(plan["missing"]),
                     screen_frame=plan["screen"])
            io.mark("restore_done")
            text = yield from self._speak(cache, ids, io, plan["on_first_token"])
            done = bool(text.strip()) and not io.cancel.is_set() and not io.warmup
            if done:
                self.history.append((io.question, text.strip()))
            r.update(answer=text.strip(), stop=self.a.last_stop if not io.cancel.is_set() else "cancel",
                     cancelled=io.cancel.is_set())
            io.mark("turn_complete")
            return {"exchange": done, "rebuild": done and self.history_over_cap()}
        return g()

    def _speak(self, cache, ids, io: TurnIO, on_first_token):
        """Decode, one token a piece, handing phrases and sentences to TTS as they complete (GpuWorker._answer's
        rules: an early first phrase, then whole sentences). TTS runs on its own thread: Kokoro
        synthesizes on the Neural Engine while decoding goes on, instead of decoding waiting for each chunk."""
        r = io.result
        text, buf, chunks = "", "", 0
        q: queue.SimpleQueue = queue.SimpleQueue()

        def tts_loop():
            first = True
            while True:
                item = q.get()
                if item is None:
                    return
                sentence, mid_sentence = item
                if io.cancel.is_set():
                    continue                            # drain: nothing more is spoken
                t0 = time.perf_counter()
                # The answer's first chunk starts at its first sample above -50 dB (the audible threshold): the 30 ms
                # pad kept before later chunks spaces them; before the first it only delays the answer.
                said = False
                for audio in self.tts.synth(sentence, cancel=io.cancel, trim=self.trim,
                                            tail_keep_ms=40 if mid_sentence else 120, lead_keep_ms=0 if first else 30):
                    if not said and io.on_say is not None:
                        said = True
                        try:
                            io.on_say(sentence)
                        except Exception:
                            pass                        # captions never break an answer
                    if first:
                        io.mark("tts_first_chunk")
                        r["tts_first_ms"] = {"round_trip": round((time.perf_counter() - t0) * 1000, 1),
                                             "kokoro": getattr(self.tts, "last_seconds", None)}
                        first = False
                    io.emit_audio(audio, self.tts.sample_rate)
                    if io.cancel.is_set():
                        break

        if self.tts is not None:
            io.tts_thread = threading.Thread(target=tts_loop, daemon=True, name="tts")
            io.tts_thread.start()

        def first():
            io.mark("first_token")
            on_first_token()

        def speak(sentence, mid_sentence=False):
            nonlocal chunks
            chunks += 1
            if chunks == 1:
                io.mark("first_sentence_ready")
                r["first_chunk_text"] = sentence
            if self.tts is not None:
                q.put((sentence, mid_sentence))

        try:
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
                if chunks == 0 and not done:
                    ph = split_first_phrase(buf, min_words=FIRST_MIN_WORDS)
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
        finally:
            q.put(None)                                 # the TTS thread ends after what is queued
        r["chunks_spoken"] = chunks
        return text
