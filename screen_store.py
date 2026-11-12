"""screen_store.py — kept screens as memory (ARCHITECTURE.md §5.12, §12.15–12.19).

A kept screen is its vision features (the encoder's output, ~1,066 image tokens; written to disk, owner-only, int8 with a
scale per token: Andrew's default, 2026-11-11, ~2.7 MB instead of ~5.5 MB a screen) and its
retrieval keys (image-token keys of full-attention layers 11 and 17, int8 with a scale per key; kept in memory),
plus what the text index needs: time, page key, title, page-area OCR lines. No screenshot is stored: a frame's image
is held only until it has been encoded.

Lifecycle: the OCR thread decides a frame is kept (recall.FrameKeeper) -> the Neural Engine thread encodes it and
puts the result here as pending -> the GPU thread keys it (recall_backend) and adds it -> it can be found and read.
Everything older than the horizon (an hour) is dropped, files included; the folder is deleted on close (Andrew's
privacy decision, 2026-11-11: plain on disk, owner-only, gone on quit and after an hour).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from memory import BM25, terms

KEY_LAYERS = (11, 17)             # §12.19's decision: layers 11 + 17, every image token, int8
HORIZON_S = 3600.0


def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-vector symmetric int8 over the last axis: (int8 values, float16 scales with a trailing 1)."""
    s = np.abs(x).max(axis=-1, keepdims=True).astype(np.float32) / 127 + 1e-12
    return np.clip(np.round(x / s), -127, 127).astype(np.int8), s.astype(np.float16)


@dataclass
class Screen:
    fid: int
    t: float                      # seconds since the store's start (the page store's clock)
    key: str                      # page key (§12.9)
    title: str
    lines: list
    keys: dict = field(default_factory=dict)      # layer -> (int8 [kv heads, n_img, D], float16 scales [.., 1])
    null: dict = field(default_factory=dict)      # layer -> this frame's score for the content-free query
    path: Optional[Path] = None   # vision features on disk
    size: tuple = (0, 0)          # the frame's size in pixels (the processor's resize, so the token count, depends on it)
    T: int = 0                    # tokens of its screen turn with <bos> (where the queries are re-rotated to)


class ScreenStore:
    def __init__(self, directory: Optional[Path] = None, horizon_s: float = HORIZON_S):
        self.dir = Path(directory) if directory else Path(tempfile.mkdtemp(prefix="lv-screens-"))
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        self.horizon_s = horizon_s
        self.lock = threading.RLock()
        self.pending: dict[int, dict] = {}        # fid -> {"meta": {...}, "tokens": ndarray | None, "image": PIL | None}
        self.screens: dict[int, Screen] = {}
        self.version = 0                          # bumps when the set of ready screens changes
        self._bm25 = None
        self._more: dict[int, list] = {}          # fid -> text of dropped frames waiting for that screen to be ready

    # -- the pipeline
    def put_pending(self, fid: int, meta: dict, tokens=None, image=None):
        with self.lock:
            self.pending[fid] = {"meta": meta, "tokens": tokens, "image": image}

    def take_pending(self, fid: int) -> Optional[dict]:
        with self.lock:
            return self.pending.pop(fid, None)

    def save_features(self, fid: int, feats: np.ndarray) -> Path:
        """int8 per token (the last axis) with a float16 scale each."""
        path = self.dir / f"{fid}.npz"
        q, sc = quantize_int8(np.asarray(feats, dtype=np.float32))
        with open(path, "wb") as f:
            np.savez(f, q=q, s=sc)
        os.chmod(path, 0o600)
        return path

    def load_features(self, fid: int) -> np.ndarray:
        with self.lock:
            path = self.screens[fid].path
        z = np.load(path)
        return (z["q"].astype(np.float32) * z["s"].astype(np.float32)).astype(np.float16)

    def add(self, s: Screen):
        with self.lock:
            extra = self._more.pop(s.fid, [])
            if extra:
                s.lines = _union(s.lines, extra)
            self.screens[s.fid] = s
            self.version += 1
            self._bm25 = None

    def add_lines(self, fid: int, lines: list):
        """Text of a frame that wasn't kept, given to the kept screen that stands in for it (its unit, §5.21 X1). The
        screen may not be ready yet (still being encoded): the lines wait for it."""
        with self.lock:
            if fid in self.screens:
                self.screens[fid].lines = _union(self.screens[fid].lines, lines)
                self._bm25 = None
            else:
                self._more[fid] = _union(self._more.get(fid, []), lines)

    def ready(self, fid: int) -> bool:
        with self.lock:
            return fid in self.screens

    def expire(self, now_t: float) -> int:
        """Drop screens older than the horizon, and their files. Returns how many were dropped."""
        with self.lock:
            old = [f for f, s in self.screens.items() if now_t - s.t > self.horizon_s]
            for f in old:
                s = self.screens.pop(f)
                if s.path is not None:
                    s.path.unlink(missing_ok=True)
            if old:
                self.version += 1
                self._bm25 = None
            return len(old)

    def close(self):
        with self.lock:
            self.screens.clear()
            self.pending.clear()
            self._more.clear()
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- finding
    def ids(self, exclude_visit: Optional[tuple] = None) -> list[int]:
        """Ready screens, oldest first. exclude_visit=(page key, visit start): leave out the frames of the visit still
        on screen, which the current screen supersedes (§5.12, after the first live test)."""
        with self.lock:
            out = sorted(self.screens, key=lambda f: self.screens[f].t)
            if exclude_visit is not None:
                key, t0 = exclude_visit
                out = [f for f in out if not (self.screens[f].key == key and self.screens[f].t >= t0)]
            return out

    def text_scores(self, question: str, fids: list[int]) -> np.ndarray:
        """BM25 of each of fids (title twice + its unit's page-area text) for the question."""
        with self.lock:
            allf = self.ids()
            if self._bm25 is None or self._bm25[0] != allf:
                docs = [terms(self.screens[f].title) * 2 + terms(" ".join(self.screens[f].lines)) for f in allf]
                self._bm25 = (allf, BM25(docs) if docs else None)
            if not allf:
                return np.zeros(0)
            sc = self._bm25[1].scores(terms(question))
            at = {f: i for i, f in enumerate(allf)}
            return np.array([sc[at[f]] for f in fids], dtype=np.float64)

    def text_ranking(self, question: str) -> list[int]:
        """F1 (§12.15): BM25 over each screen's title (twice) and page-area OCR lines, ties to the latest."""
        with self.lock:
            fids = self.ids()
            if self._bm25 is None or self._bm25[0] != fids:
                docs = [terms(self.screens[f].title) * 2 + terms(" ".join(self.screens[f].lines)) for f in fids]
                self._bm25 = (fids, BM25(docs) if docs else None)
            if not fids:
                return []
            sc = self._bm25[1].scores(terms(question))
            return [fids[i] for i in sorted(range(len(fids)), key=lambda i: (-sc[i], -self.screens[fids[i]].t))]


def _union(a: list, b: list) -> list:
    seen = set(a)
    return list(a) + [x for x in b if not (x in seen or seen.add(x))]


def fuse(text_rank: list[int], key_rank: list[int], k: int) -> list[int]:
    """§12.17's combination: each screen's better rank of the two routes; ties go to the keys' order (they found
    what text couldn't). Returns the top k screens."""
    pos_t = {f: i for i, f in enumerate(text_rank)}
    pos_k = {f: i for i, f in enumerate(key_rank)}
    allf = set(pos_t) | set(pos_k)
    big = len(allf) + 1
    return sorted(allf, key=lambda f: (min(pos_t.get(f, big), pos_k.get(f, big)), pos_k.get(f, big)))[:k]
