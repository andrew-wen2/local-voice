"""recall_worker.py — the GPU thread for recall mode (ARCHITECTURE.md §5.9): owns MLX, the turn scheduler
(scheduler.py) and its backend (recall_backend.py). Other threads send events; this thread applies them between
pieces and runs one piece at a time, so a piece is never interrupted and events wait at most one piece.

Events (any thread): key_down, partial(text), mic_silent, speech, key_up(io), final(text), cancel_turn,
frame(fid, image), visit_ended(page key), interrupt. A turn's result arrives in io.result, and io.done is set,
when the scheduler closes the turn (answered, cancelled, or replaced by a new question).
"""
from __future__ import annotations

import queue
import threading
import time
from functools import lru_cache
from typing import Callable, Optional

import mlx.core as mx

from recall import HINT, PAGES_HEAD
from recall_backend import KEY, OPENER, SCREEN, RecallBackend, TurnIO
from scheduler import Scheduler

CAP = 4000          # memory slot cap, tokens (timeline + pages), as the M1 runs (§3.11)


class RecallWorker(threading.Thread):
    def __init__(self, adapter, tts, store, budget: int = 1120, cap: int = CAP, freshness: str = "B",
                 first_chunk: str = "phrase", trim_silence: bool = True, mlx_cache_gb: Optional[float] = 0.0,
                 history_max_tokens: int = 2000, backend_factory: Optional[Callable] = None, screen_store=None):
        super().__init__(daemon=True, name="recall-gpu")
        self.adapter, self.tts, self.store, self.budget, self.cap = adapter, tts, store, budget, cap
        self.freshness, self.first_chunk, self.trim, self.mlx_cache_gb = freshness, first_chunk, trim_silence, mlx_cache_gb
        self.history_max_tokens = history_max_tokens
        self.backend_factory = backend_factory
        self.screen_store = screen_store
        self._found: dict[str, tuple] = {}             # retrieval by text, for this turn (final == last partial)
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.ready = threading.Event()
        self.error: Optional[BaseException] = None
        self.turn_active = False
        self.io: Optional[TurnIO] = None
        self.sched: Optional[Scheduler] = None
        self.backend = None
        self.startup: dict = {}
        self._stop = False

    # ------------------------------------------------------------ events (any thread)

    def _put(self, *ev):
        self.events.put(ev)

    def key_down(self):
        self.turn_active = True
        self._put("key_down", time.perf_counter())

    def partial(self, text: str):
        self._put("partial", text)

    def mic_silent(self):
        self._put("mic_silent")

    def speech(self):
        self._put("speech")

    def key_up(self, io: TurnIO):
        self._put("key_up", time.perf_counter(), io)

    def final(self, text: str):
        self._put("final", text)

    def cancel_turn(self):
        self._put("cancel_turn")

    def frame(self, fid: int, image):
        with self.backend.frames_lock:
            self.backend.frames[fid] = image
        self._put("frame", fid)

    def visit_ended(self, key: str):
        self._put("visit_ended", key)

    def key_frame(self, fid: int):
        """A kept screen's features are pending: key it in quiet time."""
        self._put("key_frame", fid)

    def interrupt(self):
        io = self.io
        if io is not None:
            io.cancel.set()

    def run_on_worker(self, fn, timeout: float = 300.0):
        done, out = threading.Event(), {}
        self._put("call", fn, done, out)
        assert done.wait(timeout)
        if "error" in out:
            raise out["error"]
        return out.get("value")

    def stop(self):
        self._stop = True
        self._put("stop")

    def wait_ready(self, timeout: float = 300) -> bool:
        """Models loaded, the first screen read and the first prefix built."""
        if not self.ready.wait(timeout):
            return False
        if self.error:
            raise self.error
        t = time.time() + timeout
        while time.time() < t:
            if self.sched.screen >= 0 and self.sched.prefix > 0:
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------ the GPU thread

    def run(self):
        try:
            self._startup()
            self.ready.set()
            recent = []
            while not self._stop:
                try:
                    self._drain()
                    ran = self.sched.step()
                except Exception as e:                  # a bug in one turn must not stop the app
                    recent = self._failed(e, recent)
                    continue
                if not ran:
                    try:
                        ev = self.events.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    try:
                        self._apply(ev)
                    except Exception as e:
                        recent = self._failed(e, recent)
        except BaseException as e:      # surface to the app instead of dying silently
            self.error = e
            self.ready.set()
            if self.io is not None:
                self.io.result["error"] = f"worker failed: {e!r}"
                self.io.done.set()
            raise
        finally:
            if self.backend is not None and hasattr(self.backend, "close"):
                self.backend.close()

    def _failed(self, e: Exception, recent: list) -> list:
        """Log the error, end the current turn with it, carry on. Raises if errors repeat fast (a loop that can't
        recover shouldn't spin)."""
        import traceback
        tb = traceback.format_exc()
        print(f"(recall worker: {type(e).__name__}: {e}; turn ended, carrying on)", flush=True)
        now = time.time()
        recent = [t for t in recent if now - t < 10] + [now]
        io = self.io
        if io is not None:
            io.result["error"] = f"worker error: {e!r}"
            io.result["traceback"] = tb
        try:
            self.sched.jobs = [j for j in self.sched.jobs if j.gen is not None]   # drop half-made jobs
            self.sched.cancel_turn()
        except Exception:
            pass
        if io is not None and not io.done.is_set():
            self.io = None
            io.done.set()
        if len(recent) >= 5:
            raise RuntimeError("recall worker: 5 errors in 10 s") from e
        return recent

    def _drain(self):
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                return
            self._apply(ev)

    def _apply(self, ev):
        name, *args = ev
        s = self.sched
        if name == "key_down":
            self._found = {}
            s.key_down(t=args[0])
        elif name == "key_frame":
            s.add_build((KEY, args[0]))
            s.add_build((f"view:{args[0]}", 0, 0))      # its view's stored piece, once keying has registered it
        elif name == "key_up":
            self.io = self.backend.io = args[1]
            s.key_up(t=args[0])
        elif name == "visit_ended":
            key = args[0]
            with self.store.lock:
                p = self.store.m.pages.get(key)
                if p is not None:
                    s.visit_ended((key, self.store.version[key], len(p.blocks)))
        elif name == "call":
            fn, done, out = args
            try:
                out["value"] = fn()
            except BaseException as e:
                out["error"] = e
            done.set()
        elif name == "stop":
            self._stop = True
        else:                            # partial, final, mic_silent, speech, cancel_turn, frame
            getattr(s, name)(*args)

    def _turn_closed(self, turn: dict):
        io, self.io = self.io, None
        if io is None:
            return
        keep = ("release_s", "waited_for_read", "need_frame", "screen_at_up", "cancelled", "retrievals",
                "page_changes", "pages_cut", "wasted_s", "gpu_s", "prefix_at_down", "error")
        io.result.update({k: turn.get(k) for k in keep if k in turn})
        if turn.get("error") and self.sched.errors:
            io.result["traceback"] = self.sched.errors[-1][2]
        p = turn.get("plan") or {}
        io.result.update(plan_frontier=p.get("frontier"), plan_question_reused=p.get("question_reused"),
                         plan_prefill=p.get("prefill"), plan_base_reused=p.get("base_reused"),
                         plan_missing=len(p.get("missing") or ()))
        io.done.set()

    def _startup(self):
        if self.mlx_cache_gb is not None:
            mx.set_cache_limit(int(self.mlx_cache_gb * 1e9))
        t0 = time.perf_counter()
        a = self.adapter
        if self.backend_factory is None:
            a.load()
            a.set_budget(self.budget)
        self.startup["gemma_load_s"] = round(time.perf_counter() - t0, 2)
        if self.tts is not None:
            t0 = time.perf_counter()
            self.tts.load()
            for _ in self.tts.synth("Ready."):
                pass
            self.startup["kokoro_load_warm_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        self.backend = (self.backend_factory() if self.backend_factory is not None else
                        RecallBackend(a, self.store, self.tts, first_chunk=self.first_chunk,
                                      trim_silence=self.trim, history_max_tokens=self.history_max_tokens,
                                      screen_store=self.screen_store))
        self.startup["anchors_s"] = round(time.perf_counter() - t0, 2)
        self.sched = Scheduler(self.backend, self._retrieve, freshness=self.freshness)
        self.sched.on_turn = self._turn_closed
        tok = self.backend.a.tok if hasattr(self.backend, "a") else None
        self._count = lru_cache(maxsize=4096)(
            (lambda s: len(tok.encode(s, add_special_tokens=False))) if tok is not None else (lambda s: len(s.split())))

    def _retrieve(self, text: str) -> tuple:
        """The memory slots for this (partial) question, in context order: kept screens (found by text and Gemma's
        keys, oldest first), the pages turn's opener, then pages [(page key, version, lines shown)], oldest first.
        The page cap counts the timeline and the pages header, as eval/recall/rag_arm.py does; screens come on top
        of it, as in the reading test (§12.16). Cached by text for the turn: the final transcript is usually the
        last partial."""
        if text in self._found:
            return self._found[text]
        used = self._count(self.store.timeline() + "\n\n") + self._count(PAGES_HEAD + HINT)
        find = getattr(self.backend, "find_screens", None)
        if find is None:                              # backends without kept screens (tests)
            out = tuple(self.store.pick(text, self.cap, self._count, used=used))
        else:
            # §5.25 (Andrew, 2026-11-12): the memory text is the top kept screen's own view text, up to the same
            # budget; whole pages only while no screen has been kept yet.
            fids = find(text)
            ss = getattr(self.backend, "ss", None)
            if fids and ss is not None and ss.ready(fids[0]):
                vk = self.store.set_view(fids[0], ss.screens[fids[0]])
                mem = (self.store.view_slot(vk, self.cap - used, self._count),)
            else:
                mem = tuple(self.store.pick(text, self.cap, self._count, used=used))
            out = tuple((SCREEN, f) for f in fids) + ((OPENER,),) + mem
        self._found[text] = out
        return out


class ScreenEncoder(threading.Thread):
    """Encodes kept screens on the Neural Engine in quiet time (§5.12: background, low power) and hands them to the GPU
    thread for keying. A frame's image is held only until it is encoded. Without a converted Neural Engine encoder for
    the frame's size, the image itself is handed over and the GPU encodes it while keying."""

    def __init__(self, adapter, screens, quiet: Callable[[], bool], on_ready: Callable[[int], None], ane=None):
        super().__init__(daemon=True, name="screen-encoder")
        self.a, self.screens, self.quiet, self.on_ready, self.ane = adapter, screens, quiet, on_ready, ane
        self.cv = threading.Condition()
        self.q: list[tuple] = []
        self._stop = False
        self.stats = {"encoded_ane": 0, "handed_to_gpu": 0, "errors": 0}

    def submit(self, fid: int, image, meta: dict):
        with self.cv:
            self.q.append((fid, image, {**meta, "size": tuple(image.size)}))
            self.cv.notify_all()

    def poke(self):
        with self.cv:
            self.cv.notify_all()

    def stop(self):
        with self.cv:
            self._stop = True
            self.cv.notify_all()

    def run(self):
        import numpy as np
        while True:
            with self.cv:
                while not self._stop and not (self.q and self.quiet()):
                    self.cv.wait(0.1)
                if self._stop:
                    return
                fid, image, meta = self.q.pop(0)
            tokens = None
            if self.ane is not None:
                try:
                    inp = self.a._inputs(image, self.a.screen_turn_prompt())
                    pv = np.asarray(inp["pixel_values"], dtype=np.float32)
                    if self.ane.has(pv.shape[2], pv.shape[3]):
                        tokens = self.ane.encode(pv)
                except Exception:                      # fall back to the GPU for this frame
                    self.stats["errors"] += 1
            if tokens is not None:
                self.screens.put_pending(fid, meta, tokens=tokens)
                self.stats["encoded_ane"] += 1
            else:
                self.screens.put_pending(fid, meta, image=image)
                self.stats["handed_to_gpu"] += 1
            self.on_ready(fid)
