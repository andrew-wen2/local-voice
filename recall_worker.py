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
from recall_backend import BLOCK, KEY, OPENER, SCREEN, RecallBackend, TurnIO
from scheduler import Scheduler

CAP = 4000          # memory slot cap, tokens (timeline + pages), as the M1 runs (§3.11)


class RecallWorker(threading.Thread):
    def __init__(self, adapter, tts, store, budget: int = 1120, cap: int = CAP, freshness: str = "B",
                 first_chunk: str = "phrase", trim_silence: bool = True, mlx_cache_gb: Optional[float] = 0.0,
                 history_max_tokens: int = 2000, backend_factory: Optional[Callable] = None, screen_store=None):
        super().__init__(daemon=True, name="recall-gpu")
        self.adapter, self.tts, self.store, self.budget, self.cap = adapter, tts, store, budget, cap
        self._slots: dict = {}                          # (view, version, room) -> a kept screen's block slot
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
            s.add_build((KEY, args[0]))     # its block is queued once keyed (_queue_blocks); no separate view piece
        elif name == "key_up":
            self.io = self.backend.io = args[1]
            s.key_up(t=args[0])
        elif name == "visit_ended":
            key = args[0]
            with self.store.lock:
                p = self.store.m.pages.get(key)
                if p is not None:
                    s.visit_ended((key, self.store.version[key], len(p.blocks)))
            self._queue_blocks()
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
        end = getattr(self.backend, "end_turn", None)
        if end is not None:
            end()
        self._queue_blocks()
        io, self.io = self.io, None
        if io is None:
            return
        keep = ("release_s", "waited_for_read", "need_frame", "screen_at_up", "cancelled", "retrievals",
                "page_changes", "pages_cut", "wasted_s", "gpu_s", "prefix_at_down", "error", "states", "pieces", "blocked",
                "frontier_found", "page_log", "frontier_log", "frontier_at_up", "pages_at_up", "alt_ready_at_up", "job_errors", "decode_log")
        io.result.update({k: turn.get(k) for k in keep if k in turn})
        if turn.get("error") and self.sched.errors:
            io.result["traceback"] = self.sched.errors[-1][2]
        p = turn.get("plan") or {}
        io.result.update(plan_frontier=p.get("frontier"), plan_question_reused=p.get("question_reused"),
                         plan_prefill=p.get("prefill"), plan_base_reused=p.get("base_reused"),
                         plan_missing=len(p.get("missing") or ()))
        th = io.tts_thread
        if th is not None and th.is_alive():            # the answer is decoded; done once its audio is all out
            threading.Thread(target=lambda: (th.join(120), io.done.set()), daemon=True, name="tts-wait").start()
        else:
            io.done.set()

    def _startup(self):
        if self.mlx_cache_gb is not None:
            mx.set_cache_limit(int(self.mlx_cache_gb * 1e9))
        t0 = time.perf_counter()
        a = self.adapter
        if self.backend_factory is None:
            a.load()
            a.set_budget(self.budget)
            self.startup["wired_gb"] = wire_weights()
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
        self.sched = Scheduler(self.backend, self._retrieve, freshness=self.freshness, retrieve_alt=self._retrieve_alt,
                               retrieve_steps=self._retrieve_steps)
        self.sched.on_turn = self._turn_closed
        tok = self.backend.a.tok if hasattr(self.backend, "a") else None
        self._count = lru_cache(maxsize=512)(
            (lambda s: len(tok.encode(s, add_special_tokens=False))) if tok is not None else (lambda s: len(s.split())))

    def _retrieve(self, text: str) -> tuple:
        """The memory slots for this (partial) question, in context order: kept screens (found by text and Gemma's
        keys, oldest first), the pages turn's opener, then pages [(page key, version, lines shown)], oldest first.
        The page cap counts the timeline and the pages header, as eval/recall/rag_arm.py does; screens come on top
        of it, as in the reading test (§12.16). Cached by text for the turn: the final transcript is usually the
        last partial."""
        if text not in self._found:
            self._found[text] = self._find(text)
        return self._found[text][0]

    def _retrieve_alt(self, text: str):
        """The runner-up context for this (partial) question: the next-best kept screen, with its own view as the
        memory, laid out as _retrieve lays out the best one; None if there is none. The scheduler computes it while
        the GPU would otherwise idle (M3, 2026-11-13)."""
        if text not in self._found:
            self._found[text] = self._find(text)
        return self._found[text][1]

    def _used(self) -> int:
        return self._count(self.store.timeline() + "\n\n") + self._count(PAGES_HEAD + HINT)

    def _block_slot(self, fid: int, used: int) -> tuple:
        """Cached by (view, its version, room for it): counting a view's lines is quadratic in them, and every kept
        screen's slot is looked up after each visit and turn."""
        vk = self.store.set_view(fid, self.backend.ss.screens[fid])
        with self.store.lock:
            k = (vk, self.store.version[vk], self.cap - used)
        out = self._slots.get(k)
        if out is None:
            if len(self._slots) > 4096:
                self._slots.clear()                     # bounded: memory stays flat
            out = self._slots[k] = (BLOCK, fid, self.store.view_slot(vk, self.cap - used, self._count))
        return out

    def _queue_blocks(self):
        """Every kept screen's block as the context would use it now (its view text grows as frames arrive, and the
        room for it shrinks as the timeline grows): built in quiet time when missing or out of date."""
        be, ss = self.backend, getattr(self.backend, "ss", None)
        if ss is None or getattr(be, "blocks", None) is None:
            return
        used = self._used()
        for fid in ss.ids():
            if ss.ready(fid):
                slot = self._block_slot(fid, used)
                if not be.has_piece(slot):
                    self.sched.add_build(slot)

    def _retrieve_steps(self, text: str):
        """(best, runner-up) for a partial transcript, as GPU pieces the scheduler interleaves with page work (its
        Retrieve job, M3 2026-11-15). Same result as _retrieve/_retrieve_alt, and fills the same cache."""
        yield "rprep"
        if text not in self._found:
            self._found[text] = yield from self._find_steps(text)
        return self._found[text]

    def _find(self, text: str) -> tuple:
        gen = self._find_steps(text)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    def _find_steps(self, text: str):
        used = self._count(self.store.timeline() + "\n\n") + self._count(PAGES_HEAD + HINT)
        steps = getattr(self.backend, "find_screens_steps", None)
        find = getattr(self.backend, "find_screens", None)
        if steps is None and find is None:            # backends without kept screens (tests)
            return tuple(self.store.pick(text, self.cap, self._count, used=used)), None
        # §5.25 (Andrew, 2026-11-12): the memory text is the top kept screen's own view text, up to the same
        # budget; whole pages only while no screen has been kept yet.
        fids = (yield from steps(text)) if steps is not None else find(text)
        yield "rlay"
        ss = getattr(self.backend, "ss", None)

        def layout(fs):
            if fs and ss is not None and ss.ready(fs[0]):
                if len(fs) == 1 and getattr(self.backend, "blocks", None) is not None:
                    return (self._block_slot(fs[0], used),)     # one slot: screen turn, opener, view (§5.29)
                vk = self.store.set_view(fs[0], ss.screens[fs[0]])
                mem = (self.store.view_slot(vk, self.cap - used, self._count),)
            else:
                mem = tuple(self.store.pick(text, self.cap, self._count, used=used))
            return tuple((SCREEN, f) for f in fs) + ((OPENER,),) + mem

        out, alt = layout(fids), None
        ranked = list(getattr(self.backend, "last_ranked", None) or [])
        if len(fids) == 1 and ss is not None:            # one screen in the context (today's setting)
            nxt = next((f for f in ranked if f not in fids and ss.ready(f)), None)
            if nxt is not None:
                alt = layout([nxt])
        return out, alt


WIRED_HEADROOM_GB = 0.25          # on top of the weights (the prefix); wired memory can't be reclaimed for other apps


def wire_weights() -> float:
    """Keep MLX's memory resident (M3, 2026-11-13). Background work in quiet time runs text layers 0-23 only, so for an
    hour the KV-shared layers (24-41) are untouched and macOS paged them out: the first answer after an hour took
    3-3.8 s to its first token instead of ~0.15 s. Wired up to the weights plus headroom (buffers allocated first, the
    weights, are wired first), below the system's recommended working set. "No swapping of the assistant" (M3 bar).
    Returns the limit in GB."""
    info = mx.device_info()
    want = mx.get_active_memory() + int(WIRED_HEADROOM_GB * 1e9)
    limit = int(min(want, 0.9 * info.get("max_recommended_working_set_size", want)))
    mx.set_wired_limit(limit)
    return round(limit / 1e9, 2)


class ScreenEncoder(threading.Thread):
    """Hands kept screens to the GPU thread for keying, in quiet time. Since 2026-11-13 (Andrew: memory under 4 GB) the
    image itself is handed over and the GPU encodes it while keying, or reuses the current-screen read's features when
    that frame was read: one image encoder instead of two (the Neural Engine copy's process held ~0.74 GB, mostly its 16
    chunks' input/output buffers). A frame's image is held only until it is encoded."""

    def __init__(self, screens, quiet: Callable[[], bool], on_ready: Callable[[int], None]):
        super().__init__(daemon=True, name="screen-encoder")
        self.screens, self.quiet, self.on_ready = screens, quiet, on_ready
        self.cv = threading.Condition()
        self.q: list[tuple] = []
        self._stop = False
        self.stats = {"handed_to_gpu": 0}

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
        while True:
            with self.cv:
                while not self._stop and not (self.q and self.quiet()):
                    self.cv.wait(0.1)
                if self._stop:
                    return
                fid, image, meta = self.q.pop(0)
            self.screens.put_pending(fid, meta, image)
            self.stats["handed_to_gpu"] += 1
            self.on_ready(fid)
