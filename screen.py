"""screen.py — capture the main display and decide when the screen has really changed.

Capture (review D13): `screencapture -x -m -t jpg` into a private 0700 temp folder; the file is
loaded into memory and deleted immediately (D4). Images are never saved or logged.
Change filter (D20): any pixel change outside a masked menu-bar status area (clock, battery,
Wi-Fi) counts; once a change is seen, wait for the screen to settle (no change for 0.3 s),
but hand the frame over at most 3 s after the first change even if it never settles.
Permission (D5): `has_screen_permission()` asks CoreGraphics without triggering a prompt.
"""

from __future__ import annotations

import ctypes
import os
import resource
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

IDLE_POLL_S = 1.0
ACTIVE_POLL_S = 0.3
SETTLE_S = 0.3
MAX_DELAY_S = 3.0
# "cells" rule (ARCHITECTURE.md §8.4; real-use log 2026-11-06: 13.4 -> 4.5 reads/min vs any-pixel)
CELL_SCALE = 4              # compare a 4x-shrunk copy...
CELL_PX = 16                # ...in 16 px cells = 64 screen px, about one image token at 1120
CELL_MIN = 40               # changed cells needed (or 5% of the screen, whichever is smaller)
CELL_FRAC = 0.05
CELL_SETTLE_S = 1.5         # quiet this long before reading
STRUCTURAL_FRAC = 0.30      # this much changing at once = window switch / scroll / page load
STRUCTURAL_SETTLE_S = 0.5
STALE_CAP_S = 25.0          # an unread change is read after this long regardless
ANIMATED_OF = (5, 10)       # a cell that changed in 5 of the last 10 polls is "animated" and ignored
MENU_BAR_PX = 80            # Retina menu-bar height incl. margin (device pixels)
STATUS_AREA_FRAC = 0.30     # right part of the menu bar: clock, battery, Wi-Fi, etc.


WINLIST = Path(__file__).resolve().parent / "tools/winlist"


def front_apps(limit: int = 4) -> list[str]:
    """On-screen app windows, front to back: app names only, no titles (tools/winlist)."""
    try:
        import json
        out = subprocess.run([str(WINLIST)], capture_output=True, text=True, timeout=2).stdout
        return [w["app"] for w in json.loads(out)][:limit]
    except Exception:
        return []


def _windows() -> list[dict]:
    try:
        import json
        return json.loads(subprocess.run([str(WINLIST)], capture_output=True, text=True, timeout=2).stdout)
    except Exception:
        return []


def window_summary(wins: Optional[list] = None, screen_pt=(1512, 982), grid_pt: int = 8) -> dict:
    """Open windows front to back with how much of each is covered, for the model and the log.
    App names only (no titles). Windows under 100x100 pt (tab strips, palettes) are skipped.
    Returns {"apps": [...], "line": "Open windows, front to back: A; B (mostly covered); ..."}."""
    wins = _windows() if wins is None else wins
    wins = [w for w in wins if w["w"] >= 100 and w["h"] >= 100]
    gw, gh = screen_pt[0] // grid_pt, screen_pt[1] // grid_pt
    taken = np.zeros((gh, gw), bool)
    parts = []
    for w in wins:                                          # front to back
        x0, y0 = max(0, int(w["x"] // grid_pt)), max(0, int(w["y"] // grid_pt))
        x1, y1 = min(gw, int((w["x"] + w["w"]) // grid_pt)), min(gh, int((w["y"] + w["h"]) // grid_pt))
        area = max(1, (x1 - x0) * (y1 - y0))
        region = taken[y0:y1, x0:x1]
        visible = 1 - region.sum() / area if area else 0
        region[:] = True
        label = ("" if visible >= 0.9 else " (partly covered)" if visible >= 0.5
                 else " (mostly covered)" if visible >= 0.05 else " (hidden behind others)")
        parts.append(w["app"] + label)
    line = ("Open windows, front to back: " + "; ".join(parts) + ".") if parts else ""
    return {"apps": [w["app"] for w in wins], "line": line}


def active_displays() -> int:
    """Number of active displays (the capture takes only the main one)."""
    try:
        cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        n = ctypes.c_uint32(0)
        cg.CGGetActiveDisplayList(0, None, ctypes.byref(n))
        return int(n.value)
    except Exception:
        return -1


def has_screen_permission() -> bool:
    """CoreGraphics CGPreflightScreenCaptureAccess(): True if Screen Recording is granted. No prompt."""
    cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
    return bool(cg.CGPreflightScreenCaptureAccess())


def private_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="localvoice-cap-"))
    os.chmod(d, 0o700)
    return d


class Capturer:
    """One screenshot of the main display -> RGB PIL image in memory; the file is deleted at once."""

    def __init__(self, fmt: str = "jpg", directory: Optional[Path] = None):
        self.fmt = fmt
        self.dir = directory or private_dir()
        self.stats = {"n": 0, "total_s": 0.0, "max_s": 0.0, "child_cpu_s": 0.0}

    def grab(self) -> Image.Image:
        path = self.dir / f"frame.{self.fmt}"
        cpu0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.perf_counter()
        try:
            subprocess.run(["screencapture", "-x", "-m", f"-t{self.fmt}", str(path)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with Image.open(path) as im:
                img = im.convert("RGB")
                img.load()
        finally:
            path.unlink(missing_ok=True)
        dt = time.perf_counter() - t0
        cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        s = self.stats
        s["n"] += 1
        s["total_s"] += dt
        s["max_s"] = max(s["max_s"], dt)
        s["child_cpu_s"] += (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)
        return img

    def cleanup(self):
        for p in self.dir.glob("*"):
            p.unlink(missing_ok=True)
        self.dir.rmdir()


_OUTPUT_CLASS = None


def _stream_output_class():
    """The SCStreamOutput receiver, defined once per process (Objective-C classes can't be redefined)."""
    global _OUTPUT_CLASS
    if _OUTPUT_CLASS is not None:
        return _OUTPUT_CLASS
    import CoreMedia
    import objc
    import Quartz
    import ScreenCaptureKit as SCK
    from Foundation import NSObject

    class LocalVoiceStreamOutput(NSObject, protocols=[objc.protocolNamed("SCStreamOutput")]):
        owner = None                                  # the StreamCapturer (a Python attribute)

        def stream_didOutputSampleBuffer_ofType_(self, stream, sb, kind):
            owner = self.owner
            if owner is None:
                return
            atts = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(sb, False)
            if not atts or atts[0].get(SCK.SCStreamFrameInfoStatus) != SCK.SCFrameStatusComplete:
                return                                # idle / blank frames carry no new pixels
            pb = CoreMedia.CMSampleBufferGetImageBuffer(sb)
            Quartz.CVPixelBufferLockBaseAddress(pb, 1)
            try:
                h, w = Quartz.CVPixelBufferGetHeight(pb), Quartz.CVPixelBufferGetWidth(pb)
                bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
                raw = np.frombuffer(Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(bpr * h), np.uint8)
                rgb = raw.reshape(h, bpr)[:, :w * 4].reshape(h, w, 4)[:, :, 2::-1].copy()
            finally:
                Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
            with owner._lock:
                owner._latest = rgb
                owner.stats["stream_frames"] += 1

    _OUTPUT_CLASS = LocalVoiceStreamOutput
    return _OUTPUT_CLASS


class StreamCapturer:
    """ScreenCaptureKit, in process (ARCHITECTURE.md §8.6): one stream of the main display, scaled by macOS
    to the change gate's size (1/CELL_SCALE), delivered only when something on screen changed, at most every
    `interval_s`; and a full-resolution screenshot (SCScreenshotManager) only when a frame is handed over.
    No files, no process per poll. Frames stay in memory (D4).

      cap = StreamCapturer().start()
      ScreenWatcher(on_frame, grab=cap.grab, grab_small=cap.grab_small)
    """

    def __init__(self, interval_s: float = ACTIVE_POLL_S, timeout_s: float = 5.0):
        self.interval_s, self.timeout_s = interval_s, timeout_s
        self.stats = {"n": 0, "total_s": 0.0, "max_s": 0.0, "cpu_s": 0.0, "stream_frames": 0, "small_reads": 0}
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stream = self._filter = self._full_cfg = None

    @staticmethod
    def _wait(start, timeout):
        ev, box = threading.Event(), {}

        def done(*a):
            box["a"] = a
            ev.set()
        start(done)
        if not ev.wait(timeout):
            raise TimeoutError("ScreenCaptureKit did not answer")
        return box["a"]

    def start(self) -> "StreamCapturer":
        import CoreMedia
        import libdispatch
        import Quartz
        import ScreenCaptureKit as SCK

        content, err = self._wait(lambda h: SCK.SCShareableContent.
                                  getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                                      False, True, h), self.timeout_s)
        if content is None:
            raise RuntimeError(f"no shareable content: {err}")
        main = Quartz.CGMainDisplayID()
        display = next(d for d in content.displays() if d.displayID() == main)
        self._filter = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])
        scale = 2                                     # device px per point (Retina); full = points * scale
        full_w, full_h = display.width() * scale, display.height() * scale
        cfg = SCK.SCStreamConfiguration.alloc().init()
        cfg.setWidth_(full_w // CELL_SCALE)
        cfg.setHeight_(full_h // CELL_SCALE)
        cfg.setMinimumFrameInterval_(CoreMedia.CMTimeMake(int(round(self.interval_s * 1000)), 1000))
        cfg.setPixelFormat_(1111970369)               # kCVPixelFormatType_32BGRA
        cfg.setShowsCursor_(False)
        cfg.setQueueDepth_(3)
        self._full_cfg = SCK.SCStreamConfiguration.alloc().init()
        self._full_cfg.setWidth_(full_w)
        self._full_cfg.setHeight_(full_h)
        self._full_cfg.setShowsCursor_(False)
        self._out = _stream_output_class().alloc().init()
        self._out.owner = self
        self._queue = libdispatch.dispatch_queue_create(b"localvoice.sck", None)
        self._stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(self._filter, cfg, None)
        ok, err = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._out, SCK.SCStreamOutputTypeScreen, self._queue, None)
        if not ok:
            raise RuntimeError(f"addStreamOutput failed: {err}")
        (err,) = self._wait(lambda h: self._stream.startCaptureWithCompletionHandler_(h), self.timeout_s)
        if err is not None:
            raise RuntimeError(f"stream did not start: {err}")
        t0 = time.time()
        while self._latest is None and time.time() - t0 < self.timeout_s:
            time.sleep(0.02)                          # the first frame always arrives
        return self

    def grab_small(self) -> np.ndarray:
        """The latest stream frame (already shrunk). Unchanged screen -> the same array as last time."""
        with self._lock:
            self.stats["small_reads"] += 1
            return self._latest

    def grab(self) -> Image.Image:
        """Full-resolution screenshot of the main display, in process."""
        import Quartz
        import ScreenCaptureKit as SCK
        t0, c0 = time.perf_counter(), time.process_time()
        img, err = self._wait(lambda h: SCK.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
            self._filter, self._full_cfg, h), self.timeout_s)
        if img is None:
            raise RuntimeError(f"screenshot failed: {err}")
        w, h, bpr = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img), Quartz.CGImageGetBytesPerRow(img)
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
        bgra = np.frombuffer(data, np.uint8)[:h * bpr].reshape(h, bpr)[:, :w * 4].reshape(h, w, 4)
        out = Image.fromarray(bgra[:, :, 2::-1].copy())     # 32-bit little-endian, alpha first = BGRA
        dt = time.perf_counter() - t0
        s = self.stats
        s["n"] += 1
        s["total_s"] += dt
        s["max_s"] = max(s["max_s"], dt)
        s["cpu_s"] += time.process_time() - c0          # process-wide while the call ran (an upper bound)
        return out

    def cleanup(self):
        if self._stream is not None:
            try:
                self._wait(lambda h: self._stream.stopCaptureWithCompletionHandler_(h), self.timeout_s)
            except TimeoutError:
                pass
            self._stream = None


def open_capture():
    """The app's screen source: (capturer, grab_small or None). A ScreenCaptureKit stream (§8.6, ~5x less CPU
    than a `screencapture` process per poll); falls back to `screencapture` only if the stream can't start."""
    try:
        cap = StreamCapturer().start()
        return cap, cap.grab_small
    except Exception as e:                            # older macOS, missing bindings, permission quirks
        print(f"ScreenCaptureKit unavailable ({e}); using screencapture", flush=True)
    return Capturer(fmt="jpg"), None


def change_mask(h: int, w: int) -> np.ndarray:
    """True where changes count. Masks the right part of the menu bar (status items, clock)."""
    m = np.ones((h, w), bool)
    m[:MENU_BAR_PX, int(w * (1 - STATUS_AREA_FRAC)):] = False
    return m


def shrink(img: Image.Image) -> np.ndarray:
    return np.asarray(img.reduce(CELL_SCALE))


def cell_changes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean [rows, cols] grid: any pixel differs inside the cell (inputs are shrunk frames)."""
    h, w = (a.shape[0] // CELL_PX) * CELL_PX, (a.shape[1] // CELL_PX) * CELL_PX
    d = np.any(a[:h, :w] != b[:h, :w], axis=-1)
    return d.reshape(h // CELL_PX, CELL_PX, w // CELL_PX, CELL_PX).any(axis=(1, 3))


def cell_mask(rows: int, cols: int) -> np.ndarray:
    """Cells that count: the menu-bar status area (clock, battery) is ignored."""
    m = np.ones((rows, cols), bool)
    top = -(-MENU_BAR_PX // (CELL_PX * CELL_SCALE))
    m[:top, int(cols * (1 - STATUS_AREA_FRAC)):] = False
    return m


def changed(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
    """Any pixel difference where the mask is True (a single character counts)."""
    if a.shape != b.shape:
        return True
    diff = np.any(a != b, axis=-1) if a.ndim == 3 else (a != b)
    if mask is not None:
        diff &= mask
    return bool(diff.any())


class ScreenWatcher(threading.Thread):
    """Polls the screen and hands settled, changed frames to `on_frame(frame_id, captured_at, image)`.

    `grab` and `clock`/`sleep` are injectable so the policy can be tested without a display.
    `grab_small` (rule "cells" only): a source of already-shrunk frames, e.g. StreamCapturer's stream; the
    change gate runs on those, and `grab` (full resolution) is called only for a frame being handed over.
    """

    def __init__(self, on_frame: Callable, grab: Callable[[], Image.Image],
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 describe: Optional[Callable[[], list]] = None, rule: str = "cells",
                 grab_small: Optional[Callable[[], np.ndarray]] = None):
        super().__init__(daemon=True, name="screen-watcher")
        self.on_frame, self.grab, self.clock, self.sleep = on_frame, grab, clock, sleep
        self.grab_small = grab_small
        assert grab_small is None or rule == "cells"
        self.describe = describe              # diagnostics: front apps when a frame is sent
        assert rule in ("cells", "mvp")
        self.rule = rule                      # "cells" (default) or "mvp" (any changed pixel)
        self.last_sent_small: Optional[np.ndarray] = None
        self.cmask = None
        self.cstats = {"polls": 0, "animated_cells": 0}
        self.frame_id = 0
        self.handed: list[dict] = []
        self._stop = threading.Event()
        self.last_sent: Optional[np.ndarray] = None
        self.mask = None
        self._force = threading.Event()

    def stop(self):
        self._stop.set()
        self._force.set()

    def capture_now(self):
        """Talk key pressed (ARCHITECTURE.md §8.4): capture right away and hand the frame over if it
        changed since the last one sent, skipping the settle wait. The GPU is idle while you talk."""
        self._force.set()

    def step(self, state: dict) -> float:
        """One poll. Returns how long to sleep before the next poll. `state` carries pending change info."""
        if self.rule == "cells":
            return self._step_cells(state)
        return self._step_mvp(state)

    def _step_cells(self, state: dict) -> float:
        """Read only changes worth reading (ARCHITECTURE.md §8.4): count changed token-sized cells
        against the last frame SENT (flicker that reverts cancels out), ignore animated cells, and
        send when enough changed and it has been quiet for a while, on a structural change after a
        short settle, after the staleness cap, or at once on the talk key."""
        t = self.clock()
        if self.grab_small is not None:
            img, small = None, self.grab_small()      # full frame only if this poll hands one over
        else:
            img = self.grab()
            small = shrink(img)

        def full():
            return img if img is not None else self.grab()
        self.cstats["polls"] += 1
        grid = (small.shape[0] // CELL_PX, small.shape[1] // CELL_PX)
        if self.cmask is None or self.cmask.shape != grid:
            self.cmask = cell_mask(*grid)
        if self.last_sent_small is None or self.last_sent_small.shape != small.shape:
            self._send(full(), None, t, reason="first", small=small)
            state.clear()
            state["prev_small"] = small
            return IDLE_POLL_S
        prev = state.get("prev_small")
        if prev is None or prev.shape != small.shape:
            prev = self.last_sent_small                   # no earlier poll: compare with the last frame sent
        state["prev_small"] = small
        ch_prev = cell_changes(small, prev) & self.cmask
        hist = state.setdefault("hist", [])
        hist.append(ch_prev)
        del hist[:-ANIMATED_OF[1]]
        animated = (np.sum(hist, axis=0) >= ANIMATED_OF[0]) if len(hist) >= ANIMATED_OF[0] else np.zeros_like(ch_prev)
        self.cstats["animated_cells"] = int(animated.sum())
        ch_ref = cell_changes(small, self.last_sent_small) & self.cmask
        if state.pop("force", False):                        # talk key: any change at all, now
            if ch_ref.any():
                self._send(full(), None, t, reason="talk_key",
                           waited=t - state["first"] if state.get("first") else 0.0, small=small)
            state.pop("first", None)
            return IDLE_POLL_S
        ch_ref &= ~animated
        n_ref, n_prev, n = int(ch_ref.sum()), int((ch_prev & ~animated).sum()), int(self.cmask.sum())
        if n_ref == 0:
            state.pop("first", None)
            return IDLE_POLL_S
        state.setdefault("first", t)
        if n_prev:
            state["last_change"] = t
            if n_prev >= STRUCTURAL_FRAC * n:
                state["structural"] = True
        quiet = t - state.get("last_change", state["first"])
        reason = None
        if state.get("structural") and quiet >= STRUCTURAL_SETTLE_S:
            reason = "structural"
        elif n_ref >= min(CELL_MIN, max(1, int(CELL_FRAC * n))) and quiet >= CELL_SETTLE_S:
            reason = "settled"
        elif t - state["first"] >= STALE_CAP_S:
            reason = "stale_cap"
        if reason:
            self._send(full(), None, t, reason=reason, waited=t - state["first"], small=small)
            for k in ("first", "last_change", "structural"):
                state.pop(k, None)
            return IDLE_POLL_S
        return ACTIVE_POLL_S

    def _step_mvp(self, state: dict) -> float:
        t = self.clock()
        img = self.grab()
        arr = np.asarray(img)
        if self.mask is None or self.mask.shape != arr.shape[:2]:
            self.mask = change_mask(*arr.shape[:2])
        ref = state.get("last_seen")
        state["last_seen"] = arr
        if self.last_sent is None:                       # first frame: always hand over
            self._send(img, arr, t, reason="first")
            return IDLE_POLL_S
        if state.pop("force", False):                    # talk key: no settle wait
            if changed(arr, self.last_sent, self.mask):
                self._send(img, arr, t, reason="talk_key",
                           waited=t - state["first_change"] if state.get("first_change") else 0.0)
            state["first_change"] = None
            return IDLE_POLL_S
        if state.get("first_change") is None:
            if changed(arr, self.last_sent, self.mask):
                state["first_change"] = t
                state["last_change"] = t
                return ACTIVE_POLL_S
            return IDLE_POLL_S
        # a change is pending: settled, or out of time?
        if ref is not None and changed(arr, ref, self.mask):
            state["last_change"] = t
        settled = t - state["last_change"] >= SETTLE_S
        overdue = t - state["first_change"] >= MAX_DELAY_S
        if settled or overdue:
            if changed(arr, self.last_sent, self.mask):  # it may have changed back
                self._send(img, arr, t, reason="settled" if settled else "max_delay",
                           waited=t - state["first_change"])
            state["first_change"] = None
            return IDLE_POLL_S
        return ACTIVE_POLL_S

    def _send(self, img, arr, t, reason, waited=0.0, small=None):
        self.frame_id += 1
        self.last_sent = arr
        if small is not None:
            self.last_sent_small = small
        d = self.describe() if self.describe else None
        note = d.get("line") if isinstance(d, dict) else None
        self.handed.append({"frame_id": self.frame_id, "captured_at": t, "reason": reason,
                            "waited_s": round(waited, 3),
                            "apps": d.get("apps") if isinstance(d, dict) else d, "note": note})
        if note:
            self.on_frame(self.frame_id, t, img, note=note)
        else:
            self.on_frame(self.frame_id, t, img)

    def run(self):
        state: dict = {}
        while not self._stop.is_set():
            if self._force.is_set() and not self._stop.is_set():
                self._force.clear()
                state["force"] = True
            delay = self.step(state)
            if self.sleep is time.sleep:                 # real use: wake early on the talk key
                self._force.wait(delay)
            else:
                self.sleep(delay)
