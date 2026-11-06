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
MENU_BAR_PX = 80            # Retina menu-bar height incl. margin (device pixels)
STATUS_AREA_FRAC = 0.30     # right part of the menu bar: clock, battery, Wi-Fi, etc.


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


def change_mask(h: int, w: int) -> np.ndarray:
    """True where changes count. Masks the right part of the menu bar (status items, clock)."""
    m = np.ones((h, w), bool)
    m[:MENU_BAR_PX, int(w * (1 - STATUS_AREA_FRAC)):] = False
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
    """

    def __init__(self, on_frame: Callable, grab: Callable[[], Image.Image],
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        super().__init__(daemon=True, name="screen-watcher")
        self.on_frame, self.grab, self.clock, self.sleep = on_frame, grab, clock, sleep
        self.frame_id = 0
        self.handed: list[dict] = []
        self._stop = threading.Event()
        self.last_sent: Optional[np.ndarray] = None
        self.mask = None

    def stop(self):
        self._stop.set()

    def step(self, state: dict) -> float:
        """One poll. Returns how long to sleep before the next poll. `state` carries pending change info."""
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

    def _send(self, img, arr, t, reason, waited=0.0):
        self.frame_id += 1
        self.last_sent = arr
        self.handed.append({"frame_id": self.frame_id, "captured_at": t, "reason": reason,
                            "waited_s": round(waited, 3)})
        self.on_frame(self.frame_id, t, img)

    def run(self):
        state: dict = {}
        while not self._stop.is_set():
            delay = self.step(state)
            self.sleep(delay)
