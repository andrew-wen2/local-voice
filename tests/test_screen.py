"""Tests for screen.py: change filter (D20), settle + max delay, privacy of captures (D4)."""
import os
import stat
from pathlib import Path

import numpy as np
from PIL import Image

import screen
from screen import Capturer, ScreenWatcher, change_mask, changed

H, W = 400, 600


def frame(seed=0, edits=()):
    a = np.full((H, W, 3), 240, np.uint8)
    a[100:110, 50:400] = 20                      # a "line of text"
    for (y, x) in edits:
        a[y, x] = (0, 0, 0)
    return a


def test_one_pixel_edit_counts_as_change():
    a, b = frame(), frame(edits=[(200, 300)])      # a one-character-sized edit
    assert changed(a, b, change_mask(H, W))


def test_status_area_changes_are_ignored():
    a = frame()
    b = frame(edits=[(10, W - 20), (40, W - 100)])  # clock / battery in the right of the menu bar
    assert not changed(a, b, change_mask(H, W))
    c = frame(edits=[(10, 20)])                     # left of the menu bar (app menus) still counts
    assert changed(a, c, change_mask(H, W))


class FakeClock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def sleep(self, s): self.t += s


def run_watcher(frames_at):
    """frames_at(t) -> np array. Steps the watcher on a fake clock for 12 s; returns handed frames."""
    clk = FakeClock()
    got = []
    w = ScreenWatcher(on_frame=lambda i, t, img: got.append((i, t)),
                      grab=lambda: Image.fromarray(frames_at(clk.t)), clock=clk, sleep=clk.sleep)
    state = {}
    while clk.t < 1012:
        clk.sleep(w.step(state))
    return w.handed


def test_change_is_handed_over_after_it_settles():
    base, edited = frame(), frame(edits=[(200, 300)])
    handed = run_watcher(lambda t: edited if t >= 1003.0 else base)
    assert [h["reason"] for h in handed] == ["first", "settled"]
    assert 0.25 <= handed[1]["waited_s"] <= 1.0


def test_never_settling_screen_still_read_within_3s():
    def typing(t):
        if t < 1002:
            return frame()
        return frame(edits=[(200, 50 + int(t * 10) % 500)])   # changes on every poll
    handed = run_watcher(typing)
    reasons = [h["reason"] for h in handed]
    assert reasons[0] == "first" and "max_delay" in reasons
    assert all(h["waited_s"] <= screen.MAX_DELAY_S + screen.ACTIVE_POLL_S for h in handed[1:])


def test_change_and_back_is_not_handed_over():
    base, edited = frame(), frame(edits=[(200, 300)])
    handed = run_watcher(lambda t: edited if 1003.0 <= t < 1003.5 else base)
    assert [h["reason"] for h in handed] == ["first"]


def test_capture_dir_private_and_file_deleted(monkeypatch, tmp_path):
    cap = Capturer(fmt="png")
    assert stat.S_IMODE(os.stat(cap.dir).st_mode) == 0o700

    def fake_run(cmd, **kw):                       # stand in for screencapture: write a file
        Image.fromarray(frame()).save(cmd[-1])
    monkeypatch.setattr(screen.subprocess, "run", fake_run)
    img = cap.grab()
    assert img.size == (W, H)
    assert list(cap.dir.iterdir()) == []           # nothing left on disk
    cap.cleanup()
    assert not cap.dir.exists()


def test_gitignore_covers_private_paths():
    gi = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert "logs/" in gi and "captures/" in gi
