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
    w = ScreenWatcher(rule="mvp", on_frame=lambda i, t, img: got.append((i, t)),
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


def test_talk_key_hands_over_a_change_without_settling():
    """capture_now (talk key): a changed frame goes to the reader at once, no 0.3 s settle wait."""
    import numpy as np
    from PIL import Image
    import screen as scr
    frames = [np.zeros((200, 300, 3), np.uint8), np.zeros((200, 300, 3), np.uint8)]
    frames[1][150:160, 10:20] = 255                     # a small change, not settled yet
    it = iter(frames)
    sent = []
    w = scr.ScreenWatcher(rule="mvp", on_frame=lambda fid, t, img: sent.append(fid),
                          grab=lambda: Image.fromarray(next(it)), clock=lambda: 100.0, sleep=lambda s: None)
    state = {}
    w.step(state)                                        # first frame: always handed over
    state["force"] = True
    w.step(state)
    assert sent == [1, 2] and w.handed[-1]["reason"] == "talk_key"


# ---- "cells" rule (default): only changes worth a read

class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def _cells_watcher(frames):
    import screen as scr
    from PIL import Image
    clock, sent = _Clock(), []
    state = {"i": 0}
    def grab():
        return Image.fromarray(frames[min(state["i"], len(frames) - 1)])
    w = scr.ScreenWatcher(on_frame=lambda fid, t, img: sent.append((fid, w.handed[-1]["reason"])),
                          grab=grab, clock=clock, sleep=lambda s: None)
    return w, clock, sent, state


def _run(w, clock, state, seconds, step=0.3, frame_at=None):
    st = state.setdefault("watcher_state", {})       # the watcher's state persists across calls
    end = clock.t + seconds
    while clock.t < end:
        if frame_at:
            state["i"] = frame_at(clock.t)
        w.step(st)
        clock.t += step
    return st


def _frame(edits=(), size=(1024, 1536)):
    import numpy as np
    f = np.full((*size, 3), 255, np.uint8)
    for (y, x, h, wd) in edits:
        f[y:y + h, x:x + wd] = 0
    return f


def test_cells_ignore_a_typed_word():
    base = _frame()
    word = _frame([(500, 300, 20, 60)])                 # a word: ~2 cells
    w, clock, sent, state = _cells_watcher([base, word])
    _run(w, clock, state, 0.3)                           # first frame
    _run(w, clock, state, 10, frame_at=lambda t: 1)
    assert [r for _, r in sent] == ["first"]


def test_cells_read_a_paragraph_after_quiet():
    base = _frame()
    para = _frame([(400 + 30 * i, 100, 20, 1200) for i in range(4)])   # 4 lines across the screen
    w, clock, sent, state = _cells_watcher([base, para])
    _run(w, clock, state, 0.3)
    _run(w, clock, state, 3, frame_at=lambda t: 1)
    assert [r for _, r in sent] == ["first", "settled"]


def test_cells_structural_change_reads_fast():
    import numpy as np
    base = _frame()
    other = np.zeros_like(base)                          # window switch: everything changes
    w, clock, sent, state = _cells_watcher([base, other])
    _run(w, clock, state, 0.3)
    t0 = clock.t
    _run(w, clock, state, 1.2, frame_at=lambda t: 1)
    assert sent[-1][1] == "structural" and w.handed[-1]["captured_at"] - t0 <= 1.0


def test_cells_animation_is_ignored_and_never_settles_into_reads():
    a = _frame([(300, 300, 200, 300)])                    # a video region alternating every poll
    b = _frame([(300, 300, 100, 300)])
    w, clock, sent, state = _cells_watcher([a, b])
    _run(w, clock, state, 0.3)
    _run(w, clock, state, 20, frame_at=lambda t: int(t * 10 / 3) % 2)
    assert [r for _, r in sent].count("settled") == 0 and len(sent) <= 2


def test_cells_staleness_cap_reads_small_lasting_changes():
    base = _frame()
    word = _frame([(500, 300, 20, 60)])
    w, clock, sent, state = _cells_watcher([base, word])
    _run(w, clock, state, 0.3)
    _run(w, clock, state, 27, step=1.0, frame_at=lambda t: 1)
    assert [r for _, r in sent] == ["first", "stale_cap"]


def test_cells_talk_key_sends_even_a_tiny_change():
    base = _frame()
    word = _frame([(500, 300, 20, 60)])
    w, clock, sent, state = _cells_watcher([base, word])
    _run(w, clock, state, 0.3)
    state["i"] = 1
    st = state.setdefault("watcher_state", {})
    st["force"] = True
    w.step(st)
    assert sent[-1][1] == "talk_key"


def test_window_summary_marks_coverage_and_skips_strips():
    import screen as scr
    wins = [{"app": "Terminal", "x": 0, "y": 33, "w": 1512, "h": 68},          # tab strip: skipped
            {"app": "Terminal", "x": 0, "y": 100, "w": 800, "h": 800},
            {"app": "Google Chrome", "x": 0, "y": 33, "w": 1512, "h": 876},
            {"app": "Slack", "x": 100, "y": 150, "w": 600, "h": 600}]
    s = scr.window_summary(wins)
    assert s["apps"] == ["Terminal", "Google Chrome", "Slack"]
    assert s["line"] == ("Open windows, front to back: Terminal; Google Chrome (partly covered); "
                         "Slack (hidden behind others).")


def test_stream_path_decides_like_the_grab_path():
    """ScreenWatcher(grab_small=...) (StreamCapturer) must hand over the same frames at the same times as
    grab+shrink, and take a full-resolution grab only for frames it hands over."""
    import numpy as np
    import screen as scr
    from PIL import Image
    base = _frame()
    para = _frame([(400 + 30 * i, 100, 20, 1200) for i in range(4)])
    other = np.zeros_like(base)
    word = _frame([(500, 300, 20, 60)])
    seq = lambda t: 0 if t < 1002 else 1 if t < 1008 else 2 if t < 1015 else 3   # noqa: E731

    def run(stream):
        clock, sent, full_grabs, i = _Clock(), [], [0], {"i": 0}

        def grab():
            full_grabs[0] += 1
            return Image.fromarray([base, para, other, word][i["i"]])
        small = (lambda: scr.shrink(Image.fromarray([base, para, other, word][i["i"]]))) if stream else None
        w = scr.ScreenWatcher(on_frame=lambda fid, t, img: sent.append((round(t, 1), w.handed[-1]["reason"], img.size)),
                              grab=grab, clock=clock, sleep=lambda s: None, grab_small=small)
        st = {}
        while clock.t < 1050:
            i["i"] = seq(clock.t)
            clock.t += w.step(st)
        return sent, full_grabs[0], w.cstats["polls"]

    a, grabs_a, polls_a = run(False)
    b, grabs_b, polls_b = run(True)
    assert a == b
    assert [r for _, r, _ in a] == ["first", "settled", "structural", "structural"]   # black -> word is a switch too
    assert grabs_a == polls_a and grabs_b == len(b)
