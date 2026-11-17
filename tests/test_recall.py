"""Recall mode without the model (ARCHITECTURE.md §5.9): the live page store (recall.py), partial transcripts
(asr.Partials), voice activity (audio_io.Mic), cutting a KV cache back (recall_backend.cut), and the GPU worker's
event flow on the simulated backend (recall_worker.py + scheduler.SimBackend)."""
import threading
import time

import mlx.core as mx
import numpy as np
import pytest
from test_kv_reuse import Lay, close, feed, make, window_state

from asr import Partials
from audio_io import Mic
from recall import Ocr, Store
from recall_backend import TurnIO, cut, window_snapshot
from scheduler import FakeClock, SimBackend


# ---------------------------------------------------------------- the store

def test_visits_versions_and_timeline():
    s = Store()
    assert s.add(0, "A", "Page A", "a1\na2") is None
    assert s.add(5, "A", "Page A", "a2\na3") is None          # scrolled: same visit, new lines
    assert s.add(10, "B", "Page B", "b1") == "A"               # A's visit ended
    assert s.add(20, "A", "Page A", "a4") == "B"
    assert s.add(30, "B", "Page B", "b1") == "A"
    assert s.visits == [("A", 0, 10), ("B", 10, 20), ("A", 20, 30)]
    assert s.version == {"A": 2, "B": 1} and s.current_key() == "B"
    a = s.m.pages["A"]
    assert a.blocks == ["a1", "a2", "a3", "a4"] and a.dwell_s == 20 and a.first_s == 0 and a.last_s == 30
    assert s.timeline().count("\n") == 3 and "(10 s) Page A" in s.timeline()
    assert "2 visits" in s.head("A") and s.page_text("A", 2).endswith("a1\na2")


def test_pick_is_oldest_first_and_names_versions():
    s = Store()
    for t, (k, text) in enumerate([("A", "alpha cobalt"), ("B", "beta cobalt cobalt"), ("C", "gamma")]):
        s.add(10 * t, k, f"Page {k}", text)
    s.add(40, "D", "Page D", "delta")
    slots = s.pick("cobalt", cap=4000, count=lambda x: len(x.split()))
    assert slots == [("A", 1, 1), ("B", 1, 1)]                  # B scores higher, A was first seen first


def test_frame_to_page_key_follows_the_eval_rules():
    s = Store()
    chrome = {"id": 7, "app": "Google Chrome", "title": "(3) Inbox 📬", "x": 0, "y": 0, "w": 1000, "h": 800}
    lines = ["Inbox", "mail.example.com/inbox", "Message one", "Message two"]
    boxes = [[10, 10, 50, 10], [100, 60, 200, 12], [100, 200, 200, 12], [100, 300, 200, 12]]
    s.add_frame(0, [chrome], lines, boxes)
    term = {"id": 9, "app": "Terminal", "title": "zsh", "x": 0, "y": 0, "w": 800, "h": 600}
    s.add_frame(5, [term], ["$ ls"], [[10, 100, 40, 12]])
    assert s.visits == [("Inbox", 0, 5)]
    assert s.m.pages["Inbox"].blocks == ["Message one", "Message two"]      # below the address bar only
    assert s.current_key() == "Terminal window 9"


def test_ocr_waits_for_quiet_and_skips_frames_taken_mid_switch():
    s, ended, quiet = Store(), [], threading.Event()
    w1 = [{"id": 1, "app": "Notes", "title": "n", "x": 0, "y": 0, "w": 500, "h": 500}]
    w2 = [{"id": 2, "app": "Notes", "title": "m", "x": 0, "y": 0, "w": 500, "h": 500}]
    o = Ocr(s, quiet=quiet.is_set, on_visit_end=ended.append, ocr=lambda img: (["line"], [[10, 10, 10, 10]]), t0=0.0)
    o.start()
    o.submit(1.0, None, w1, w1)
    o.submit(2.0, None, w1, w2)              # the front window changed across the grab
    o.submit(3.0, None, w2, w2)
    time.sleep(0.3)
    assert o.stats["frames"] == 0            # a turn is active
    quiet.set()
    o.poke()
    for _ in range(50):
        if o.stats["frames"] == 2:
            break
        time.sleep(0.02)
    o.stop()
    assert o.stats == {"frames": 2, "skipped_mid_switch": 1, "ms": o.stats["ms"], "kept": 0}
    assert ended == ["Notes window 1"] and s.visits == [("Notes window 1", 1.0, 3.0)]


# ---------------------------------------------------------------- speech while the key is held

def test_mic_voice_activity():
    m = Mic()
    m.begin()
    quiet, loud = np.zeros(320, np.float32), 0.1 * np.ones(320, np.float32)
    m.feed(quiet)
    assert m.silent_for() is None and m.voiced_until() == -1
    m.feed(loud)
    m.feed(loud)
    for _ in range(10):
        m.feed(quiet)
    assert m.voiced_until() == 960 and m.silent_for() == pytest.approx(3200 / 16000)
    assert len(m.audio()) == 13 * 320 and len(m.end()) == 13 * 320


class FakeAsr:
    def __init__(self):
        self.calls = 0

    def transcribe(self, x):
        self.calls += 1
        words = ["what", "was", "the", "price", "of", "the", "lamp"]
        return {"text": " ".join(words[:len(x) // 8000]), "seconds": 0.04}


def test_partials_and_final_from_partial():
    audio, voiced, got = [np.zeros(0, np.float32)], [-1], []
    p = Partials(FakeAsr(), lambda: audio[0], lambda: voiced[0], got.append, min_new_s=0.25, min_audio_s=0.5)
    assert not p.step()                                       # nothing yet
    audio[0], voiced[0] = np.zeros(16000, np.float32), 16000
    assert p.step() and got == ["what was"]
    audio[0] = np.zeros(16000 + 2000, np.float32)             # 0.125 s new: below min_new
    assert not p.step()
    audio[0], voiced[0] = np.zeros(32000, np.float32), 30000
    assert p.step() and got == ["what was", "what was the price"]
    assert p.final_text() == "what was the price"             # every voiced sample was covered
    voiced[0] = 33000                                         # voice after the last partial
    assert p.final_text() is None


def test_a_partial_runs_when_speech_stops_so_the_final_is_skipped():
    """M3 (2026-11-13): 0.1 s after the last voiced block, a partial runs even with less than min_new of new audio,
    so with a pause before key up the last partial covers every voiced sample (no final transcription)."""
    audio, voiced, got = [np.zeros(16000, np.float32)], [16000], []
    p = Partials(FakeAsr(), lambda: audio[0], lambda: voiced[0], got.append, every_s=10.0, tick_s=0.01)
    assert p.step()                                           # a partial at 1.0 s
    audio[0], voiced[0] = np.zeros(16000 + 1600, np.float32), 16000 + 1600   # 0.1 s more speech, still talking
    assert not p.settled(len(audio[0])) and p.final_text() is None
    audio[0] = np.zeros(16000 + 1600 + 1600, np.float32)      # then 0.1 s of silence
    assert p.settled(len(audio[0]))
    p.start()
    deadline = time.time() + 2
    while p.final_text() is None and time.time() < deadline:
        time.sleep(0.01)
    p.finish()
    assert p.final_text() is not None and p.stats["settled"] == 1   # once: then the partial covers the speech


# ---------------------------------------------------------------- cutting a cache back

@pytest.mark.parametrize("chunk", [1, 3, 5])
def test_cut_restores_an_earlier_state(chunk):
    lay = Lay()
    base = list(range(1, 14))                                 # past the window (W = 8)
    c = feed(lay, make(), base)
    snap = window_snapshot(lay, c)
    feed(lay, c, list(range(30, 41)), chunk=chunk)            # a page that gets dropped
    cut(lay, c, len(base), snap)
    more = list(range(50, 57))
    feed(lay, c, more, chunk=chunk)
    d = feed(lay, make(), base + more, chunk=chunk)
    for i, (x, y) in enumerate(zip(c, d)):
        a, b = (window_state(x), window_state(y)) if lay.window[i] else (x.state, y.state)
        assert int(x.offset) == int(y.offset)
        assert close(a[0], b[0]) and close(a[1], b[1])


def test_snapshot_is_a_copy():
    lay = Lay()
    c = feed(lay, make(), list(range(1, 11)))
    snap = window_snapshot(lay, c)
    before = mx.array(snap[0][0])
    feed(lay, c, list(range(20, 30)), chunk=1)
    assert close(snap[0][0], before)


# ---------------------------------------------------------------- the worker's event flow

T = {"prep": .001, "patch": .001, "vision": .001, "pool": .001, "embed": .001, "text": .001,
     "page_layer": .001, "q_layers": .001, "assemble": .001, "decode": .001}


def test_worker_turn_on_the_simulated_backend():
    from recall_worker import RecallWorker
    s = Store()
    s.add(0, "A", "Lamp shop", "brass lamp 40 dollars")
    s.add(10, "B", "Train times", "the 8:15 to Boston")
    s.add(20, "C", "Notes", "todo")
    be = SimBackend(FakeClock(), T)
    w = RecallWorker(None, None, s, backend_factory=lambda: be)
    w.start()
    assert w.ready.wait(5)
    w.visit_ended("A")
    w.visit_ended("B")
    w.frame(0, object())
    assert w.wait_ready(5)
    w.key_down()
    w.partial("what was the price of the brass")
    io = TurnIO(emit_audio=lambda *a: None, mark=lambda n: None)
    w.key_up(io)
    w.final("what was the price of the brass lamp")
    assert io.done.wait(5)
    w.stop()
    w.join(5)
    r = io.result
    # the partial is retrieved only if the loop reaches it before key up (M3 2026-11-15: Retrieve job); the final always
    assert r["cancelled"] is False and r["release_s"] >= 0 and r["retrievals"] in (1, 2)
    assert ("build", ("A", 1, 1)) in be.calls and ("build", ("B", 1, 1)) in be.calls
    assert w.sched.ex_done == 1


# ---------------------------------------------------------------- §12.13: OCR rows and timeline retrieval

def test_ocr_rows_join_cells_on_one_visual_row():
    from recall import ocr_rows
    lines = ["Parking", "$24/day", "Breakfast", "$18", "A long sentence of prose on its own line."]
    boxes = [[10, 100, 60, 20], [400, 103, 50, 18], [10, 130, 80, 20], [400, 129, 30, 20], [10, 170, 500, 20]]
    assert ocr_rows(lines, boxes) == ["Parking | $24/day", "Breakfast | $18", "A long sentence of prose on its own line."]


def test_clock_times_take_the_reading_inside_the_hour():
    from datetime import datetime
    from recall import clock_times
    start = datetime(1900, 1, 1, 14, 0)                       # 2:00 PM, an hour long
    assert clock_times("What was I looking at around 2:43?", start, 3600) == [43 * 60]
    assert clock_times("What was on screen at 2 pm?", start, 3600) == [0]
    assert clock_times("between 2:10 and 2:15", start, 3600) == [600, 900]
    assert clock_times("What was the rent for Apt 3B, v2.6?", start, 3600) == []


def test_timeline_excerpt_has_the_neighbours_and_the_time():
    from recall import Store
    s = Store()
    for i, (k, text) in enumerate([("A", "alpha"), ("B", "beta"), ("C", "lamp shop"), ("D", "delta"), ("E", "echo"),
                                   ("F", "foxtrot"), ("G", "golf")]):
        s.add(60 * i, k, f"Page {k}", text)
    s.add(420, "H", "Page H", "hotel")
    ex = s.excerpt("what was right before the lamp shop?")
    assert [ln.split(") ")[-1] for ln in ex.split("\n")[1:]] == ["Page A", "Page B", "Page C", "Page D", "Page E"]
    t = s.at(360)                                             # Page G's visit
    ex = s.excerpt(f"what was I doing around {t.hour % 12 or 12}:{t.minute:02d}?")
    assert "Page G" in ex and "Page A" not in ex


def test_ocr_keeps_frames_with_new_content_and_never_calls():
    from PIL import Image
    s, kept, quiet = Store(), [], threading.Event()
    quiet.set()
    page = [{"id": 1, "app": "Notes", "title": "n", "x": 0, "y": 0, "w": 500, "h": 500}]
    call = [{"id": 2, "app": "Google Chrome", "title": "Meet - standup", "x": 0, "y": 0, "w": 500, "h": 500}]
    texts = iter([["alpha"], ["alpha"], ["alpha", "beta"], ["gamma"]])
    o = Ocr(s, quiet=quiet.is_set, on_visit_end=lambda k: None, t0=0.0,
            ocr=lambda img: (lines := next(texts), [[10, 10 + 20 * i, 10, 10] for i in range(len(lines))]),
            on_keep=lambda fid, img, meta: kept.append((fid, meta["key"], meta["lines"])))
    o.start()
    img = Image.new("RGB", (512, 512), "white")
    o.submit(1.0, img, page, page, fid=1)        # first frame of the page: kept
    o.submit(2.0, img, page, page, fid=2)        # same text, same pixels: not kept
    o.submit(3.0, img, page, page, fid=3)        # a new line: kept
    o.submit(4.0, img, call, call, fid=4)        # a video call: never kept
    for _ in range(100):
        if o.stats["frames"] == 4:
            break
        time.sleep(0.02)
    o.stop()
    assert [k[0] for k in kept] == [1, 3] and o.stats["kept"] == 2


def test_dropped_frames_give_their_text_to_the_kept_screen():
    from PIL import Image
    s, kept, more, quiet = Store(), [], [], threading.Event()
    quiet.set()
    page = [{"id": 1, "app": "Notes", "title": "n", "x": 0, "y": 0, "w": 500, "h": 500}]
    texts = iter([["alpha"], ["alpha"]])
    o = Ocr(s, quiet=quiet.is_set, on_visit_end=lambda k: None, t0=0.0,
            ocr=lambda img: (lines := next(texts), [[10, 10 + 20 * i, 10, 10] for i in range(len(lines))]),
            on_keep=lambda fid, img, meta: kept.append(fid), on_unit_text=lambda fid, lines: more.append((fid, lines)))
    o.start()
    img = Image.new("RGB", (512, 512), "white")
    o.submit(1.0, img, page, page, fid=1)        # kept
    o.submit(2.0, img, page, page, fid=2)        # dropped: its text goes to screen 1
    for _ in range(100):
        if o.stats["frames"] == 2:
            break
        time.sleep(0.02)
    o.stop()
    assert kept == [1] and more == [(1, ["alpha"])]


def test_clock_times_across_midnight():
    """A session that starts before midnight: "12:04" is 12:04 AM the next day, 6 minutes in (it was ~24 h off)."""
    from datetime import datetime
    from recall import clock_times
    start = datetime(2026, 11, 15, 23, 58)
    assert clock_times("what was I doing around 12:04?", start, 3600) == [360.0]
    assert clock_times("what was on screen at 12:04 am?", start, 3600) == [360.0]
    assert clock_times("what was on screen at 11:59 pm?", start, 3600) == [60.0]
