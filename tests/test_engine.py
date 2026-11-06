"""Tests for engine.py.

Fast (default): stop rule, sentence splitting, GPU worker job policy with a fake model.
Slow (-m slow): real Gemma 4 E4B — fixed split joins exactly, cached == uncached on the
20 M0 screen questions (review D2), cancel stops within one token (D6).
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import engine
from engine import count_sentences, split_complete_sentences

ROOT = Path(__file__).resolve().parents[1]
M0 = ROOT / "m0_screen"
M0_INSTRUCTION = "Answer the question about this screenshot. Reply with only the answer, no explanation."


# ---------------------------------------------------------------- fast

@pytest.mark.parametrize("text,n", [
    ("Hello there.", 1),
    ("The rate is 0.0825 percent.", 1),        # decimal point is not a sentence end
    ("One. Two! Three? Four.", 4),
    ("No ending", 0),
    ("Version 3.5 is out. It works.", 2),
])
def test_count_sentences(text, n):
    assert count_sentences(text) == n


def test_split_complete_sentences_keeps_unfinished_tail():
    done, rest = split_complete_sentences("It is 3.5 hours. That is 210 min")
    assert done == ["It is 3.5 hours."]
    assert rest == " That is 210 min"
    # a trailing period with nothing after it is not yet complete (could be "3." then "5")
    done, rest = split_complete_sentences("It is 3.")
    assert done == [] and rest == "It is 3."


class FakeAdapter:
    """Stands in for GemmaAdapter: a read takes `read_s`; generate yields a fixed answer."""
    def __init__(self, read_s=0.3):
        self.read_s, self.reads_started, self.last_stop = read_s, [], None
    def load(self): return self
    def set_budget(self, n): self.budget = n
    def read_screen(self, image, frame_id=0, captured_at=None):
        self.reads_started.append((frame_id, time.time()))
        time.sleep(self.read_s)
        return engine.Snapshot(frame_id, captured_at or time.time(), time.time(), time.time(), 560, 10, master=[])
    def suffix_ids(self, q): return [1, 2, 3]
    def generate(self, cache, ids, cancel=None, on_first_token=None, **kw):
        if on_first_token: on_first_token()
        for w in ["The answer ", "is four. ", "Second sentence. ", "Third one. "]:
            if cancel is not None and cancel.is_set():
                self.last_stop = "cancel"; return
            time.sleep(getattr(self, "piece_delay", 0.01))
            yield w
        self.last_stop = "sentences"


class FakeTts:
    sample_rate = 24000
    def load(self): return self
    def synth(self, sentence, cancel=None, **kw):
        yield np.ones(240, np.float32) * 0.1


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(engine, "clone_cache", lambda c: list(c))
    w = engine.GpuWorker(FakeAdapter(), FakeTts(), budget=560)
    w.start()
    assert w.ready.wait(5)
    yield w
    w.stop()


def _job(q="what?"):
    marks = {}
    audio = []
    return engine.AnswerJob(q, emit_audio=lambda a, sr: audio.append(a),
                            mark=lambda n: marks.setdefault(n, time.perf_counter())), marks, audio


def test_no_read_starts_while_turn_active(worker):
    worker.set_turn_active(True)
    worker.submit_frame(1, time.time(), "img")
    time.sleep(0.5)
    assert worker.adapter.reads_started == []          # naive policy: blocked during a turn
    worker.set_turn_active(False)
    assert worker.first_snapshot_wait(3)
    assert [f for f, _ in worker.adapter.reads_started] == [1]


def test_latest_frame_wins(worker):
    worker.set_turn_active(True)
    for i in range(1, 4):
        worker.submit_frame(i, time.time(), "img")
    worker.set_turn_active(False)
    assert worker.first_snapshot_wait(3)
    assert [f for f, _ in worker.adapter.reads_started] == [3]
    assert worker.stats["frames_dropped"] == 2


def test_answer_waits_behind_inflight_read_and_logs_it(worker):
    worker.submit_frame(1, time.time(), "img")
    assert worker.first_snapshot_wait(3)
    worker.adapter.read_s = 0.6
    worker.submit_frame(2, time.time(), "img")
    time.sleep(0.1)                                    # read of frame 2 is now running
    job, marks, audio = _job()
    worker.submit_answer(job)
    assert job.done.wait(5)
    r = job.result
    assert r["read_in_flight_at_submit"] is True
    assert r["wait_behind_read_s"] >= 0.3              # it waited for the uninterruptible read
    assert r["frame_id"] == 2                          # and used the newest completed read
    assert "turn_complete" in marks and audio


def test_interrupt_stops_generation_and_tts(worker):
    worker.submit_frame(1, time.time(), "img")
    assert worker.first_snapshot_wait(3)
    worker.adapter.piece_delay = 0.15                   # slow enough to interrupt mid-answer
    job, marks, audio = _job()
    worker.submit_answer(job)
    deadline = time.time() + 3
    while "first_sentence_ready" not in marks and time.time() < deadline:
        time.sleep(0.01)
    t_int = worker.interrupt()
    assert job.done.wait(3)
    assert job.result["cancelled"] is True
    assert job.result["sentences_spoken"] < 3           # the rest of the answer was dropped
    assert marks["turn_complete"] - t_int < 0.3         # worker freed within about one piece


def test_answer_without_screen_reports_error(monkeypatch):
    monkeypatch.setattr(engine, "clone_cache", lambda c: list(c))
    w = engine.GpuWorker(FakeAdapter(), FakeTts())
    w.start(); w.ready.wait(5)
    job, _, _ = _job()
    w.submit_answer(job)
    assert job.done.wait(3)
    assert job.result["error"] == "no screen read yet"
    w.stop()


# ---------------------------------------------------------------- slow (real model)

@pytest.fixture(scope="module")
def gemma():
    a = engine.GemmaAdapter(instruction=M0_INSTRUCTION).load()
    a.set_budget(560)
    return a


def _m0_questions():
    return json.loads((M0 / "questions.json").read_text())


@pytest.mark.slow
def test_fixed_split_joins_exactly(gemma):
    """prefix tokens + question tokens == tokens of the full prompt, for every M0 question."""
    by_screen = {}
    for q in _m0_questions():
        by_screen.setdefault(q["screen"], []).append(q)
    for screen, qs in by_screen.items():
        img = str(M0 / f"screens/{screen}.png")
        pre = gemma.prefix_ids(img)
        for q in qs:
            assert pre + gemma.suffix_ids(q["question"]) == gemma.full_ids(img, q["question"]), q["question"]


@pytest.mark.slow
def test_cached_answers_match_uncached(gemma):
    """Review D2: the pre-read path must give the same answers as the normal one-pass path."""
    mismatches, rows = [], []
    by_screen = {}
    for q in _m0_questions():
        by_screen.setdefault(q["screen"], []).append(q)
    for screen, qs in by_screen.items():
        img = str(M0 / f"screens/{screen}.png")
        snap = gemma.read_screen(img)
        for q in qs:
            a = gemma.answer_cached(snap, q["question"], max_tokens=40)
            b = gemma.answer_uncached(img, q["question"], max_tokens=40)
            rows.append((q["question"], a, b))
            if a.strip() != b.strip():
                mismatches.append((q["question"], a, b))
    print("\n".join(f"{q[:40]!r}: cached={a!r} uncached={b!r}" for q, a, b in rows))
    assert not mismatches, mismatches


@pytest.mark.slow
def test_cancel_stops_within_one_token(gemma):
    img = str(M0 / "screens/doc.png")
    snap = gemma.read_screen(img)
    cancel = threading.Event()
    n = 0
    for _ in gemma.generate(engine.clone_cache(snap.master), gemma.suffix_ids("Describe this page in detail."),
                            cancel=cancel, max_tokens=120, max_sentences=99):
        n += 1
        if n == 3:
            cancel.set()
    assert gemma.last_stop == "cancel"
    assert n <= 4


# -- latency fixes (2026-11-06): early first phrase + Kokoro silence trim

def test_first_phrase_splits_at_comma_after_enough_words():
    assert engine.split_first_phrase("Hello, there is a typo") is None              # comma too early, < 7 words
    assert engine.split_first_phrase("The file looks fine, but line ")[0] == "The file looks fine,"


def test_first_phrase_splits_before_break_word():
    ph = engine.split_first_phrase("The capital of Australia is Canberra and ")
    assert ph == ("The capital of Australia", "is Canberra and ")


def test_first_phrase_waits_when_short_or_no_break():
    assert engine.split_first_phrase("There are ninety minutes") is None
    assert engine.split_first_phrase("Octopuses really really really truly deeply amazingly ") is None


def test_first_phrase_ignores_unfinished_word():
    assert engine.split_first_phrase("A compiler translates source code written in") is None  # 'in' not complete yet


def test_trim_silence_keeps_short_pad():
    sr = 24000
    a = np.concatenate([np.zeros(int(0.3 * sr)), np.ones(sr) * 0.1, np.zeros(int(0.45 * sr))]).astype(np.float32)
    t = engine.trim_silence(a, sr, lead_keep_ms=30, tail_keep_ms=120)
    assert abs(len(t) / sr - (1.0 + 0.03 + 0.12)) < 0.002
    assert engine.trim_silence(np.zeros(100, np.float32), sr).size == 100       # all-silent: untouched
