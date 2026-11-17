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
    w = engine.GpuWorker(FakeAdapter(), FakeTts(), budget=560, policy="p0")   # MVP baseline tests
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


# ---------------------------------------------------------------- scheduler (p1 / p2), fake pieces

class FakeStepAdapter(FakeAdapter):
    """Reads come in `pieces` pieces of `piece_s` seconds each (like read_screen_steps)."""
    def __init__(self, pieces=10, piece_s=0.03):
        super().__init__()
        self.pieces, self.piece_s, self.pieces_run = pieces, piece_s, []
    def read_screen_steps(self, image, frame_id=0, captured_at=None, layers_per_piece=2, **kw):
        self.reads_started.append((frame_id, time.time()))
        for i in range(self.pieces):
            time.sleep(self.piece_s)
            self.pieces_run.append((frame_id, i, time.perf_counter()))
            yield i + 1, self.pieces
        return engine.Snapshot(frame_id, captured_at or time.time(), time.time(), time.time(), 1120, 10, master=[])


def _sched(monkeypatch, **kw):
    monkeypatch.setattr(engine, "clone_cache", lambda c: list(c))
    opts = dict(policy="p2", freshness="A", fresh_ms=300.0)
    opts.update(kw)
    pieces = opts.pop("pieces", 10)
    piece_s = opts.pop("piece_s", 0.03)
    w = engine.GpuWorker(FakeStepAdapter(pieces, piece_s), FakeTts(), **opts)
    w.start()
    assert w.ready.wait(5)
    return w


def _read_first(w, fid=1):
    w.submit_frame(fid, time.time(), "img")
    deadline = time.time() + 5
    while (w.snapshot is None or w.snapshot.frame_id != fid) and time.time() < deadline:
        time.sleep(0.01)
    assert w.snapshot.frame_id == fid


def test_p2_reads_run_during_a_turn(monkeypatch):
    w = _sched(monkeypatch)
    w.set_turn_active(True)                      # you are talking: the GPU is idle, so read
    _read_first(w, 1)
    w.stop()


def test_p2_answer_waits_at_most_one_piece_freshness_c(monkeypatch):
    w = _sched(monkeypatch, freshness="C", pieces=20, piece_s=0.05)
    _read_first(w, 1)
    w.submit_frame(2, time.time(), "img")
    time.sleep(0.12)                             # read of frame 2 is mid-way
    job, marks, _ = _job()
    t_submit = time.perf_counter()
    w.submit_answer(job)
    assert job.done.wait(5)
    r = job.result
    assert marks["restore_done"] - t_submit < 0.05 + 0.03     # at most one piece (+ slack)
    assert r["fresh_decision"] == "answer" and r["frame_id"] == 1 and r["stale_vs_submit"] is True
    assert r.get("read_paused") is True
    deadline = time.time() + 5                   # the paused read resumes and finishes afterwards
    while w.snapshot.frame_id != 2 and time.time() < deadline:
        time.sleep(0.02)
    assert w.snapshot.frame_id == 2 and w.read_log[-1]["pauses"] == 1
    w.stop()


def test_p2_freshness_a_finishes_a_read_that_is_nearly_done(monkeypatch):
    w = _sched(monkeypatch, freshness="A", fresh_ms=300, pieces=10, piece_s=0.03)
    _read_first(w, 1)
    w.submit_frame(2, time.time(), "img")
    time.sleep(0.15)                             # ~5 of 10 pieces left: ~150 ms < 300 ms
    job, _, _ = _job()
    w.submit_answer(job)
    assert job.done.wait(5)
    assert job.result["fresh_decision"] == "read_first"
    assert job.result["frame_id"] == 2 and job.result["stale_vs_submit"] is False
    w.stop()


def test_p2_freshness_a_answers_stale_when_the_read_is_far_from_done(monkeypatch):
    w = _sched(monkeypatch, freshness="A", fresh_ms=300, pieces=40, piece_s=0.03)
    _read_first(w, 1)
    w.submit_frame(2, time.time(), "img")
    time.sleep(0.1)                              # ~37 pieces left: ~1.1 s > 300 ms
    job, _, _ = _job()
    w.submit_answer(job)
    assert job.done.wait(5)
    assert job.result["fresh_decision"] == "answer" and job.result["frame_id"] == 1
    assert job.result["wait_behind_read_s"] < 0.1
    w.stop()


def test_p2_freshness_b_always_waits_for_the_new_screen(monkeypatch):
    w = _sched(monkeypatch, freshness="B", pieces=20, piece_s=0.03)
    _read_first(w, 1)
    w.submit_frame(2, time.time(), "img")
    job, _, _ = _job()
    w.submit_answer(job)
    assert job.done.wait(5)
    assert job.result["fresh_decision"] == "read_first" and job.result["frame_id"] == 2
    w.stop()


def test_p2_latest_frame_drops_a_partial_read(monkeypatch):
    w = _sched(monkeypatch, pieces=20, piece_s=0.03)
    w.submit_frame(1, time.time(), "img")
    time.sleep(0.15)                             # frame 1 read part-way
    w.submit_frame(2, time.time(), "img")
    assert w.first_snapshot_wait(5)
    deadline = time.time() + 5
    while w.snapshot.frame_id != 2 and time.time() < deadline:
        time.sleep(0.02)
    assert w.snapshot.frame_id == 2 and w.stats["reads_dropped"] == 1
    assert [e["frame_id"] for e in w.read_log] == [2]
    w.stop()


def test_p2_read_an_answer_waits_for_is_not_dropped(monkeypatch):
    w = _sched(monkeypatch, freshness="B", pieces=20, piece_s=0.03)
    _read_first(w, 1)
    w.submit_frame(2, time.time(), "img")
    job, _, _ = _job()
    w.submit_answer(job)                         # waits for frame 2
    time.sleep(0.1)
    w.submit_frame(3, time.time(), "img")        # must not restart the read the answer waits for
    assert job.done.wait(5)
    assert job.result["frame_id"] == 2
    w.stop()


def test_p1_answer_waits_for_the_read_in_progress(monkeypatch):
    w = _sched(monkeypatch, policy="p1", pieces=15, piece_s=0.03)
    _read_first(w, 1)
    w.submit_frame(2, time.time(), "img")
    time.sleep(0.1)
    job, _, _ = _job()
    w.submit_answer(job)
    assert job.done.wait(5)
    assert job.result["frame_id"] == 2 and job.result["wait_behind_read_s"] >= 0.2
    w.stop()


def test_rate_cap_spaces_out_read_starts(monkeypatch):
    w = _sched(monkeypatch, min_read_interval_s=0.5, pieces=2, piece_s=0.01)
    _read_first(w, 1)
    w.submit_frame(2, time.time(), "img")
    deadline = time.time() + 5
    while (w.snapshot.frame_id != 2) and time.time() < deadline:
        time.sleep(0.01)
    starts = [t for _, t in w.adapter.reads_started]
    assert starts[1] - starts[0] >= 0.45
    w.stop()


def test_duty_cap_defers_reads_but_not_reads_an_answer_needs(monkeypatch):
    w = _sched(monkeypatch, duty_cap=0.0001, freshness="B", pieces=5, piece_s=0.02)
    w.duty_cap = 1.0
    _read_first(w, 1)                            # some read time is now on the books
    w.duty_cap = 0.0001
    w.submit_frame(2, time.time(), "img")
    time.sleep(0.3)
    assert w.snapshot.frame_id == 1              # deferred by the duty cap
    job, _, _ = _job()
    w.submit_answer(job)                         # B: the answer needs frame 2, caps don't apply
    assert job.done.wait(5)
    assert job.result["frame_id"] == 2
    w.stop()


# ---------------------------------------------------------------- slow (real model)

@pytest.fixture(scope="module")
def gemma():
    if not (M0 / "questions.json").exists():
        pytest.skip("needs the local reading set (m0_screen/, not on GitHub)")
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

def test_first_phrase_splits_only_at_punctuation():
    """A split only where a speaker pauses; otherwise the first sentence is spoken whole."""
    assert engine.split_first_phrase("The rent was $3,488 a month, with ")[0] == "The rent was $3,488 a month,"
    assert engine.split_first_phrase("The file looks fine; but line ")[0] == "The file looks fine;"
    assert engine.split_first_phrase("Yes, the flight leaves ") is None                   # comma before 3 words
    assert engine.split_first_phrase("The capital of Australia is Canberra and ") is None  # no split before "is"
    assert engine.split_first_phrase("Hello, there is a typo", min_words=4) is None


def test_trim_silence_keeps_short_pad():
    sr = 24000
    a = np.concatenate([np.zeros(int(0.3 * sr)), np.ones(sr) * 0.1, np.zeros(int(0.45 * sr))]).astype(np.float32)
    t = engine.trim_silence(a, sr, lead_keep_ms=30, tail_keep_ms=120)
    assert abs(len(t) / sr - (1.0 + 0.03 + 0.12)) < 0.002
    assert engine.trim_silence(np.zeros(100, np.float32), sr).size == 100       # all-silent: untouched
