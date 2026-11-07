"""Conversation history + context jobs in the scheduler (ARCHITECTURE.md §5, §9), with a fake model
whose "cache" is a list of labels, so each test can check the exact context the model would see."""
import time

import numpy as np
import pytest

import engine


class FakeCtxAdapter:
    budget = 1120

    def __init__(self, piece_s=0.01, tok_per_ex=1):
        self.piece_s, self.tok_per_ex = piece_s, tok_per_ex
        self.seen, self.last_stop = {}, None

    def load(self): return self
    def set_budget(self, n): self.budget = n
    def read_pieces_estimate(self, lpp=2): return 8

    def _pieces(self, n):
        for _ in range(n):
            time.sleep(self.piece_s)
            yield None

    def exchange_ids(self, q, a, first_after_screen=False):
        return [("EX1:" if first_after_screen else "EX:") + q] + ["."] * (self.tok_per_ex - 1)

    def prefill_text_steps(self, cache, ids, lpp=2):
        yield from self._pieces(2)
        cache.extend(ids)
        return cache

    def new_prefix_steps(self, pairs, lpp=2):
        cache = ["BOS"]
        for q, a in pairs:
            cache = yield from self.prefill_text_steps(cache, self.exchange_ids(q, a), lpp)
        return cache

    def extend_prefix_steps(self, prefix, pairs, lpp=2):
        cache = list(prefix)
        for q, a in pairs:
            cache = yield from self.prefill_text_steps(cache, self.exchange_ids(q, a), lpp)
        return cache

    def encode_steps(self, image):
        yield from self._pieces(4)
        return (image, "feats")

    def screen_steps(self, prefix, encoded, lpp=2):
        cache = list(prefix) if prefix is not None else ["BOS"]
        yield from self._pieces(2)
        cache.append("SCREEN:" + encoded[0])
        return cache, len(cache)

    def suffix_ids(self, q, tail="screen"):
        return [("Q:" if tail == "screen" else "UQ:") + q]

    def generate(self, cache, ids, cancel=None, on_first_token=None, **kw):
        self.seen[ids[0].split(":", 1)[1]] = [t for t in list(cache) + list(ids) if t != "."]
        if on_first_token:
            on_first_token()
        yield "Fine. "
        self.last_stop = "eos"


class FakeTts:
    sample_rate = 24000
    def load(self): return self
    def synth(self, sentence, cancel=None, **kw):
        yield np.ones(24, np.float32)


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(engine, "clone_cache", lambda c: list(c))
    workers = []

    def _make(**kw):
        ad = FakeCtxAdapter(piece_s=kw.pop("piece_s", 0.01), tok_per_ex=kw.pop("tok_per_ex", 1))
        w = engine.GpuWorker(ad, FakeTts(), policy="p2", **kw)
        w.start()
        assert w.ready.wait(5) and w.history
        workers.append(w)
        return w
    yield _make
    for w in workers:
        w.stop()


def ask(w, q, timeout=5):
    job = engine.AnswerJob(q, emit_audio=lambda a, sr: None, mark=lambda n: None)
    w.submit_answer(job)
    assert job.done.wait(timeout)
    return job.result


def show(w, fid, image):
    w.submit_frame(fid, time.time(), image)
    deadline = time.time() + 5
    while (w.snapshot is None or w.snapshot.frame_id != fid or w.read is not None) and time.time() < deadline:
        time.sleep(0.005)
    assert w.snapshot.frame_id == fid


def test_follow_up_sees_the_previous_exchange(make):
    w = make()
    show(w, 1, "A")
    ask(w, "q1")
    ask(w, "q2")                                     # asked right away: must wait for q1 to be appended
    assert w.adapter.seen["q1"] == ["BOS", "SCREEN:A", "Q:q1"]
    assert w.adapter.seen["q2"] == ["BOS", "SCREEN:A", "EX1:q1", "UQ:q2"]


def test_new_screen_moves_turns_into_older_history(make):
    w = make()
    show(w, 1, "A")
    ask(w, "q1")
    ask(w, "q2")
    show(w, 2, "B")
    ask(w, "q3")
    assert w.adapter.seen["q3"] == ["BOS", "EX:q1", "EX:q2", "SCREEN:B", "Q:q3"]
    show(w, 3, "C")                                  # prefix is extended, not rebuilt
    ask(w, "q4")
    assert w.adapter.seen["q4"] == ["BOS", "EX:q1", "EX:q2", "EX:q3", "SCREEN:C", "Q:q4"]


def test_turn_finishing_during_a_read_restarts_it_from_the_prefix(make):
    w = make(piece_s=0.05, freshness="C")
    show(w, 1, "A")
    w.submit_frame(2, time.time(), "B")
    time.sleep(0.25)                                 # encode done (0.2 s): B's screen turn is being added
    r = ask(w, "q1")                                 # C: answer now, from screen A
    assert r["frame_id"] == 1
    deadline = time.time() + 5
    while (w.snapshot.frame_id != 2 or w.read is not None or w.turn_q) and time.time() < deadline:
        time.sleep(0.01)
    assert w.stats.get("read_restarts", 0) >= 1
    ask(w, "q2")
    assert w.adapter.seen["q2"] == ["BOS", "EX:q1", "SCREEN:B", "Q:q2"]


def test_history_cap_drops_the_oldest_exchanges(make):
    w = make(tok_per_ex=10, history_max_tokens=25)
    show(w, 1, "A")
    for q in ("q1", "q2", "q3"):
        ask(w, q)
    show(w, 2, "B")
    ask(w, "q4")
    assert w.adapter.seen["q4"] == ["BOS", "EX:q2", "EX:q3", "SCREEN:B", "Q:q4"]


def test_long_tail_of_turns_is_reanchored_on_the_same_screen(make):
    w = make(tok_per_ex=10, since_max_tokens=15)
    show(w, 1, "A")
    ask(w, "q1")
    ask(w, "q2")                                     # tail now 20 tokens > 15: re-anchor in quiet time
    deadline = time.time() + 5
    while (w.snapshot.tail != "screen" or w.ctx_since or w.read is not None) and time.time() < deadline:
        time.sleep(0.01)
    ask(w, "q3")
    assert w.adapter.seen["q3"] == ["BOS", "EX:q1", "EX:q2", "SCREEN:A", "Q:q3"]


def test_turn_finishing_during_encoding_is_simply_included(make):
    w = make(piece_s=0.05, freshness="C")
    show(w, 1, "A")
    w.submit_frame(2, time.time(), "B")
    time.sleep(0.05)                                 # still encoding: history not taken yet
    ask(w, "q1")
    deadline = time.time() + 5
    while (w.snapshot.frame_id != 2 or w.read is not None or w.turn_q) and time.time() < deadline:
        time.sleep(0.01)
    assert w.stats.get("read_restarts", 0) == 0
    ask(w, "q2")
    assert w.adapter.seen["q2"] == ["BOS", "EX:q1", "SCREEN:B", "Q:q2"]


def test_answers_never_wait_for_more_than_their_own_conversation(make):
    w = make(piece_s=0.03, freshness="C")
    show(w, 1, "A")
    w.submit_frame(2, time.time(), "B")               # a long read is under way
    t0 = time.perf_counter()
    r = ask(w, "q1")
    assert r["wait_behind_read_s"] < 0.1              # C and no earlier turns: answer at once
