"""Split placement (P4, ARCHITECTURE.md §6.1): quiet-time encodes on the Neural Engine, everything urgent
on the GPU. Fake model and fake Neural Engine encoder with timed chunks."""
import time

import numpy as np
import pytest

import engine
from test_context import FakeCtxAdapter, FakeTts, ask

BUCKET = (48, 48)


class FakeSplitAdapter(FakeCtxAdapter):
    """Images are strings; "wide:..." images have no Neural Engine bucket."""

    def __init__(self, piece_s=0.01):
        super().__init__(piece_s=piece_s)
        self.gpu_encodes, self.ane_feats, self.resumed = [], [], []

    def prepare_screen(self, image, note=None):
        h, w = (48, 96) if str(image).startswith("wide") else BUCKET
        return (image, note), np.zeros((1, 3, h, w), np.float32)

    def encode_steps(self, image, note=None, inp=None, resume=None):
        self.gpu_encodes.append(image)
        self.resumed.append(None if resume is None else resume[0])
        assert resume is None or resume[1] == f"hidden-after-{resume[0]}"
        yield from self._pieces(4 - (resume[0] if resume else 0) // 4)
        return (image, "feats-gpu")

    def feats_from_tokens(self, inp, tokens):
        self.ane_feats.append(inp[0])
        return (inp[0], "feats-ane")


class FakeAne:
    def __init__(self, chunks=4, chunk_s=0.05):
        self.chunks, self.chunk_s = chunks, chunk_s
        self.starts: list[float] = []

    def has(self, h, w):
        return (h, w) == BUCKET

    def encode_steps(self, pv):
        for i in range(self.chunks):
            self.starts.append(time.perf_counter())
            time.sleep(self.chunk_s)
            if i == self.chunks - 1:
                return "tokens"
            yield i + 1, self.chunks, i + 1, f"hidden-after-{i + 1}"


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(engine, "clone_cache", lambda c: list(c))
    workers = []

    def _make(history=False, chunks=4, chunk_s=0.05, **kw):
        ad, ane = FakeSplitAdapter(), FakeAne(chunks, chunk_s)
        w = engine.GpuWorker(ad, FakeTts(), policy="p2", placement="split", ane_encoder=ane, history=history, **kw)
        w.start()
        assert w.ready.wait(5) and w.ane_runner is not None
        workers.append(w)
        return w, ad, ane
    yield _make
    for w in workers:
        w.stop()


def wait_for(cond, timeout=5):
    deadline = time.time() + timeout
    while not cond() and time.time() < deadline:
        time.sleep(0.005)
    assert cond()


def settled(w, fid):
    return lambda: w.snapshot is not None and w.snapshot.frame_id == fid and w.read is None


@pytest.mark.parametrize("history", [False, True])
def test_quiet_time_read_is_encoded_on_the_neural_engine(make, history):
    w, ad, ane = make(history=history)
    w.submit_frame(1, time.time(), "A")
    wait_for(settled(w, 1))
    assert ad.ane_feats == ["A"] and ad.gpu_encodes == []
    assert len(ane.starts) == 4 and w.read_log[-1]["encoder"] == "ane"
    ask(w, "q1")
    assert ad.seen["q1"] == ["BOS", "SCREEN:A", "Q:q1"]      # same context as a GPU encode


def test_answer_runs_while_the_neural_engine_encodes(make):
    w, ad, ane = make(chunks=6, chunk_s=0.1, freshness="C")
    w.submit_frame(1, time.time(), "A")
    wait_for(settled(w, 1))
    w.submit_frame(2, time.time(), "B")
    wait_for(lambda: len(ane.starts) >= 8)                   # frame 2 is on the Neural Engine
    t0 = time.perf_counter()
    r = ask(w, "q1")                                         # C: answer now from frame 1
    assert r["frame_id"] == 1 and r["wait_behind_read_s"] < 0.05
    assert time.perf_counter() - t0 < 0.3                    # never waited for the 0.6 s encode
    wait_for(settled(w, 2))
    assert w.read_log[-1]["encoder"] == "ane"


def test_a_turn_moves_the_encode_to_the_gpu_and_no_chunk_starts_during_it(make):
    w, ad, ane = make(chunks=10, chunk_s=0.05)
    w.submit_frame(1, time.time(), "A")
    wait_for(lambda: len(ane.starts) >= 2)
    w.set_turn_active(True)
    t_turn = time.perf_counter()
    wait_for(settled(w, 1))
    assert ad.gpu_encodes == ["A"] and w.stats["ane_to_gpu"] == 1
    k = ad.resumed[0]                                        # continued after the layers already done
    assert k is not None and k >= 1 and w.read_log[-1]["encoder"] == f"ane{k}+gpu"
    w.submit_frame(2, time.time(), "B")                      # during the turn: read on the GPU
    wait_for(settled(w, 2))
    assert ad.gpu_encodes == ["A", "B"] and ad.resumed[1] is None
    time.sleep(0.1)
    assert all(s < t_turn for s in ane.starts)               # at most the chunk in flight ran on
    w.set_turn_active(False)


def test_freshness_b_answer_takes_the_frame_from_the_neural_engine_to_the_gpu(make):
    w, ad, ane = make(chunks=20, chunk_s=0.05, freshness="B")
    w.submit_frame(1, time.time(), "A")
    wait_for(settled(w, 1))
    w.submit_frame(2, time.time(), "B")
    wait_for(lambda: len(ane.starts) >= 22)
    r = ask(w, "q1")                                         # turn_active stays False: the pin alone moves it
    assert r["frame_id"] == 2 and r["fresh_decision"] == "read_first"
    assert ad.gpu_encodes == ["B"] and w.stats["ane_to_gpu"] == 1


def test_newer_frame_drops_the_neural_engine_encode(make):
    w, ad, ane = make(chunks=6, chunk_s=0.05)
    w.submit_frame(1, time.time(), "A")
    wait_for(lambda: len(ane.starts) >= 2)
    w.submit_frame(2, time.time(), "B")
    wait_for(settled(w, 2))
    assert ad.ane_feats == ["B"] and w.stats["reads_dropped"] == 1
    assert w.ane_runner.stats["wasted_chunks"] <= 1
    assert len(ane.starts) <= 2 + 1 + 6


def test_image_without_a_bucket_is_encoded_on_the_gpu(make):
    w, ad, ane = make()
    w.submit_frame(1, time.time(), "wide:A")
    wait_for(settled(w, 1))
    assert ad.gpu_encodes == ["wide:A"] and ane.starts == []


def test_split_placement_needs_pausable_reads():
    with pytest.raises(AssertionError):
        engine.GpuWorker(FakeSplitAdapter(), FakeTts(), policy="p0", placement="split")
