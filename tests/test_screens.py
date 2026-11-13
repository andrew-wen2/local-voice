"""Kept screens without the model (ARCHITECTURE.md §5.12): the store (screen_store.py), combining text and key
rankings, the Neural Engine encoder thread, and how the scheduler and worker treat screen slots."""
import os
import stat
import threading
import time

import numpy as np
from PIL import Image

from recall import Store
from recall_backend import KEY, OPENER, SCREEN
from recall_worker import RecallWorker, ScreenEncoder
from scheduler import FakeClock, Scheduler, SimBackend, simulate
from screen_store import Screen, ScreenStore, fuse, quantize_int8


def screen(fid, t, title="Page", lines=()):
    return Screen(fid, t, f"key{fid}", title, list(lines), {}, {}, None, size=(100, 100), T=10)


def test_store_files_are_private_and_go_away(tmp_path):
    st = ScreenStore(tmp_path / "s")
    assert stat.S_IMODE(os.stat(st.dir).st_mode) == 0o700
    p = st.save_features(7, np.ones((1, 4, 3), np.float32))
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    s = screen(7, 10.0)
    s.path = p
    st.add(s)
    assert st.ready(7) and st.load_features(7).dtype == np.float16
    assert np.allclose(st.load_features(7), 1.0, atol=1 / 127)        # int8 on disk, close to what was saved
    st.add(screen(8, 4000.0))
    assert st.expire(3000.0) == 0                    # 7 is 2990 s old: still in the hour
    assert st.expire(10.0 + 3601) == 1               # now it's past the hour
    assert not st.ready(7) and not p.exists() and st.ready(8)
    st.close()
    assert not st.dir.exists()


def test_pending_is_taken_once():
    st = ScreenStore()
    st.put_pending(3, {"t": 1}, tokens=np.zeros(2))
    assert st.take_pending(3)["meta"] == {"t": 1} and st.take_pending(3) is None
    st.close()


def test_text_ranking_and_fuse():
    st = ScreenStore()
    st.add(screen(1, 1.0, "Lamp shop", ["brass lamp"]))
    st.add(screen(2, 2.0, "Trains", ["the 8:15"]))
    st.add(screen(3, 3.0, "Notes", ["todo"]))
    assert st.text_ranking("brass lamp")[0] == 1
    # better rank of the two routes; ties to the keys' order
    assert fuse([1, 2, 3], [3, 1, 2], 2) == [3, 1]
    assert fuse([1, 2, 3], [2, 3, 1], 3) == [2, 1, 3]
    st.close()


def test_int8_keys_roundtrip():
    x = np.random.default_rng(0).normal(size=(2, 5, 8)).astype(np.float32)
    q, s = quantize_int8(x)
    assert q.dtype == np.int8 and s.shape == (2, 5, 1)
    assert np.abs(q * s.astype(np.float32) - x).max() <= np.abs(x).max() / 127 * 1.01


class FakeAdapter:
    def screen_turn_prompt(self):
        return "<|image|>Question:"

    def _inputs(self, image, prompt):
        return {"pixel_values": np.zeros((1, 3, 8, 8), np.float32)}


class FakeAne:
    def has(self, h, w):
        return True

    def encode(self, pv):
        return np.ones((1, 4, 2), np.float32)


def test_encoder_uses_the_neural_engine_or_hands_the_image_over():
    for ane, want in ((FakeAne(), "tokens"), (None, "image")):
        st, ready, quiet = ScreenStore(), [], threading.Event()
        e = ScreenEncoder(FakeAdapter(), st, quiet=quiet.is_set, on_ready=ready.append, ane=ane)
        e.start()
        e.submit(5, Image.new("RGB", (64, 32)), {"t": 1.0, "key": "k", "title": "t", "lines": []})
        time.sleep(0.2)
        assert ready == []                           # a turn is active
        quiet.set()
        e.poke()
        for _ in range(50):
            if ready:
                break
            time.sleep(0.02)
        e.stop()
        p = st.take_pending(5)
        assert ready == [5] and p[want] is not None and p["meta"]["size"] == (64, 32)
        st.close()


T = {"prep": .001, "patch": .001, "vision": .001, "pool": .001, "embed": .001, "text": .001,
     "page_layer": .001, "q_layers": .001, "assemble": .001, "decode": .001}


class ScreenSim(SimBackend):
    """Screens and the opener are computed exactly, never built; keying work is built."""

    def buildable(self, slot):
        return slot[0] not in (SCREEN, OPENER)


def test_screens_are_never_built_and_are_computed_at_release_if_not_reached():
    clock = FakeClock()
    be = ScreenSim(clock, T, built=["a"])
    s = Scheduler(be, lambda text: ((SCREEN, 1), (OPENER,), "a"), clock=clock)
    s.frame(0)
    simulate(s, [])
    clock.t = 10
    simulate(s, [(10.0, "key_down"), (10.3, "key_up"), (10.33, "final", "what was it")])
    p = s.turns[-1]["plan"]
    assert p["missing"] == [(SCREEN, 1), (OPENER,)]
    assert not any(c[0] == "build" and c[1][0] in (SCREEN, OPENER) for c in be.calls)


def test_keying_runs_in_quiet_time_only():
    clock = FakeClock()
    be = ScreenSim(clock, T, built=["a"])
    s = Scheduler(be, lambda text: ("a",), clock=clock)
    s.frame(0)
    simulate(s, [])
    clock.t = 10
    simulate(s, [(10.0, "key_down"), (10.01, "add_build", (KEY, 9)), (10.3, "key_up"), (10.33, "final", "q")])
    from scheduler import Cls, State
    assert (KEY, 9) in be.built
    assert all(st is State.QUIET for _, _, st, c, _, _ in s.log if c is Cls.BACKGROUND)


def test_worker_puts_screens_first_then_the_opener_then_pages():
    st = Store()
    st.add(0, "A", "Lamp shop", "brass lamp")
    st.add(10, "B", "Notes", "todo")
    be = SimBackend(FakeClock(), T)
    calls = []
    be.find_screens = lambda text: calls.append(text) or [4, 2]
    w = RecallWorker(None, None, st, backend_factory=lambda: be)
    w.start()
    assert w.ready.wait(5)
    slots = w.run_on_worker(lambda: w._retrieve("brass lamp"))
    again = w.run_on_worker(lambda: w._retrieve("brass lamp"))
    w.stop()
    w.join(5)
    assert slots[:3] == ((SCREEN, 4), (SCREEN, 2), (OPENER,)) and slots[3][0] == "A"
    assert again == slots and calls == ["brass lamp"]          # cached for the turn


def test_the_visit_still_on_screen_is_left_out():
    st = ScreenStore()
    st.add(Screen(1, 10.0, "yt", "Video", [], {}, {}, None))      # an earlier visit of the same page
    st.add(Screen(2, 50.0, "docs", "Docs", [], {}, {}, None))
    st.add(Screen(3, 90.0, "yt", "Video", [], {}, {}, None))      # the visit on screen now started at t=80
    st.add(Screen(4, 95.0, "yt", "Video", [], {}, {}, None))
    assert st.ids(exclude_visit=("yt", 80.0)) == [1, 2] and st.ids() == [1, 2, 3, 4]
    st.close()


def test_eval_screen_cache_round_trip(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "eval/recall"), str(root / "eval/kernels"), str(root / "eval/real")]
    spec = importlib.util.spec_from_file_location("visual_e2e", root / "eval/real/visual_e2e.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    st = ScreenStore(tmp_path / "live")
    p = st.save_features(3, np.ones((1, 4, 2), np.float32))
    q, s = quantize_int8(np.ones((2, 4, 8), np.float32))
    st.add(Screen(3, 5.0, "k", "T", ["a"], {11: (q, s), 17: (q, s)}, {11: 0.5, 17: -0.2}, p, size=(10, 20), T=99))
    out = tmp_path / "cache" / "abc"
    E.save_cache(out, st)
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o700
    st2 = ScreenStore(tmp_path / "again")
    assert E.load_cache(out, st2) and st2.ready(3)
    s2 = st2.screens[3]
    assert s2.null == {11: 0.5, 17: -0.2} and s2.T == 99 and s2.size == (10, 20)
    assert np.array_equal(s2.keys[17][0], q) and np.allclose(st2.load_features(3), 1.0, atol=1 / 127)
    st.close()
    st2.close()
    assert out.exists()                                # the cache outlives the run's temporary store


def test_unit_text_joins_its_screen_even_before_it_is_ready():
    st = ScreenStore()
    st.add_lines(2, ["brass lamp"])                       # the screen is still being encoded
    st.add(screen(1, 1.0, "Notes", ["todo"]))
    st.add(screen(2, 2.0, "Shop", ["lamps"]))
    assert st.screens[2].lines == ["lamps", "brass lamp"]
    st.add_lines(1, ["the 8:15 train", "todo"])            # after it's ready; duplicates dropped
    assert st.screens[1].lines == ["todo", "the 8:15 train"]
    sc = st.text_scores("brass lamp", [1, 2])
    assert sc[1] > sc[0] and st.text_scores("8:15 train", [2, 1])[1] > 0
    st.close()


def test_a_kept_screens_view_text_is_a_memory_slot():
    st = Store()
    st.add(0, "A", "Lamp shop", "brass lamp")
    st.add(10, "B", "Notes", "todo")
    s = screen(7, 5.0, "Lamp shop", ["brass lamp", "price 40", "in stock"])
    s.key = "A"
    vk = st.set_view(7, s)
    assert vk == "view:7" and st.page(vk).url == "A" and st.version[vk] == 1
    assert "brass lamp" in st.page_text(vk) and "Lamp shop" in st.page_text(vk)
    assert st.set_view(7, s) == vk and st.version[vk] == 1                 # same text: same version
    s.lines = s.lines + ["blue shade"]
    st.set_view(7, s)
    assert st.version[vk] == 2                                             # text grew: new version
    key, v, n = st.view_slot(vk, 10**6, lambda t: len(t.split()))
    assert (key, v, n) == (vk, 2, 4)
    assert st.view_slot(vk, 1, lambda t: len(t.split()))[2] == 1           # always at least one line
    st.drop_view(7)
    assert st.page(vk) is None


def test_worker_gives_the_top_screens_view_instead_of_pages():
    st = Store()
    st.add(0, "A", "Lamp shop", "brass lamp")
    st.add(10, "B", "Notes", "todo")
    ss = ScreenStore()
    s = screen(4, 1.0, "Lamp shop", ["brass lamp"])
    s.key = "A"
    ss.add(s)
    be = SimBackend(FakeClock(), T)
    be.find_screens = lambda text: [4]
    be.ss = ss
    w = RecallWorker(None, None, st, backend_factory=lambda: be)
    w.start()
    assert w.ready.wait(5)
    slots = w.run_on_worker(lambda: w._retrieve("brass lamp"))
    w.stop()
    w.join(5)
    assert slots[:2] == ((SCREEN, 4), (OPENER,)) and len(slots) == 3 and slots[2][0] == "view:4"
    ss.close()
