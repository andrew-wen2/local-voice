"""Memory doesn't grow during use: what scales with the hour lives on disk, without the model.

Stored pieces don't pin the cache they were cut from, round-trip through disk exactly, and go away with their page;
kept screens' keys are memory-mapped and their files go away when they expire.
"""
import gc

import mlx.core as mx
import numpy as np

from kv_reuse import take
from recall_backend import RecallBackend
from screen_store import ScreenStore
from test_kv_reuse import Lay, feed, make, piece_of
from test_screens import screen


def active_mb():
    gc.collect()
    mx.synchronize()
    return mx.get_active_memory() / 1e6


def test_a_piece_does_not_keep_its_cache_alive():
    lay = Lay()
    m0 = active_mb()
    big = make()
    for c in big:
        c.step = 16384                                  # a big preallocated buffer, as a long cache has
    feed(lay, big, list(range(60)))
    mx.eval([x for c in big for x in c.state])
    cache_mb = active_mb() - m0
    p = take(lay, big, 50, window=True)
    del big, c
    held = active_mb() - m0
    assert p.n == 10 and cache_mb > 1.0
    assert held < cache_mb / 20, f"a 10-token piece holds {held:.2f} MB of its {cache_mb:.2f} MB cache"   # unfixed: all


def backend(tmp_path):
    be = object.__new__(RecallBackend)                  # only the disk part
    be.dir, be.stats, be.pieces = tmp_path, {"piece_bytes_on_disk": 0}, {}
    return be


def test_pieces_round_trip_through_disk_exactly(tmp_path):
    lay, be = Lay(), backend(tmp_path)
    for window in (True, False):
        p = piece_of(lay, [1, 2, 3], list(range(10, 30)), window)
        path = be._save_piece(f"page{window}", p)
        q = be._load_piece(path, window=window)
        assert (q.n, q.pos) == (p.n, p.pos)
        for i in p.full:
            assert all(mx.array_equal(a, b).item() for a, b in zip(p.full[i], q.full[i]))
        assert (q.window is None) == (not window)
        for i in (p.window or {}):
            assert all(mx.array_equal(a, b).item() for a, b in zip(p.window[i], q.window[i]))
        assert be._load_piece(path, window=False).window is None   # assembly loads only what it needs
    assert be.stats["piece_bytes_on_disk"] == sum(f.stat().st_size for f in tmp_path.iterdir())


def test_a_forgotten_page_takes_its_files_along(tmp_path):
    lay, be = Lay(), backend(tmp_path)
    p = piece_of(lay, [1, 2, 3], list(range(10, 30)), True)
    be.pieces["view:7"] = {"version": 1, "ids": np.arange(20, dtype=np.int32), "parents": [],
                           "path": be._save_piece("view:7", p),
                           "cuts": {5: be._save_piece("view:7#5", piece_of(lay, [1, 2, 3], list(range(10, 15)), False))}}
    be._forget("view:7")
    assert not be.pieces and not list(tmp_path.iterdir()) and be.stats["piece_bytes_on_disk"] == 0


def test_keys_are_memory_mapped_and_expire_with_the_screen(tmp_path):
    st = ScreenStore(tmp_path / "screens", horizon_s=100)
    rng = np.random.default_rng(0)
    keys = {L: (rng.integers(-127, 127, (2, 50, 8)).astype(np.int8), rng.random((2, 50, 1)).astype(np.float16))
            for L in (11, 17)}
    s = screen(1, 0.0)
    s.keys = {L: (k.copy(), sc.copy()) for L, (k, sc) in keys.items()}
    st.add(s)
    for L in keys:
        k, sc = st.screens[1].keys[L]
        assert isinstance(k, np.memmap) and isinstance(sc, np.memmap)
        assert np.array_equal(k, keys[L][0]) and np.array_equal(sc, keys[L][1])
    files = sorted(p.name for p in (tmp_path / "screens").iterdir())
    assert files == ["1.keys.npy", "1.scales.npy"]
    st.expire(500.0)
    assert not st.screens and not list((tmp_path / "screens").iterdir())
