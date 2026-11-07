"""ane_vision.patchify must lay out patches exactly as mlx_vlm's Gemma 4 patch embedder does,
or the Core ML encoder would see a scrambled image."""
import mlx.core as mx
import pytest
import numpy as np

from ane_vision import patchify


def test_patchify_matches_mlx_patch_embedder():
    from mlx_vlm.models.gemma4.config import VisionConfig
    from mlx_vlm.models.gemma4.vision import VisionPatchEmbedder
    pe = VisionPatchEmbedder(VisionConfig(hidden_size=768))
    pe.input_proj.weight = mx.eye(768)                 # identity: _patchify returns 2 * (patches - 0.5)
    pv = np.random.default_rng(0).random((1, 3, 48, 96), dtype=np.float32)
    with mx.stream(mx.cpu):                            # exact fp32 matmul (the GPU rounds ~1e-3)
        ref = np.asarray(pe._patchify(mx.array(pv)))
    mine = 2 * (patchify(pv) - 0.5)
    assert mine.shape == (1, 3 * 6, 768)
    np.testing.assert_allclose(mine, ref, atol=1e-6)


def test_patchify_raster_order_x_fastest():
    pv = np.zeros((1, 3, 32, 48), np.float32)
    pv[:, :, 0:16, 16:32] = 1.0                        # patch at x=1, y=0
    p = patchify(pv)
    assert p[0, 1].min() == 1.0 and p[0, 0].max() == 0.0 and p[0, 3].max() == 0.0


@pytest.mark.slow
def test_child_process_matches_in_process_and_keeps_other_threads_running():
    """Needs the converted encoder (python ane_vision.py convert)."""
    import threading
    import time
    from ane_vision import AneProcess, AneVisionEncoder
    p = AneProcess().load()
    try:
        if not p.buckets:
            pytest.skip("no converted encoder")
        h, w = next(iter(p.buckets))
        pv = np.random.default_rng(0).random((1, 3, h, w), dtype=np.float32)
        spins, stop = [0], threading.Event()

        def spin():
            while not stop.is_set():
                spins[0] += 1
        t = threading.Thread(target=spin)
        t0 = time.perf_counter()
        t.start()
        tok = p.encode(pv)
        stop.set()
        t.join()
        rate = spins[0] / (time.perf_counter() - t0)
        assert rate > 10e6                                    # in-process Core ML held the GIL: ~2 M/s
        np.testing.assert_array_equal(tok, AneVisionEncoder().load([(h, w)]).encode(pv))
    finally:
        p.close()
