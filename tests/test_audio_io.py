"""Tests for audio_io.py (no audio device needed)."""
import numpy as np

from audio_io import PlaybackBuffer, near_silent, resample_linear, rms_dbfs


def test_buffer_pull_spans_chunks_and_pads_with_silence():
    b = PlaybackBuffer()
    b.push(np.ones(3)); b.push(np.full(4, 2.0))
    out = b.pull(5)
    assert out.tolist() == [1, 1, 1, 2, 2]
    out = b.pull(5)
    assert out.tolist() == [2, 2, 0, 0, 0] and b.empty()


def test_clear_drops_everything_queued():
    b = PlaybackBuffer()
    b.push(np.ones(1000)); b.push(np.ones(500))
    assert b.clear() == 1500
    assert b.empty() and b.pull(10).sum() == 0


def test_near_silent_flags_silence_not_speech():
    assert near_silent(np.zeros(16000, np.float32))
    assert near_silent(np.random.default_rng(0).normal(0, 1e-4, 16000).astype(np.float32))
    assert not near_silent(0.1 * np.sin(np.linspace(0, 2000, 16000)).astype(np.float32))
    assert near_silent(np.zeros(0, np.float32))          # empty recording (pressed Enter twice)


def test_resample_length_and_identity():
    x = np.sin(np.linspace(0, 10, 24000)).astype(np.float32)
    assert len(resample_linear(x, 24000, 48000)) == 48000
    assert resample_linear(x, 24000, 24000) is not None and rms_dbfs(x) > -10
