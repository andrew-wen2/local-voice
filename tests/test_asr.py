"""Tests for asr.py: server-down error paths and WAV encoding."""
import io
import wave

import numpy as np
import pytest

from asr import AsrError, ParakeetServer, make_asr, wav_bytes, wav_to_samples


def test_wav_bytes_roundtrip():
    x = np.sin(np.linspace(0, 100, 16000)).astype(np.float32) * 0.5
    with wave.open(io.BytesIO(wav_bytes(x))) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 16000)


def test_wav_to_samples_roundtrip():
    x = (np.sin(np.linspace(0, 100, 16000)) * 0.5).astype(np.float32)
    assert np.abs(wav_to_samples(wav_bytes(x)) - x).max() < 1e-4


def test_make_asr_is_parakeet():
    assert isinstance(make_asr(), ParakeetServer)


def test_parakeet_not_running_gives_clear_error():
    s = ParakeetServer()
    assert not s.healthy()
    with pytest.raises(AsrError, match="speech recognition failed"):
        s.transcribe(np.zeros(1600, np.float32))


def test_parakeet_missing_binary_says_how_to_build(tmp_path):
    with pytest.raises(AsrError, match="swift build"):
        ParakeetServer(binary=tmp_path / "nope").start()
