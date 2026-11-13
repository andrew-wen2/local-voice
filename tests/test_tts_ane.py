"""Kokoro on the Neural Engine (engine.AneTts, tools/parakeet kokoro-serve). Skipped if not set up."""
import sys

import numpy as np
import pytest

from engine import AneTts, make_tts

pytestmark = pytest.mark.skipif(not AneTts().available(), reason="kokoro-serve or models/kokoro-coreml missing")


def test_synth_returns_trimmed_audio_and_stops_cleanly():
    t = AneTts().load()
    try:
        assert t.startup.get("ready") and t.pid()
        chunks = list(t.synth("The rent was $3,488 per month."))
        assert len(chunks) == 1 and chunks[0].dtype == np.float32
        a = chunks[0]
        assert 1.0 < len(a) / t.sample_rate < 8.0 and 0.05 < np.abs(a).max() <= 1.0
        assert list(t.synth("   ")) == []                        # nothing to say
        assert len(list(t.synth("Line one\nline two."))) == 1      # newlines can't break the protocol
    finally:
        t.stop()
    assert t.pid() is None


def test_default_tts_never_imports_the_python_kokoro_stack():
    assert isinstance(make_tts(), AneTts)
    assert not any(m in sys.modules for m in ("mlx_audio", "misaki", "spacy"))


def test_a_dead_helper_is_restarted_and_the_text_still_spoken():
    t = AneTts().load()
    try:
        t.proc.kill()
        t.proc.wait()
        chunks = list(t.synth("The observatory was 36 meters high."))   # crashed the helper before §3.7's fix
        assert len(chunks) == 1 and len(chunks[0]) > 0
        assert len(t.failures) == 1 and "retry_error" not in t.failures[0]
    finally:
        t.stop()
