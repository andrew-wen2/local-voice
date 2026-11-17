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
        chunks = list(t.synth("The observatory was 36 meters high."))   # once crashed the helper
        assert len(chunks) == 1 and len(chunks[0]) > 0
        assert len(t.failures) == 1 and "retry_error" not in t.failures[0]
    finally:
        t.stop()


def _derive(text):
    import json
    import subprocess
    t = AneTts()
    out = subprocess.run([str(t.binary), "--models", str(t.model_dir), "--derive", text],
                         capture_output=True, text=True, timeout=60).stdout
    return json.loads(next(line for line in out.splitlines() if line.startswith("@@"))[2:])["entries"]


def test_inflected_words_take_their_stems_pronunciation():
    """Misaki's -s / -ed / -ing rules (kokoro-serve Inflections.swift), for words the lexicon lacks but whose stem it has.
    Expected values follow misaki/en.py 0.7.4 (_s, _ed, _ing); the port matched its functions on 38,202 such words."""
    e = _derive("The Ravens waited, then the hopping frogs rested.")
    assert e["Ravens"] == "ɹˈAvᵊnz"                      # raven + z (FluidAudio's G2P guessed ɹˈævənz)
    assert e["waited"] == "wˈAɾᵻd"                        # wait + flap ᵻd (US_TAUS before t)
    assert "The" not in e and "then" not in e             # lexicon words are left to FluidAudio


def test_inflections_skip_all_caps_compounds_and_apostrophes():
    assert _derive("The NASAS, well-known don'ts") == {}  # FluidAudio spells all caps; compounds, 's are its own
    assert _derive("AIs") == {"AIs": "ˈAˌIz"}            # mixed case: AI + z
