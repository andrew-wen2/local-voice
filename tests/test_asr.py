"""Tests for asr.py. Fast: server-down error path, WAV encoding. Slow: real WhisperKit on the 10 recorded questions."""
import io
import json
import re
import wave
from pathlib import Path

import numpy as np
import pytest

from asr import AsrError, WhisperServer, wav_bytes

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "harness/questions"


def test_wav_bytes_roundtrip():
    x = np.sin(np.linspace(0, 100, 16000)).astype(np.float32) * 0.5
    with wave.open(io.BytesIO(wav_bytes(x))) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 16000)


def test_server_down_gives_clear_error():
    s = WhisperServer(port=50999)          # nothing listening here
    assert not s.healthy()
    with pytest.raises(AsrError, match="speech recognition failed"):
        s.transcribe(np.zeros(1600, np.float32), timeout_s=2)


def _norm(t):
    t = re.sub(r"[^a-z0-9 ]", "", t.lower())
    for a, b in (("thirty one", "31"), ("three and a half", "35"), ("35", "35")):
        t = t.replace(a, b)
    return " ".join(t.split())


@pytest.mark.slow
def test_transcribes_recorded_questions():
    s = WhisperServer(log_path=ROOT / "logs/whisperkit-test.log")
    (ROOT / "logs").mkdir(exist_ok=True)
    s.start()
    try:
        qs = json.loads((QDIR / "index.json").read_text())
        bad = []
        for q in qs:
            r = s.transcribe((QDIR / q["file"]).read_bytes())
            if _norm(r["text"]) != _norm(q["text"]):
                bad.append((q["text"], r["text"]))
        assert not bad, bad
    finally:
        s.stop()


def test_short_clip_is_padded_to_min_length():
    import numpy as np
    from asr import MIN_CLIP_S, pad_short
    a = np.ones(int(0.92 * 16000), np.float32)
    out = pad_short(a)
    assert len(out) == int(MIN_CLIP_S * 16000) and out[:len(a)].all() and not out[len(a):].any()
    long = np.ones(32000, np.float32)
    assert len(pad_short(long)) == 32000
