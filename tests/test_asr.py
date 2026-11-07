"""Tests for asr.py. Fast: server-down error paths, WAV encoding, Parakeet on the recorded questions
(skipped if its binary or models are missing). Slow: real WhisperKit on the recorded questions."""
import io
import json
import re
import wave
from pathlib import Path

import numpy as np
import pytest

from asr import PARAKEET_BIN, PARAKEET_DIR, AsrError, ParakeetServer, WhisperServer, make_asr, wav_bytes, wav_to_samples

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


def test_wav_to_samples_roundtrip():
    x = (np.sin(np.linspace(0, 100, 16000)) * 0.5).astype(np.float32)
    assert np.abs(wav_to_samples(wav_bytes(x)) - x).max() < 1e-4


def test_make_asr_kinds():
    assert isinstance(make_asr("parakeet"), ParakeetServer)
    assert isinstance(make_asr("whisper"), WhisperServer)
    with pytest.raises(ValueError):
        make_asr("nope")


def test_parakeet_not_running_gives_clear_error():
    s = ParakeetServer()
    assert not s.healthy()
    with pytest.raises(AsrError, match="speech recognition failed"):
        s.transcribe(np.zeros(1600, np.float32))


def test_parakeet_missing_binary_says_how_to_build(tmp_path):
    with pytest.raises(AsrError, match="swift build"):
        ParakeetServer(binary=tmp_path / "nope").start()


needs_parakeet = pytest.mark.skipif(not (PARAKEET_BIN.exists() and (PARAKEET_DIR / "parakeet_vocab.json").exists()),
                                    reason="Parakeet binary or models not present")

# Script text vs what Parakeet writes: formatting differences only (identifiers spoken as words), plus
# one known word error (rd01 "as" -> "is"), measured 2026-11-06 (eval/asr_compare.py).
PARAKEET_KNOWN = {"rd01", "rd02", "rd05", "rd06", "rd12", "rd13"}


@needs_parakeet
def test_parakeet_transcribes_recorded_questions():
    s = ParakeetServer()
    s.start(sandbox_profile=ROOT / "eval/offline.sb")
    try:
        qs = json.loads((QDIR / "index.json").read_text())
        bad, slow = [], []
        for q in qs:
            x = wav_to_samples((QDIR / q["file"]).read_bytes())[: int(q["speech_end_s"] * 16000)]
            r = s.transcribe(x)
            if _norm(r["text"]) != _norm(q["text"]) and q["id"] not in PARAKEET_KNOWN:
                bad.append((q["id"], q["text"], r["text"]))
            if r["seconds"] > 0.2:
                slow.append((q["id"], r["seconds"]))
        assert not bad, bad
        assert not slow, slow
    finally:
        s.stop()
    assert not s.healthy()
