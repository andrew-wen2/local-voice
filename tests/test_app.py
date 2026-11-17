"""Tests for app.py: readiness gate (D8), permission exit + near-silent flag (D5), memory (D19)."""
import os
import types

import numpy as np
import pytest

import app


def test_turn_refused_before_ready():
    a = app.Assistant.__new__(app.Assistant)
    a.ready = False
    row = a.answer_turn(np.zeros(16000, np.float32), lambda *x: None, lambda n: None, 0.0, 0.0)
    assert row["refused"] is True


class FakeAsr:
    def __init__(self, text): self.text = text
    def transcribe(self, audio): return {"text": self.text, "seconds": 0.01}
    def pid(self): return None


def test_near_silent_turn_is_flagged():
    a = app.Assistant.__new__(app.Assistant)
    a.ready, a.asr = True, FakeAsr("")
    marks = []
    row = a.answer_turn(np.zeros(16000, np.float32), lambda *x: None, marks.append, 0.0, 0.0)
    assert row["near_silent"] is True
    assert "mic muted or no permission" in row["error"]
    assert "turn_complete" in marks


def test_near_silent_turn_skips_recognition():
    """Parakeet turns a padded near-silent tap into "Okay."; such a turn must not reach the model."""
    a = app.Assistant.__new__(app.Assistant)
    a.ready, a.asr = True, FakeAsr("Okay.")
    row = a.answer_turn(np.zeros(8000, np.float32), lambda *x: None, lambda n: None, 0.0, 0.0)
    assert row["transcript"] == "" and "empty transcript" in row["error"]


def test_live_exits_with_message_without_screen_permission(monkeypatch, capsys):
    monkeypatch.setattr(app.scr, "has_screen_permission", lambda: False)
    with pytest.raises(SystemExit) as e:
        app.live(types.SimpleNamespace(budget=560))
    assert e.value.code == 2
    assert "Screen Recording permission is off" in capsys.readouterr().out


def test_footprint_of_this_process():
    fp = app.footprint_bytes(os.getpid())
    assert fp is not None and fp > 10_000_000


def test_monitor_window_combines_processes():
    m = app.Monitor.__new__(app.Monitor)
    m.samples = [{"t": 100.0, "mem_total": 6e9, "mem_app": 5e9, "mem_asr": 1e9, "all_w": 10.0, "fan_rpm": 0},
                 {"t": 101.0, "mem_total": 7e9, "mem_app": 5.5e9, "mem_asr": 1.5e9, "all_w": 14.0, "fan_rpm": 0}]
    w = m.window(100.0, 101.0)
    assert w["mem_peak_gb"] == 7.0 and w["mem_asr_peak_gb"] == 1.5 and w["power_mean_w"] == 12.0
