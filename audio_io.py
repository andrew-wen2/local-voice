"""audio_io.py — microphone and speaker.

Mic: records 16 kHz mono while a push-to-talk turn is open; flags near-silent turns
(possible missing Microphone permission or muted mic; review D5).
Speaker: a playback queue fed by the GPU worker; `clear()` empties it on interrupt (D6).
First audio is timestamped when the first audible sample is handed to the queue (same
definition as the harness); the output device's own latency is reported separately.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Optional

import numpy as np

SILENT_DBFS = -50.0


def rms_dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, np.float32)
    if x.size == 0:
        return -120.0
    return float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9))


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, np.float32)
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


class PlaybackBuffer:
    """Thread-safe FIFO of float32 samples at one sample rate (no audio device needed)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.chunks: collections.deque[np.ndarray] = collections.deque()
        self.queued = 0

    def push(self, x: np.ndarray):
        x = np.asarray(x, np.float32).reshape(-1)
        with self.lock:
            self.chunks.append(x)
            self.queued += len(x)

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        filled = 0
        with self.lock:
            while filled < n and self.chunks:
                c = self.chunks[0]
                k = min(n - filled, len(c))
                out[filled:filled + k] = c[:k]
                filled += k
                if k == len(c):
                    self.chunks.popleft()
                else:
                    self.chunks[0] = c[k:]
            self.queued -= filled
        return out

    def clear(self) -> int:
        with self.lock:
            n, self.queued = self.queued, 0
            self.chunks.clear()
        return n

    def empty(self) -> bool:
        with self.lock:
            return self.queued == 0


class Speaker:
    """Plays queued audio through the default output device (sounddevice)."""

    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate
        self.buf = PlaybackBuffer()
        self.stream = None

    def start(self):
        import sounddevice as sd
        self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32",
                                      latency="low", callback=self._cb)
        self.stream.start()
        return self

    def _cb(self, outdata, frames, time_info, status):
        outdata[:, 0] = self.buf.pull(frames)

    def play(self, samples: np.ndarray, sr: int):
        self.buf.push(resample_linear(samples, sr, self.sr))

    def clear(self) -> int:
        return self.buf.clear()

    def wait_done(self, timeout: float = 30) -> bool:
        t = time.time() + timeout
        while not self.buf.empty() and time.time() < t:
            time.sleep(0.02)
        return self.buf.empty()

    def device_latency_ms(self) -> Optional[float]:
        return None if self.stream is None else round(self.stream.latency * 1000, 1)

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class Mic:
    """Records while open. `begin()` on key press, `end()` on release returns the samples."""

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.lock = threading.Lock()
        self.parts: list[np.ndarray] = []
        self.recording = False
        self.stream = None

    def start(self):
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                     blocksize=320, callback=self._cb)
        self.stream.start()
        return self

    def _cb(self, indata, frames, time_info, status):
        if self.recording:
            with self.lock:
                self.parts.append(indata[:, 0].copy())

    def begin(self):
        with self.lock:
            self.parts = []
            self.recording = True

    def end(self) -> np.ndarray:
        with self.lock:
            self.recording = False
            x = np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)
            self.parts = []
        return x

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


def near_silent(x: np.ndarray, threshold_dbfs: float = SILENT_DBFS) -> bool:
    """True if a whole turn is quieter than the threshold (no speech captured)."""
    return rms_dbfs(x) < threshold_dbfs
