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
# Voice activity while the key is held (recall mode, ARCHITECTURE.md §5.9). Provisional, and only used for
# scheduling: a block is voiced above VOICE_DBFS; the mic counts as silent after SILENCE_S without a voiced block.
VOICE_DBFS = -45.0
SILENCE_S = 0.3


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
    """Records while open. `begin()` on key press, `end()` on release returns the samples.
    While recording it also tracks voice activity per 20 ms block (`audio()`, `voiced_until()`, `silent_for()`)."""

    def __init__(self, sample_rate: int = 16000, voice_dbfs: float = VOICE_DBFS):
        self.sr = sample_rate
        self.lock = threading.Lock()
        self.parts: list[np.ndarray] = []
        self.recording = False
        self.stream = None
        self.voice_dbfs = voice_dbfs
        self.n = 0                       # samples recorded this turn
        self.last_voice = -1             # sample index where the last voiced block ended (-1: none yet)

    def start(self):
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                     blocksize=320, callback=self._cb)
        self.stream.start()
        return self

    def _cb(self, indata, frames, time_info, status):
        if self.recording:
            self.feed(indata[:, 0].copy())

    def feed(self, x: np.ndarray):
        """One block of samples (the stream callback, or a replay)."""
        with self.lock:
            self.parts.append(x)
            self.n += len(x)
            if rms_dbfs(x) > self.voice_dbfs:
                self.last_voice = self.n

    def begin(self):
        with self.lock:
            self.parts = []
            self.n, self.last_voice = 0, -1
            self.recording = True

    def audio(self) -> np.ndarray:
        """The samples so far, without ending the turn."""
        with self.lock:
            return np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)

    def voiced_until(self) -> int:
        """Sample index where the last voiced block ended, or -1."""
        return self.last_voice

    def silent_for(self) -> Optional[float]:
        """Seconds since the last voiced block, or None before any voice this turn."""
        with self.lock:
            return None if self.last_voice < 0 else (self.n - self.last_voice) / self.sr

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
