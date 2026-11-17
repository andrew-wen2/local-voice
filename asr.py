"""asr.py — speech recognition in a separate process on the Neural Engine.

ParakeetServer: Parakeet TDT 0.6B v2 through FluidAudio (Core ML), `tools/parakeet`. Audio goes over the
child's stdin and transcripts come back on stdout: no network, no audio on disk. ~40 ms for a 3 s question
(Whisper large-v3 turbo, tried first, was ~470 ms because it always encodes a 30 s window; ARCHITECTURE.md §6.2).
"""

from __future__ import annotations

import io
import json
import queue
import struct
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
PARAKEET_BIN = ROOT / "tools/parakeet/.build/release/parakeet-serve"
PARAKEET_DIR = ROOT / "models/parakeet-tdt-0.6b-v2"


class AsrError(RuntimeError):
    pass


def wav_bytes(samples: np.ndarray, sr: int = 16000) -> bytes:
    x = np.clip(np.asarray(samples, np.float32).reshape(-1), -1, 1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def wav_to_samples(data: bytes) -> np.ndarray:
    """16 kHz mono 16-bit WAV bytes -> float32 samples."""
    with wave.open(io.BytesIO(bytes(data))) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise AsrError(f"expected 16 kHz mono 16-bit WAV, got {w.getframerate()} Hz x{w.getnchannels()}")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


class ParakeetServer:
    """Parakeet TDT 0.6B v2 (English) on the Neural Engine, as a long-running child process.

    Interface: start / stop / pid / healthy / transcribe. Build the binary with
    `swift build -c release --package-path tools/parakeet`; fetch the models once with
    `tools/parakeet/.build/release/parakeet-serve --models models/parakeet-tdt-0.6b-v2 --download`.
    Protocol (see tools/parakeet/Sources/parakeet-serve/main.swift): request = uint32 sample count +
    float32 samples, reply = one "@@"-prefixed JSON line; other output lines are library logging."""

    def __init__(self, binary: Path = PARAKEET_BIN, model_dir: Path = PARAKEET_DIR,
                 log_path: Optional[Path] = None, encoder: str = "ane"):
        self.binary, self.model_dir, self.log_path, self.encoder = Path(binary), Path(model_dir), log_path, encoder
        self.proc: Optional[subprocess.Popen] = None
        self._replies: "queue.Queue[dict]" = queue.Queue()
        self._lock = threading.Lock()
        self.startup: dict = {}

    def start(self, timeout_s: float = 120, sandbox_profile: Optional[Path] = None) -> float:
        if self.healthy():
            return 0.0
        if not self.binary.exists():
            raise AsrError(f"{self.binary} missing; build it: swift build -c release --package-path tools/parakeet")
        if not (self.model_dir / "parakeet_vocab.json").exists():
            raise AsrError(f"Parakeet models missing in {self.model_dir}; fetch them with "
                           f"{self.binary} --models {self.model_dir} --download")
        cmd = [str(self.binary), "--models", str(self.model_dir), "--encoder", self.encoder]
        if sandbox_profile is not None:
            cmd = ["sandbox-exec", "-f", str(sandbox_profile)] + cmd
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        self._replies = queue.Queue()
        threading.Thread(target=self._read_stdout, args=(self.proc, log), daemon=True,
                         name="parakeet-stdout").start()
        try:
            r = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            self.stop()
            raise AsrError(f"Parakeet not ready after {timeout_s:.0f} s (see {self.log_path})")
        if not r.get("ready"):
            self.stop()
            raise AsrError(f"Parakeet failed to start: {r.get('error')}")
        self.startup = r
        return time.time() - t0

    def _read_stdout(self, proc, log):
        for line in proc.stdout:
            if line.startswith(b"@@"):
                self._replies.put(json.loads(line[2:]))
            elif log is not subprocess.DEVNULL:
                log.write(line)
        self._replies.put({"error": f"Parakeet exited (code {proc.poll()})", "exited": True})

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(struct.pack("<I", 0))
                self.proc.stdin.flush()
                self.proc.wait(5)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
        self.proc = None

    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc and self.proc.poll() is None else None

    def healthy(self) -> bool:
        return self.pid() is not None

    def transcribe(self, audio, sr: int = 16000, timeout_s: float = 30) -> dict:
        """audio: float32 samples (16 kHz) or WAV bytes. Returns {'text', 'seconds', 'raw'}."""
        if not self.healthy():
            raise AsrError("speech recognition failed: Parakeet is not running")
        x = wav_to_samples(audio) if isinstance(audio, (bytes, bytearray)) else np.asarray(audio, np.float32).reshape(-1)
        if sr != 16000:
            raise AsrError(f"expected 16 kHz audio, got {sr}")
        with self._lock:
            while not self._replies.empty():          # a reply that arrived after an earlier timeout
                self._replies.get_nowait()
            t0 = time.perf_counter()
            try:
                self.proc.stdin.write(struct.pack("<I", len(x)) + x.astype("<f4").tobytes())
                self.proc.stdin.flush()
                raw = self._replies.get(timeout=timeout_s)
            except (OSError, queue.Empty) as e:
                raise AsrError(f"speech recognition failed: {e!r}") from e
        if "error" in raw:
            raise AsrError(f"speech recognition failed: {raw['error']}")
        return {"text": raw.get("text", "").strip(), "seconds": time.perf_counter() - t0, "raw": raw}


class Partials(threading.Thread):
    """Partial transcripts while the talk key is held (recall mode, ARCHITECTURE.md §5.9). Parakeet has no streaming
    API here, so the audio so far is transcribed again, every `every_s` once `min_new_s` of new audio has arrived
    (~40 ms for a 3 s question on the Neural Engine, which nothing else uses during a turn, §6.1).

    get_audio() -> samples so far; voiced_until() -> sample index where the last voiced block ended;
    on_partial(text) for each changed transcript. `final_text()` after the key is released: the last partial
    when it covered every voiced sample (so the rest of the recording is silence), else None and the caller
    transcribes the whole recording. Settings are provisional (set before measuring).

    A partial also runs as soon as speech stops (`settle_s` of silence after the last voiced block that the last
    partial didn't reach), not only on the `every_s` tick: when you pause before releasing the key, the last partial
    then covers everything and the final transcription is skipped (M3, 2026-11-13)."""

    def __init__(self, asr, get_audio, voiced_until, on_partial, every_s: float = 0.3, min_new_s: float = 0.25,
                 min_audio_s: float = 0.5, sr: int = 16000, settle_s: float = 0.1, tick_s: float = 0.05):
        super().__init__(daemon=True, name="asr-partials")
        self.asr, self.get_audio, self.voiced_until, self.on_partial = asr, get_audio, voiced_until, on_partial
        self.every_s, self.min_new, self.min_audio, self.sr = every_s, int(min_new_s * sr), int(min_audio_s * sr), sr
        self.settle, self.tick_s = int(settle_s * sr), tick_s
        self.stop_ev = threading.Event()
        self.last: Optional[tuple[int, str]] = None     # (samples covered, text)
        self.stats = {"partials": 0, "calls": 0, "asr_s": 0.0, "errors": 0, "settled": 0}

    def settled(self, n: int) -> bool:
        """Speech stopped (settle_s of silence after the last voiced block) and the last partial didn't reach it."""
        v = self.voiced_until()
        covered = self.last[0] if self.last else 0
        return 0 <= v and covered < v and n - v >= self.settle

    def step(self, force: bool = False) -> bool:
        """Transcribe the audio so far if enough is new (or `force`). Returns True if it ran."""
        x = self.get_audio()
        covered = self.last[0] if self.last else 0
        if len(x) < self.min_audio or self.voiced_until() < 0 or (not force and len(x) - covered < self.min_new):
            return False
        try:
            r = self.asr.transcribe(x)
        except AsrError:
            self.stats["errors"] += 1
            return False
        self.stats["calls"] += 1
        self.stats["asr_s"] += r.get("seconds", 0.0)
        prev = self.last[1] if self.last else None
        self.last = (len(x), r["text"])
        if r["text"] and r["text"] != prev:
            self.stats["partials"] += 1
            self.on_partial(r["text"])
        return True

    def run(self):
        import time
        last = time.perf_counter()
        while not self.stop_ev.wait(self.tick_s):
            settled = self.settled(len(self.get_audio()))
            if settled or time.perf_counter() - last >= self.every_s:
                if self.step(force=settled) and settled:
                    self.stats["settled"] += 1
                last = time.perf_counter()

    def finish(self, timeout: float = 2.0):
        self.stop_ev.set()
        self.join(timeout)

    def final_text(self) -> Optional[str]:
        if self.last is None or not self.last[1]:
            return None
        v = self.voiced_until()
        return self.last[1] if 0 <= v <= self.last[0] else None


def make_asr(log_dir: Optional[Path] = None):
    return ParakeetServer(log_path=log_dir / "parakeet.log" if log_dir else None)
