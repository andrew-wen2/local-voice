"""asr.py — speech recognition through a local WhisperKit server (runs on the Neural Engine).

The server is a separate process (`whisperkit-cli serve`) bound to 127.0.0.1. This module
starts it, waits for /health, sends a finished recording to its OpenAI-style
`POST /v1/audio/transcriptions` endpoint, and stops it. Loopback only: nothing leaves the Mac.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models/whisperkit-coreml/openai_whisper-large-v3-v20240930_turbo_632MB"
TOKENIZER_DIR = ROOT / "models/whisper-tokenizer"


MIN_CLIP_S = 1.2   # WhisperKit returns an empty transcript for clips under ~1 s ("What's this?" = 0.92 s)


def pad_short(audio, sr: int = 16000, min_s: float = MIN_CLIP_S):
    """Pad a short clip with trailing silence so WhisperKit transcribes it."""
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(min_s * sr)
    return audio if len(audio) >= n else np.concatenate([audio, np.zeros(n - len(audio), np.float32)])


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


class WhisperServer:
    def __init__(self, port: int = 50060, model_dir: Path = MODEL_DIR, tokenizer_dir: Path = TOKENIZER_DIR,
                 log_path: Optional[Path] = None):
        self.port, self.model_dir, self.tokenizer_dir = port, Path(model_dir), Path(tokenizer_dir)
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None

    # -- lifecycle
    def start(self, timeout_s: float = 600, sandbox_profile: Optional[Path] = None) -> float:
        """Start the server (optionally under sandbox-exec) and wait for /health. Returns seconds taken."""
        if self.healthy():
            return 0.0
        exe = shutil.which("whisperkit-cli")
        if exe is None:
            raise AsrError("whisperkit-cli not found; install with `brew install whisperkit-cli`")
        cmd = [exe, "serve", "--model-path", str(self.model_dir), "--download-tokenizer-path",
               str(self.tokenizer_dir), "--host", "127.0.0.1", "--port", str(self.port)]
        if sandbox_profile is not None:
            cmd = ["sandbox-exec", "-f", str(sandbox_profile)] + cmd
        out = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                     env={**os.environ, "HF_HUB_OFFLINE": "1"})
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise AsrError(f"WhisperKit server exited with code {self.proc.returncode} (see {self.log_path})")
            if self.healthy():
                return time.time() - t0
            time.sleep(0.5)
        raise AsrError(f"WhisperKit server not healthy after {timeout_s:.0f} s")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc and self.proc.poll() is None else None

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=1) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    # -- transcription
    def transcribe(self, audio, sr: int = 16000, timeout_s: float = 30) -> dict:
        """audio: float32 samples (16 kHz) or WAV bytes. Returns {'text', 'seconds', 'raw'}."""
        if not isinstance(audio, (bytes, bytearray)):
            audio = pad_short(audio, sr)
        data = audio if isinstance(audio, (bytes, bytearray)) else wav_bytes(audio, sr)
        boundary = uuid.uuid4().hex
        parts = []
        for name, val in (("model", "whisper"), ("language", "en")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{val}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="turn.wav"\r\n'
                     f"Content-Type: audio/wav\r\n\r\n".encode() + bytes(data) + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(self.url + "/v1/audio/transcriptions", data=b"".join(parts), method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                raw = json.loads(r.read())
        except (urllib.error.URLError, OSError) as e:
            raise AsrError(f"speech recognition failed: {e}") from e
        text = raw.get("text") or " ".join(s.get("text", "") for s in raw.get("segments", []))
        return {"text": text.strip(), "seconds": time.perf_counter() - t0, "raw": raw}
