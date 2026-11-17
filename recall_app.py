"""recall_app.py — recall mode's wiring as one object (ARCHITECTURE.md §5.9, §5.12, §5.27).

The frames the app reads are OCR'd into pages and visits (recall.py), kept screens are encoded (Neural Engine when it
can) and keyed, and the turn scheduler (scheduler.py, recall_worker.py) runs the GPU. While the key is held, partial
transcripts drive retrieval, exact page upgrades and the question prefill; release prefills what is left and answers.

The inputs are injectable so the live app (`python app.py`) and the replay (`eval/replay_recall.py`) run the same
code: `grab` / `grab_small` (the screen), `windows_fn` (the window list), `mic` (audio_io.Mic or anything with its
interface), `speaker` (audio_io.Speaker or a sink), and `clock` / `sleep` for the screen watcher.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import screen as scr
from asr import AsrError, Partials, make_asr
from audio_io import SILENCE_S, near_silent, rms_dbfs
from engine import RECALL_INSTRUCTION, GemmaAdapter, make_tts
from recall import Ocr, Store, windows
from recall_backend import TurnIO
from recall_worker import RecallWorker, ScreenEncoder
from screen_store import ScreenStore

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
WARM_WAV = ROOT / "harness/questions/v01.wav"
WARM_Q = "What's on my screen right now?"


class RecallApp:
    def __init__(self, *, grab: Callable, speaker, mic, grab_small: Optional[Callable] = None,
                 windows_fn: Callable[[], list] = windows, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, budget: int = 1120, swap_weights: bool = True,
                 freshness: str = "B", first_chunk: str = "phrase", trim_silence: bool = True,
                 mlx_cache_gb: Optional[float] = 0.0, use_ane: bool = False, log_path: Optional[Path] = None,
                 show_text: bool = False, on_row: Optional[Callable[[dict], None]] = None, monitor: bool = True,
                 on_ui: Optional[Callable[[dict], None]] = None):
        self.grab_screen, self.grab_small, self.windows_fn = grab, grab_small, windows_fn
        self.clock, self.sleep = clock, sleep
        self.speaker, self.mic = speaker, mic
        self.budget, self.swap_weights, self.freshness = budget, swap_weights, freshness
        self.first_chunk, self.trim_silence, self.mlx_cache_gb = first_chunk, trim_silence, mlx_cache_gb
        self.use_ane, self.log_path, self.show_text, self.on_row = use_ane, log_path, show_text, on_row
        self.want_monitor = monitor
        self.on_ui = on_ui                               # caption events for the menu bar panel (menubar.py)
        self.asr = self.tts = self.store = self.screens = self.worker = None
        self.ane = self.encoder = self.ocr = self.watcher = self.monitor = None
        self.before: dict = {}
        self.turn: dict = {}
        self.seq = 0
        self.rows: deque = deque(maxlen=100)            # the latest turns (all of them are in the log file)
        self.startup: dict = {}

    def _ui(self, ev: str, **kw):
        if self.on_ui is not None:
            try:
                self.on_ui({"ev": ev, **kw})
            except Exception:
                pass                                     # captions never break a turn

    def data_dirs(self) -> list[str]:
        """Folders holding what it saw (kept screens, stored pieces); deleted on stop, listed so a hard kill can't leave
        them behind (menubar.py deletes them after the process ends)."""
        dirs = [getattr(self.screens, "dir", None), getattr(getattr(self.worker, "backend", None), "dir", None)]
        return [str(d) for d in dirs if d is not None]

    # ------------------------------------------------------------ start / stop

    def start(self) -> dict:
        """Load everything, read the first screen, build the first prefix, start watching. Returns startup info."""
        LOGS.mkdir(exist_ok=True)
        t_all = time.perf_counter()
        self.asr = make_asr(log_dir=LOGS)
        self.asr.start(sandbox_profile=None)
        self.asr.transcribe(WARM_WAV.read_bytes())
        self.tts = make_tts(log_dir=LOGS)
        self.store = Store(start=datetime.fromtimestamp(self.clock()))
        self.screens = ScreenStore()                     # kept screens: owner-only folder, deleted on quit (§5.12)
        adapter = GemmaAdapter(swap_weights=self.swap_weights, instruction=RECALL_INSTRUCTION)
        self.worker = RecallWorker(adapter, self.tts, self.store, budget=self.budget, freshness=self.freshness,
                                   first_chunk=self.first_chunk, trim_silence=self.trim_silence,
                                   mlx_cache_gb=self.mlx_cache_gb, screen_store=self.screens)
        print("starting (first launch can take a few minutes)...", flush=True)
        self.worker.start()
        self.worker.ready.wait()
        if self.worker.error:
            raise self.worker.error
        self._ui("dirs", paths=self.data_dirs())        # before anything is kept, so a kill during loading can't leak them
        if self.use_ane:            # off (Andrew, 2026-11-13): the Neural Engine copy held ~0.74 GB; kept screens use the GPU
            try:
                from ane_vision import OUT, AneProcess
                self.ane = AneProcess(OUT).load()
            except Exception as e:
                print(f"(Neural Engine encoder unavailable, kept screens are encoded on the GPU: {e!r})", flush=True)
        quiet = lambda: not self.worker.turn_active
        self.encoder = ScreenEncoder(adapter, self.screens, quiet=quiet, on_ready=self.worker.key_frame, ane=self.ane)
        self.encoder.start()
        self.ocr = Ocr(self.store, quiet=quiet, on_visit_end=self.worker.visit_ended,
                       t0=self.store.start.timestamp(), on_keep=self.encoder.submit, on_unit_text=self.screens.add_lines)
        self.ocr.start()
        if self.want_monitor:
            from app import Monitor
            self.monitor = Monitor(self._pids)
            self.monitor.start()
        first = self._grab()
        self._on_frame(0, self.clock(), first)
        if not self.worker.wait_ready(300):
            raise RuntimeError("first screen read or prefix did not finish")
        self.watcher = scr.ScreenWatcher(on_frame=self._on_frame, grab=self._grab, grab_small=self.grab_small,
                                         clock=self.clock, sleep=self.sleep)
        self.watcher.last_sent = np.asarray(first)
        self.watcher.last_sent_small = self.grab_small() if self.grab_small else scr.shrink(first)
        self.watcher.start()
        self.startup = {**self.worker.startup, "warmup_s": self._warm_up(),
                        "ready_s": round(time.perf_counter() - t_all, 1)}
        return self.startup

    def _warm_up(self) -> float:
        """One silent question through the answer path (retrieval, assembly, prefill, decode, Kokoro) before the first
        real one, so first-use costs are paid at startup (M3, 2026-11-13). Its audio is dropped; it isn't kept in the
        conversation. Returns seconds."""
        t = time.perf_counter()
        io = TurnIO(emit_audio=lambda samples, sr: None, mark=lambda name: None, warmup=True)
        io.question = WARM_Q
        self.worker.key_down()
        self.worker.partial(WARM_Q)
        self.worker.key_up(io)
        self.worker.final(WARM_Q)
        io.done.wait(120)
        self.worker.turn_active = False
        return round(time.perf_counter() - t, 1)

    def _pids(self) -> dict:
        ane_proc = getattr(self.ane, "proc", None)
        return {"app": os.getpid(), "asr": self.asr.pid() if self.asr else None,
                "tts": self.tts.pid() if self.tts else None, "ane": ane_proc.pid if ane_proc else None}

    def stop(self):
        for x in (self.watcher, self.ocr, self.encoder):
            if x is not None:
                x.stop()
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(10)
        if self.ane is not None:
            self.ane.close()
        if self.screens is not None:
            self.screens.close()                         # kept screens deleted on quit
        if self.tts is not None:
            self.tts.stop()
        if self.asr is not None:
            self.asr.stop()
        if self.monitor is not None:
            self.monitor.stop()

    # ------------------------------------------------------------ screen

    def _grab(self):
        self.before["windows"] = self.windows_fn()       # the window list just before the screenshot (§12.9)
        return self.grab_screen()

    def _on_frame(self, frame_id, t, img, note=None):
        self.worker.frame(frame_id, img)
        self.ocr.submit(t, img, self.before.get("windows"), self.windows_fn(), fid=frame_id)

    # ------------------------------------------------------------ turns

    def busy(self) -> bool:
        return self.turn.get("state") == "answering"

    def press(self, source: str = "key"):
        """Talk key down: read the current screen now, start recording and partial transcripts."""
        self.worker.key_down()
        self.watcher.capture_now()
        self.mic.begin()
        self.turn = {}
        self.seq += 1
        self.turn.update(source=source, t_press=time.perf_counter(), stop=threading.Event(), seq=self.seq,
                         state="recording")
        self._ui("listening")
        on_partial = self.worker.partial
        if self.on_ui is not None:
            def on_partial(text, _p=self.worker.partial):
                _p(text)
                self._ui("heard", text=text)
        self.turn["partials"] = Partials(self.asr, self.mic.audio, self.mic.voiced_until, on_partial)
        self.turn["partials"].start()
        threading.Thread(target=self._watch_voice, args=(self.turn["stop"],), daemon=True).start()

    def _watch_voice(self, stop):
        """mic_silent / speech events while the key is held (scheduling only, §5.9)."""
        silent = False
        while not stop.wait(0.05):
            s = self.mic.silent_for()
            if s is None:
                continue
            if not silent and s >= SILENCE_S:
                silent = True
                self.worker.mic_silent()
            elif silent and s < SILENCE_S:
                silent = False
                self.worker.speech()

    def cancel_tap(self):
        """A tap, not a question: drop the turn."""
        self.mic.end()
        self.turn["stop"].set()
        self.turn["partials"].finish()
        self.worker.cancel_turn()
        self.worker.turn_active = False
        self.turn["state"] = "idle"
        self._ui("idle")

    def release(self, extra: Optional[dict] = None) -> threading.Thread:
        """Talk key up: finish the transcript, answer, play it, log the turn. Returns the turn's thread."""
        audio = self.mic.end()
        t_eos, t_eos_wall = time.perf_counter(), time.time()
        turn = self.turn
        turn["stop"].set()
        marks: dict = {}
        first_audio: dict = {}

        def mark(n):
            marks.setdefault(n, time.perf_counter())

        def emit(samples, sr):
            if "t" not in first_audio:
                from harness.harness import first_audible
                i = first_audible(samples)
                if i is not None:
                    first_audio["t"] = time.perf_counter() + i / sr
            self.speaker.play(samples, sr)

        io = TurnIO(emit_audio=emit, mark=mark,
                    on_say=(lambda text: self._ui("say", text=text)) if self.on_ui is not None else None)
        turn["io"] = io
        my = turn["seq"]
        self._ui("thinking")
        self.worker.key_up(io)

        def run():
            row = {"mode": "live-recall", "near_silent": near_silent(audio), "input_dbfs": round(rms_dbfs(audio), 1),
                   **(extra or {})}
            p = turn["partials"]
            p.finish()
            text = p.final_text()
            row["final_from_partial"] = text is not None
            if text is None and not row["near_silent"]:
                try:
                    text = self.asr.transcribe(audio)["text"]
                except AsrError as e:
                    row["error"] = str(e)
            mark("asr_done")
            row.update(transcript=text, partials=p.stats["partials"], partial_calls=p.stats["calls"])
            if not text:
                self.worker.cancel_turn()
                row.setdefault("error", "empty transcript" + (" (mic muted or no permission?)" if row["near_silent"] else ""))
            else:
                io.question = text
                self.worker.final(text)
                self._ui("heard", text=text, final=True)
            io.done.wait(120)
            row.update(io.result)
            self.speaker.wait_done(60)
            if self.seq == my:                           # not replaced by a newer question (barge-in)
                self.worker.turn_active = False
                self.ocr.poke()
                self.encoder.poke()
            ms = {k: round((v - t_eos) * 1000, 1) for k, v in marks.items()}
            row.update(t_eos_wall=time.time(), marks_ms=ms,
                       first_audio_ms=round((first_audio["t"] - t_eos) * 1000, 1) if "t" in first_audio else None,
                       output_device_latency_ms=self.speaker.device_latency_ms(), store_pages=len(self.store.m.pages),
                       store_visits=len(self.store.visits), ocr_frames=self.ocr.stats["frames"],
                       kept_screens=len(self.screens.ids()), screens_pending=len(self.screens.pending),
                       encoder=dict(self.encoder.stats))
            if self.monitor is not None:
                row.update(self.monitor.window(t_eos_wall - 5, time.time()))
            if self.show_text:
                print(f"  you: {row.get('transcript')!r}\n  it:  {row.get('answer') or row.get('error')!r}", flush=True)
            self.rows.append(row)
            if self.log_path is not None:
                with self.log_path.open("a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                os.chmod(self.log_path, 0o600)
            if self.on_row is not None:
                self.on_row(row)
            if self.seq == my:
                turn["state"] = "idle"
                self._ui("done", error=row.get("error"))

        turn["state"] = "answering"
        turn["thread"] = threading.Thread(target=run, daemon=True)
        turn["thread"].start()
        return turn["thread"]

    def interrupt(self) -> int:
        """Barge-in: drop the current answer and its queued audio. Returns samples dropped."""
        self.worker.interrupt()
        return self.speaker.clear()
