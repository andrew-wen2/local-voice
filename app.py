#!/usr/bin/env python
"""app.py — offline, screen-aware voice assistant (MVP, naive baseline).

  python app.py                      live, with memory of past screens (recall mode, ARCHITECTURE.md §5.9): hold Right
                                     Option in any app to talk (release = stop, press while it speaks = interrupt);
                                     Enter in the terminal also works (talk / stop / interrupt), q = quit
  python app.py --screen-only        the previous app: the current screen only, no memory of past screens
  python app.py --replay reading     replay a test set through the timing harness (reading|reasoning|voice|contention)
  python app.py --replay voice --sandbox-profile eval/offline.sb   (used by the offline test)

Review decisions implemented here: readiness gate (D8), Screen Recording check + near-silent
flag (D5), interrupt stops the work (D6), per-turn logs with combined memory (D19), screen age
from capture time (D16), contention cases (D17), offline env (D3/D14), private logs (D4).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402

import screen as scr  # noqa: E402
from asr import AsrError, make_asr  # noqa: E402
from audio_io import Mic, Speaker, near_silent, rms_dbfs  # noqa: E402
from engine import RECALL_INSTRUCTION, AnswerJob, GemmaAdapter, GpuWorker, make_tts  # noqa: E402

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
QDIR = ROOT / "harness/questions"
M0_SCREENS = ROOT / "m0_screen/screens"


# ---------------------------------------------------------------- memory, power, fans

class _RusageInfoV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
        "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
        "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
        "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten")]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint_bytes(pid: int) -> Optional[int]:
    """Physical footprint of a process (includes GPU/unified-memory allocations)."""
    info = _RusageInfoV2()
    if _libproc.proc_pid_rusage(int(pid), 2, ctypes.byref(info)) != 0:
        return None
    return int(info.phys_footprint)


class Monitor(threading.Thread):
    """Once a second: combined memory of this app + the speech recognition process (D19), and macmon power/fan if available."""

    def __init__(self, pids_fn):
        super().__init__(daemon=True, name="monitor")
        self.pids_fn = pids_fn
        self.samples: deque = deque(maxlen=900)         # once a second: the last 15 minutes
        self._stop = threading.Event()
        self.macmon = None
        try:
            self.macmon = subprocess.Popen(["macmon", "pipe", "-i", "1000"], stdout=subprocess.PIPE,
                                           stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read_macmon, daemon=True).start()
        except FileNotFoundError:
            pass
        self.power: dict = {}

    def _read_macmon(self):
        for line in self.macmon.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.power = {"all_w": d.get("all_power"), "gpu_w": d.get("gpu_power"), "ane_w": d.get("ane_power"),
                          "fan_rpm": (d.get("fans") or [{}])[0].get("rpm")}

    def run(self):
        while not self._stop.is_set():
            mem = {}
            total = 0
            for name, pid in self.pids_fn().items():
                fp = footprint_bytes(pid) if pid else None
                mem[name] = fp
                total += fp or 0
            self.samples.append({"t": time.time(), "mem_total": total, **{f"mem_{k}": v for k, v in mem.items()},
                                 **self.power})
            self._stop.wait(1.0)

    def window(self, t0: float, t1: float) -> dict:
        s = [x for x in list(self.samples) if t0 - 1 <= x["t"] <= t1 + 1]   # snapshot: the monitor thread appends
        if not s:
            return {}
        gb = lambda b: round(b / 1e9, 2)
        out = {"mem_peak_gb": gb(max(x["mem_total"] for x in s))}
        for k in ("mem_app", "mem_asr", "mem_ane", "mem_tts"):
            v = [x[k] for x in s if x.get(k)]
            if v:
                out[f"{k}_peak_gb"] = gb(max(v))
        w = [x["all_w"] for x in s if x.get("all_w") is not None]
        if w:
            out["power_mean_w"] = round(statistics.mean(w), 2)
        f = [x["fan_rpm"] for x in s if x.get("fan_rpm") is not None]
        if f:
            out["fan_max_rpm"] = max(f)
        return out

    def stop(self):
        self._stop.set()
        if self.macmon:
            self.macmon.terminate()


# ---------------------------------------------------------------- the assistant

class Assistant:
    def __init__(self, budget: int = 1120, sandbox_profile: Optional[Path] = None, swap_weights: bool = True,
                 **worker_opts):
        """worker_opts go to GpuWorker: first_chunk, trim_silence, mlx_cache_gb, policy, freshness,
        fresh_ms, layers_per_piece, min_read_interval_s, duty_cap."""
        self.budget = budget
        self.worker_opts = worker_opts
        LOGS.mkdir(exist_ok=True)
        self.asr = make_asr(log_dir=LOGS)
        self.sandbox_profile = sandbox_profile
        self.tts = make_tts(sandbox_profile=sandbox_profile, log_dir=LOGS)
        self.swap_weights = swap_weights
        self.worker: Optional[GpuWorker] = None
        self.monitor = Monitor(lambda: {"app": os.getpid(), "asr": self.asr.pid(), "ane": self._ane_pid(),
                                        "tts": self.tts.pid()})
        self.startup: dict = {}
        self.ready = False

    def start(self, warm_image, first_frame=None):
        """Readiness gate (D8): ASR healthy + warm, models loaded + warm, first screen read done."""
        t_all = time.perf_counter()
        self.monitor.start()
        self.startup["asr_start_s"] = round(self.asr.start(sandbox_profile=self.sandbox_profile), 2)
        t0 = time.perf_counter()
        self.asr.transcribe((QDIR / "v01.wav").read_bytes())
        self.startup["asr_warm_s"] = round(time.perf_counter() - t0, 2)
        self.startup["swap_weights"] = self.swap_weights
        self.worker = GpuWorker(GemmaAdapter(swap_weights=self.swap_weights), self.tts,
                                budget=self.budget,
                                warm_image=warm_image,
                                **self.worker_opts)
        self.worker.start()
        self.worker.ready.wait()
        if self.worker.error:
            raise self.worker.error
        self.startup.update(self.worker.startup)
        if first_frame is not None:
            self.worker.submit_frame(*first_frame)
        if not self.worker.first_snapshot_wait(120):
            raise RuntimeError("first screen read did not finish")
        while self.worker.snapshot.spare is None:
            time.sleep(0.02)
        self.startup["total_s"] = round(time.perf_counter() - t_all, 2)
        self.ready = True

    def _ane_pid(self):
        """The Neural Engine encoder's child process (placement split), counted in the memory budget."""
        ane = getattr(self.worker, "ane", None)
        return ane.pid() if hasattr(ane, "pid") else None

    def stop(self):
        if self.worker:
            self.worker.stop()
        if hasattr(self.tts, "stop"):
            self.tts.stop()
        self.asr.stop()
        self.monitor.stop()

    def answer_turn(self, audio, emit_audio, mark, t_eos_wall: float, t_eos: float) -> dict:
        """Everything after end of speech. Returns the per-turn log row (times in ms from end of speech)."""
        if not self.ready:
            return {"error": "not ready", "refused": True}
        row = {"near_silent": None}
        if not isinstance(audio, (bytes, bytearray)):
            row["near_silent"] = near_silent(audio)
            row["input_dbfs"] = round(rms_dbfs(audio), 1)
            if row["near_silent"]:          # Parakeet turns a padded near-silent tap into "Okay." / "Uh"
                mark("turn_complete")
                return {**row, "transcript": "", "error": "empty transcript (mic muted or no permission?)"}
        try:
            tr = self.asr.transcribe(audio)
        except AsrError as e:
            mark("turn_complete")
            return {**row, "error": str(e)}
        mark("asr_done")
        row["transcript"] = tr["text"]
        if not tr["text"]:
            mark("turn_complete")
            return {**row, "error": "empty transcript" + (" (mic muted or no permission?)" if row["near_silent"] else "")}
        job = self.worker.submit_answer(AnswerJob(tr["text"], emit_audio=emit_audio, mark=mark))
        job.done.wait(120)
        r = job.result
        row.update({k: r.get(k) for k in ("answer", "stop", "cap_hit", "cancelled", "sentences_spoken", "chunks_spoken", "first_chunk_text", "first_chunk_mode", "frame_id",
                                          "budget", "read_in_flight_at_submit", "question_tokens", "error",
                                          "policy", "fresh_decision", "read_paused", "stale_vs_submit",
                                          "context_tokens", "history_exchanges")})
        row["wait_behind_read_ms"] = round(r.get("wait_behind_read_s", 0) * 1000, 1)
        row["restore_ms"] = round(r.get("restore_s", 0) * 1000, 2)
        if r.get("screen_captured_at"):
            row["screen_age_s"] = round(t_eos_wall - r["screen_captured_at"], 3)        # from capture (D16)
            row["screen_read_done_before_eos_s"] = round(t_eos_wall - r["screen_read_done"], 3)
        return row


# ---------------------------------------------------------------- live mode

def talk_events(args):
    """Talk-key events: "down" / "up" from the global hotkey helper, "enter" / "quit" from the terminal.
    Returns (queue, hotkey process or None, status line)."""
    import queue
    events: "queue.Queue[str]" = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            events.put("quit" if line.strip().lower() == "q" else "enter")
        events.put("quit")
    threading.Thread(target=read_stdin, daemon=True).start()

    hotkey_proc, hotkey_msg = None, "off"
    hk = getattr(args, "hotkey", "right_option")
    if hk != "none" and (ROOT / "tools/hotkey").exists():
        hotkey_proc = subprocess.Popen([str(ROOT / "tools/hotkey"), hk], stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True)
        first_line = hotkey_proc.stdout.readline().strip()
        if first_line.startswith("ready"):
            hotkey_msg = f"hold {hk.replace('_', ' ')} anywhere = talk (release = stop, press while speaking = interrupt)"

            def read_hotkey():
                for line in hotkey_proc.stdout:
                    events.put(line.strip())
            threading.Thread(target=read_hotkey, daemon=True).start()
        else:
            hotkey_msg = f"global hotkey unavailable: {first_line or 'no output'}"
            hotkey_proc = None
    return events, hotkey_proc, hotkey_msg


def live(args):
    if not scr.has_screen_permission():
        print("Screen Recording permission is off for this terminal app.\n"
              "Turn it on in System Settings > Privacy & Security > Screen & System Audio Recording,\n"
              "then restart the terminal. (Without it, macOS returns only the wallpaper.)")
        sys.exit(2)
    cap, grab_small = scr.open_capture()
    asst = Assistant(budget=args.budget, swap_weights=getattr(args, "swap_weights", "on") == "on", **worker_opts(args))
    speaker = Speaker().start()
    mic = Mic().start()
    print("starting (first launch can take a few minutes)...", flush=True)
    first = cap.grab()
    asst.start(warm_image=first, first_frame=(0, time.time(), first))
    watcher = scr.ScreenWatcher(on_frame=asst.worker.submit_frame, grab=cap.grab, describe=scr.window_summary,
                                grab_small=grab_small)
    watcher.last_sent = np.asarray(first)
    # the gate compares like with like: the stream's own small frame, or our shrink of a full grab
    watcher.last_sent_small = grab_small() if grab_small else scr.shrink(first)
    watcher.start()
    log = LOGS / f"turns-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    events, hotkey_proc, hotkey_msg = talk_events(args)
    print(f"ready in {asst.startup['total_s']} s  {asst.startup}\n"
          f"Enter = talk / stop / interrupt, q = quit;  {hotkey_msg}", flush=True)

    state = "idle"
    turn: dict = {}

    def start_recording(source):
        asst.worker.set_turn_active(True)
        turn["apps_at_press"] = scr.front_apps()
        watcher.capture_now()                          # talk key: read the current screen during the question
        mic.begin()
        turn["source"], turn["t_press"] = source, time.perf_counter()
        print("listening..." + (" (release to stop)" if source == "hotkey" else " (Enter to stop)"), flush=True)

    def interrupt():
        t_int = asst.worker.interrupt()
        dropped = speaker.clear()
        print(f"  interrupted ({dropped / speaker.sr:.1f} s of audio dropped)", flush=True)
        turn["thread"].join(10)
        print(f"  GPU free {round((time.perf_counter() - t_int) * 1000)} ms after interrupt", flush=True)

    def stop_and_answer():
        audio = mic.end()
        t_eos, t_eos_wall = time.perf_counter(), time.time()
        marks: dict = {}
        first_audio: dict = {}

        def emit(samples, sr):
            if "t" not in first_audio:
                from harness.harness import first_audible
                i = first_audible(samples)
                if i is not None:
                    first_audio["t"] = time.perf_counter() + i / sr
            speaker.play(samples, sr)

        apps_at_eos = scr.front_apps()
        last_handed = dict(watcher.handed[-1]) if watcher.handed else {}

        def run():
            row = asst.answer_turn(audio, emit, lambda n: marks.setdefault(n, time.perf_counter()), t_eos_wall, t_eos)
            used = next((h for h in reversed(list(watcher.handed)) if h["frame_id"] == row.get("frame_id")), {})
            row.update(apps_at_press=turn.get("apps_at_press"), apps_at_eos=apps_at_eos, talk_source=turn.get("source"),
                       answered_frame={k: used.get(k) for k in ("frame_id", "reason", "apps", "waited_s")},
                       answered_frame_age_s=round(t_eos_wall - used["captured_at"], 2) if used else None,
                       newest_frame_at_eos={k: last_handed.get(k) for k in ("frame_id", "reason", "apps")},
                       frames_sent_total=watcher.frame_id, displays=scr.active_displays())
            speaker.wait_done(60)
            asst.worker.set_turn_active(False)
            ms = {k: round((v - t_eos) * 1000, 1) for k, v in marks.items()}
            row.update(mode="live", t_eos_wall=t_eos_wall, marks_ms=ms,
                       first_audio_ms=round((first_audio["t"] - t_eos) * 1000, 1) if "t" in first_audio else None,
                       output_device_latency_ms=speaker.device_latency_ms(),
                       capture=dict(cap.stats), **asst.monitor.window(t_eos_wall - 5, time.time()))
            if row.get("near_silent"):
                print("  (recording was nearly silent: mic muted or no Microphone permission?)")
            # Don't print the words: the assistant sees this terminal and would read its own answers
            # back as "terminal output" (live session 2026-11-06). Full text is in the log file.
            if getattr(args, "show_text", False):
                print(f"  you: {row.get('transcript')!r}\n  it:  {row.get('answer') or row.get('error')!r}", flush=True)
            status = "error: " + row["error"] if row.get("error") else "answered"
            print(f"  {status}, first audio {row['first_audio_ms']} ms after you stopped", flush=True)
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            turn["state"] = "idle"

        turn["state"] = "answering"
        turn["thread"] = threading.Thread(target=run, daemon=True)
        turn["thread"].start()

    while True:
        ev = events.get()
        if ev == "quit":
            break
        if state == "answering" and turn.get("state") == "idle":
            state = "idle"
        if ev in ("enter", "down"):
            if state == "recording":
                if ev == "enter":                           # Enter toggles: stop and answer
                    stop_and_answer()
                    state = "answering"
                continue
            if state == "answering":                        # barge in: stop the answer, then listen
                interrupt()
            start_recording("hotkey" if ev == "down" else "enter")
            state = "recording"
        elif ev == "up" and state == "recording" and turn.get("source") == "hotkey":
            if time.perf_counter() - turn["t_press"] < 0.25:  # a tap, not a question: discard
                mic.end()
                asst.worker.set_turn_active(False)
                state = "idle"
                print("  (too short, ignored)", flush=True)
                continue
            stop_and_answer()
            state = "answering"
    if hotkey_proc:
        hotkey_proc.terminate()
    watcher.stop()
    asst.stop()
    speaker.stop()
    mic.stop()
    cap.cleanup()


# ---------------------------------------------------------------- live mode with recall (ARCHITECTURE.md §5.9)

def live_recall(args):
    """Live mode with memory of past screens (recall_app.RecallApp; ARCHITECTURE.md §5.9, §5.12)."""
    from recall_app import RecallApp
    if not scr.has_screen_permission():
        print("Screen Recording permission is off for this terminal app (System Settings > Privacy & Security > "
              "Screen & System Audio Recording); restart the terminal after turning it on.")
        sys.exit(2)
    LOGS.mkdir(exist_ok=True)
    cap, grab_small = scr.open_capture()
    speaker, mic = Speaker().start(), Mic().start()
    o = worker_opts(args)

    def on_row(row):
        status = "error: " + row["error"] if row.get("error") else "answered"
        print(f"  {status}, first audio {row['first_audio_ms']} ms after you stopped", flush=True)

    app = RecallApp(grab=cap.grab, grab_small=grab_small, speaker=speaker, mic=mic, budget=args.budget,
                    swap_weights=getattr(args, "swap_weights", "on") == "on",
                    freshness="C" if o["freshness"] == "C" else "B", first_chunk=o["first_chunk"],
                    trim_silence=o["trim_silence"], mlx_cache_gb=o["mlx_cache_gb"],
                    log_path=LOGS / f"turns-recall-{time.strftime('%Y%m%d-%H%M%S')}.jsonl",
                    show_text=getattr(args, "show_text", False), on_row=on_row)
    startup = app.start()
    events, hotkey_proc, hotkey_msg = talk_events(args)
    print(f"ready in {startup['ready_s']} s  {startup}\n"
          f"Enter = talk / stop / interrupt, q = quit;  {hotkey_msg}", flush=True)
    state = "idle"
    while True:
        ev = events.get()
        if ev == "quit":
            break
        if state == "answering" and not app.busy():
            state = "idle"
        if ev in ("enter", "down"):
            if state == "recording":
                if ev == "enter":
                    app.release()
                    state = "answering"
                continue
            if state == "answering":                        # barge in: the scheduler drops the old answer
                dropped = app.interrupt()
                print(f"  interrupted ({dropped / speaker.sr:.1f} s of audio dropped)", flush=True)
            app.press("hotkey" if ev == "down" else "enter")
            print("listening..." + (" (release to stop)" if ev == "down" else " (Enter to stop)"), flush=True)
            state = "recording"
        elif ev == "up" and state == "recording" and app.turn.get("source") == "hotkey":
            if time.perf_counter() - app.turn["t_press"] < 0.25:  # a tap, not a question
                app.cancel_tap()
                state = "idle"
                print("  (too short, ignored)", flush=True)
                continue
            app.release()
            state = "answering"
    if hotkey_proc:
        hotkey_proc.terminate()
    app.stop()
    speaker.stop()
    mic.stop()
    cap.cleanup()


def norm(s: str) -> str:
    return re.sub(r"[\s,$*`'\"]", "", s.lower())


_NUM_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
              "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
              "hundred thousand million point and").split()


def spoken_forms(a: str) -> list[str]:
    """'23' -> ['twenty three']; '0.0825' -> ['zero point zero eight two five']. Non-numbers: []."""
    if not re.fullmatch(r"\d+(\.\d+)?", a):
        return []
    from num2words import num2words
    w = num2words(float(a) if "." in a else int(a)).replace("-", " ").replace(",", "")
    return sorted({w, w.replace(" and ", " ")})


def _spoken_match(answer: str, form: str) -> bool:
    """Whole-word match that is not part of a bigger number ('three' does not match 'thirty three')."""
    words = [w.strip(".") for w in re.sub(r"[^a-z0-9.]+", " ", answer.lower().replace("-", " ")).split()]
    f = form.split()
    for i in range(len(words) - len(f) + 1):
        if words[i:i + len(f)] == f:
            before = words[i - 1] if i else ""
            after = words[i + len(f)] if i + len(f) < len(words) else ""
            if before not in _NUM_WORDS and after not in _NUM_WORDS:
                return True
    return False


def score_reading(answer: str, accept) -> bool:
    """Substring match as in M0, plus spelled-out numbers (the spoken-style prompt says 'three', not '3')."""
    if any(a in norm(answer or "") for a in accept):
        return True
    return any(_spoken_match(answer or "", f) for a in accept for f in spoken_forms(a))


def score_key_points(answer: str, key_points) -> bool:
    a = (answer or "").lower()
    return all(any(alt.lower() in a for alt in group) for group in key_points)


class ReplayPipeline:
    """Harness pipeline for push-to-talk replay (D7 end_turn, D15 turn complete, D17 scripted screen changes)."""

    signals_turn_complete = True

    def __init__(self, asst: Assistant, screens: dict, rows: list):
        self.asst, self.screens, self.rows = asst, screens, rows
        self.case = None
        self.frame_id = 1000

    def prepare(self, case: dict, speech_end_s: float):
        """Before a turn: show `before`, wait until it is read and a spare copy is ready."""
        self.case, self.speech_end_s = case, speech_end_s
        w = self.asst.worker
        before = case.get("before") or case.get("screen")
        if before and (w.snapshot is None or getattr(w.snapshot, "_screen", None) != before):
            self.frame_id += 1
            fid = self.frame_id
            w.submit_frame(fid, time.time(), self.screens[before])
            while w.snapshot is None or w.snapshot.frame_id != fid:
                time.sleep(0.02)
            w.snapshot._screen = before
        while w.snapshot.spare is None:
            time.sleep(0.02)
        self.frame_screens = {w.snapshot.frame_id: before}
        self.pre_changed = False
        c = case
        if c.get("after") and c["ref"] == "rec_start" and c["offset_s"] < 0:
            # change happens |offset| before recording starts: submit now, start the turn |offset| later
            self._change(c["after"])
            time.sleep(-c["offset_s"])
            self.pre_changed = True

    def _change(self, name: str):
        self.frame_id += 1
        self.frame_screens[self.frame_id] = name
        self.asst.worker.submit_frame(self.frame_id, time.time(), self.screens[name])

    def reset(self):
        self.audio, self.started_at, self.timer = [], None, None

    def start(self, emit_audio, mark):
        self.emit, self.mark = emit_audio, mark

    def on_audio(self, chunk):
        if self.started_at is None:                          # recording starts with the first chunk
            self.started_at = time.perf_counter()
            self.asst.worker.set_turn_active(True)
            c = self.case
            if c.get("after") and not self.pre_changed:   # change during the question
                at = c["offset_s"] if c["ref"] == "rec_start" else self.speech_end_s + c["offset_s"]
                self.timer = threading.Timer(max(0.0, at), self._change, args=(c["after"],))
                self.timer.start()
        self.audio.append(chunk)

    def end_turn(self):
        t_eos, t_eos_wall = time.perf_counter(), time.time()
        audio = np.concatenate(self.audio).astype(np.float32)

        def run():
            row = self.asst.answer_turn(audio, self.emit, self.mark, t_eos_wall, t_eos)
            self.asst.worker.set_turn_active(False)
            row["answered_screen"] = self.frame_screens.get(row.get("frame_id"))
            self.rows.append(row)
            self.mark("turn_complete")

        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        if self.timer:
            self.timer.cancel()


def replay(args):
    from PIL import Image
    from harness import harness as H

    sets = json.loads((QDIR / "sets.json").read_text())
    index = {q["id"]: q for q in json.loads((QDIR / "index.json").read_text())}
    screens = {p.stem: Image.open(p).convert("RGB") for p in M0_SCREENS.glob("*.png")}
    if args.replay == "contention":
        cases = json.loads((ROOT / "eval/contention.json").read_text())["cases"]
    else:
        cases = sets[args.replay]
    asst = Assistant(budget=args.budget, sandbox_profile=Path(args.sandbox_profile) if args.sandbox_profile else None,
                     swap_weights=getattr(args, "swap_weights", "on") == "on", **worker_opts(args))
    first = cases[0].get("before") or cases[0].get("screen")
    asst.start(warm_image=screens[first], first_frame=(1, time.time(), screens[first]))
    rows: list = []
    pipe = ReplayPipeline(asst, screens, rows)
    results = []
    for rep in range(args.repeats):
        for c in cases:
            qid = c.get("q") or c["id"]
            q = index[qid]
            pipe.prepare(c, q["speech_end_s"])
            t_wall0 = time.time()
            r = H.run_turn(pipe, q, timeout_s=60)
            row = rows[-1] if rows else {}
            target = c.get("after") or c.get("screen")
            ans = row.get("answer") or ""
            ok = None
            if "accept" in c:
                ok = score_reading(ans, c["accept"])
            elif "key_points" in c:
                ok = score_key_points(ans, c["key_points"])
            res = {"case": c["id"], "q": qid, "repeat": rep, "latency_ms": r["latency_ms"], "marks_ms": r["marks_ms"],
                   "turn_complete_ms": r["turn_complete_ms"], "audio_s": r["audio_s"],
                   "end_turn_late_ms": r["end_turn_late_ms"], "target_screen": target,
                   "stale": (row.get("answered_screen") != target) if target else None,
                   "script_correct": ok, **row, **asst.monitor.window(t_wall0, time.time())}
            results.append(res)
            print(json.dumps({k: res.get(k) for k in ("case", "latency_ms", "script_correct", "stale",
                                                     "wait_behind_read_ms", "read_in_flight_at_submit",
                                                     "fresh_decision", "read_paused", "answer")}),
                  flush=True)
    asst.stop()
    out = LOGS / f"replay-{args.replay}-{args.budget}-{getattr(args, 'policy', 'p2')}{getattr(args, 'freshness', 'B')}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary = summarize(results)
    summary.update(set=args.replay, budget=args.budget, policy=getattr(args, "policy", "p2"),
                   freshness=getattr(args, "freshness", "B"), startup=asst.startup, log=str(out),
                   reads_dropped=asst.worker.stats.get("reads_dropped"))
    runner = asst.worker.ane_runner
    summary.update(placement=getattr(args, "placement", "gpu"), ane_reads=asst.worker.stats.get("ane_reads"),
                   ane_to_gpu=asst.worker.stats.get("ane_to_gpu"), ane=runner.stats if runner else None)
    print("SUMMARY", json.dumps(summary))
    return summary


def summarize(results: list) -> dict:
    lat = sorted(r["latency_ms"] for r in results if r.get("latency_ms") is not None)
    p = lambda q: lat[min(len(lat) - 1, int(np.ceil(q * len(lat))) - 1)] if lat else None
    scored = [r for r in results if r.get("script_correct") is not None]
    return {
        "n": len(results), "answered": len(lat),
        "latency_median_ms": round(statistics.median(lat), 1) if lat else None, "latency_p95_ms": p(0.95),
        "script_correct": f"{sum(r['script_correct'] for r in scored)}/{len(scored)}" if scored else None,
        "stale": sum(1 for r in results if r.get("stale")),
        "waited_behind_read": sum(1 for r in results if r.get("read_in_flight_at_submit")),
        "wait_ms_max": max((r.get("wait_behind_read_ms") or 0) for r in results) if results else None,
        "read_paused": sum(1 for r in results if r.get("read_paused")),
        "waited_for_fresh": sum(1 for r in results if r.get("fresh_decision") == "read_first"),
        "mem_peak_gb": max((r.get("mem_peak_gb") or 0) for r in results) or None,
        "fan_max_rpm": max((r.get("fan_max_rpm") or 0) for r in results) if results else None,
        "cap_hits": sum(1 for r in results if r.get("cap_hit")),
    }


def add_worker_args(ap):
    ap.add_argument("--first-chunk", choices=["phrase", "sentence"], default="phrase",
                    help="start speaking on an early phrase of the first sentence, or wait for the whole sentence")
    ap.add_argument("--no-trim", action="store_true", help="keep Kokoro's built-in leading/trailing silence")
    ap.add_argument("--mlx-cache-gb", type=float, default=0.0,
                    help="cap on MLX's reusable buffer cache (default 0; -1 = MLX default, uncapped)")
    ap.add_argument("--policy", choices=["p0", "p1", "p2"], default="p2",
                    help="scheduler: p0 MVP baseline, p1 answer waits for the read, p2 pausable reads (default)")
    ap.add_argument("--freshness", choices=["A", "B", "C"], default="B",
                    help="p2 when a newer screen is still being read: A finish if close, B always fresh (default), C always fast")
    ap.add_argument("--fresh-ms", type=float, default=300.0, help="rule A: finish the read if it needs at most this long")
    ap.add_argument("--layers-per-piece", type=int, default=2, help="text layers per read piece")
    ap.add_argument("--min-read-interval", type=float, default=0.0, help="rate cap: seconds between read starts (0 = off)")
    ap.add_argument("--duty-cap", type=float, default=1.0, help="reads may use at most this share of GPU time per minute (1 = off)")
    ap.add_argument("--placement", choices=["gpu", "split"], default="gpu",
                    help="image encoder: gpu (default) or split = P4, quiet-time encodes on the Neural Engine (ane_vision.py)")
    ap.add_argument("--ane-root", default=None, help="folder of converted Core ML encoders (default models/vision-coreml)")
    ap.add_argument("--show-text", action="store_true",
                    help="print the transcript and answer in the terminal (the assistant can then read them on screen)")
    ap.add_argument("--hotkey", choices=["right_option", "right_command", "right_control", "fn", "none"],
                    default="right_option", help="hold this key in any app to talk (needs Input Monitoring permission)")
    ap.add_argument("--swap-weights", choices=["on", "off"], default="on",
                    help="keep only one of the image encoder / output layer in memory at a time (-0.33 GB, §3.9)")
    ap.add_argument("--history", choices=["on", "off", "auto"], default="auto",
                    help="conversation history in the context: auto = on live, off in replays (independent test questions)")


def worker_opts(args) -> dict:
    g = lambda k, d: getattr(args, k, d)
    return dict(first_chunk=g("first_chunk", "phrase"), trim_silence=not g("no_trim", False),
                mlx_cache_gb=None if g("mlx_cache_gb", 0.0) < 0 else g("mlx_cache_gb", 0.0),
                policy=g("policy", "p2"), freshness=g("freshness", "B"), fresh_ms=g("fresh_ms", 300.0),
                layers_per_piece=g("layers_per_piece", 2), min_read_interval_s=g("min_read_interval", 0.0),
                duty_cap=g("duty_cap", 1.0), placement=g("placement", "gpu"), ane_root=g("ane_root", None),
                history={"on": True, "off": False}.get(g("history", "auto"), not g("replay", None)))


def main():
    ap = argparse.ArgumentParser()
    add_worker_args(ap)
    ap.add_argument("--budget", type=int, default=1120, help="image token budget: 1120 (default, reads normal ~13 pt app text) or 560 (light mode, misreads small digits/IDs)")
    ap.add_argument("--replay", choices=["reading", "reasoning", "voice", "contention"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sandbox-profile", default="")
    ap.add_argument("--screen-only", action="store_true",
                    help="the previous app: answer from the current screen only, no memory of past screens")
    args = ap.parse_args()
    if args.replay:
        replay(args)                     # the timing harness still replays the current-screen app (GpuWorker)
    elif args.screen_only:
        live(args)
    else:
        live_recall(args)                # the shipped app: recall mode (ARCHITECTURE.md §5.9)


if __name__ == "__main__":
    main()
