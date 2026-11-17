#!/usr/bin/env python
"""app.py — offline, screen-aware voice assistant (MVP, naive baseline).

  python app.py                      live, with memory of past screens (recall mode, ARCHITECTURE.md §5.9): hold Right
                                     Option in any app to talk (release = stop, press while it speaks = interrupt);
                                     Enter in the terminal also works (talk / stop / interrupt), q = quit
  python app.py --screen-only        the previous app: the current screen only, no memory of past screens

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
from engine import AnswerJob, GemmaAdapter, GpuWorker, make_tts  # noqa: E402

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
WARM_WAV = ROOT / "assets/warmup.wav"         # a spoken question for warming up speech recognition


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
        for k in ("mem_app", "mem_asr", "mem_tts"):
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
        """worker_opts go to GpuWorker: trim_silence, mlx_cache_gb, policy, freshness,
        fresh_ms, layers_per_piece, min_read_interval_s, duty_cap."""
        self.budget = budget
        self.worker_opts = worker_opts
        LOGS.mkdir(exist_ok=True)
        self.asr = make_asr(log_dir=LOGS)
        self.sandbox_profile = sandbox_profile
        self.tts = make_tts(sandbox_profile=sandbox_profile, log_dir=LOGS)
        self.swap_weights = swap_weights
        self.worker: Optional[GpuWorker] = None
        self.monitor = Monitor(lambda: {"app": os.getpid(), "asr": self.asr.pid(),
                                        "tts": self.tts.pid()})
        self.startup: dict = {}
        self.ready = False

    def start(self, warm_image, first_frame=None):
        """Readiness gate (D8): ASR healthy + warm, models loaded + warm, first screen read done."""
        t_all = time.perf_counter()
        self.monitor.start()
        self.startup["asr_start_s"] = round(self.asr.start(sandbox_profile=self.sandbox_profile), 2)
        t0 = time.perf_counter()
        self.asr.transcribe(WARM_WAV.read_bytes())
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
        row.update({k: r.get(k) for k in ("answer", "stop", "cap_hit", "cancelled", "sentences_spoken", "chunks_spoken", "first_chunk_text", "frame_id",
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
                from audio_io import first_audible
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
    ui = getattr(args, "ui_events", False)
    # Under the menu bar app, its caption panel is left out of every screen read: the assistant must not see (and
    # remember) its own answers. The menu bar is this process's parent.
    cap, grab_small = scr.open_capture(exclude_pids=(os.getppid(),) if ui else ())
    speaker, mic = Speaker().start(), Mic().start()
    o = worker_opts(args)
    ui_lock = threading.Lock()

    def on_ui(ev):                                       # one line per event; menubar.py reads them (not logged)
        with ui_lock:
            print("@ui " + json.dumps(ev), flush=True)

    def on_row(row):
        status = "error: " + row["error"] if row.get("error") else "answered"
        print(f"  {status}, first audio {row['first_audio_ms']} ms after you stopped", flush=True)

    app = RecallApp(grab=cap.grab, grab_small=grab_small, speaker=speaker, mic=mic, budget=args.budget,
                    swap_weights=getattr(args, "swap_weights", "on") == "on",
                    freshness="C" if o["freshness"] == "C" else "B",
                    trim_silence=o["trim_silence"], mlx_cache_gb=o["mlx_cache_gb"],
                    log_path=LOGS / f"turns-recall-{time.strftime('%Y%m%d-%H%M%S')}.jsonl",
                    show_text=getattr(args, "show_text", False), on_row=on_row, on_ui=on_ui if ui else None)
    startup = app.start()
    if ui:
        on_ui({"ev": "ready", "captions": os.getppid() in getattr(cap, "excluded", set()),
               "dirs": app.data_dirs() + [str(d) for d in [getattr(cap, "dir", None)] if d]})
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


def add_worker_args(ap):
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
    return dict(trim_silence=not g("no_trim", False),
                mlx_cache_gb=None if g("mlx_cache_gb", 0.0) < 0 else g("mlx_cache_gb", 0.0),
                policy=g("policy", "p2"), freshness=g("freshness", "B"), fresh_ms=g("fresh_ms", 300.0),
                layers_per_piece=g("layers_per_piece", 2), min_read_interval_s=g("min_read_interval", 0.0),
                duty_cap=g("duty_cap", 1.0),
                history={"on": True, "off": False}.get(g("history", "auto"), not g("replay", None)))


def main():
    ap = argparse.ArgumentParser()
    add_worker_args(ap)
    ap.add_argument("--budget", type=int, default=1120, help="image token budget: 1120 (default, reads normal ~13 pt app text) or 560 (light mode, misreads small digits/IDs)")
    ap.add_argument("--sandbox-profile", default="")
    ap.add_argument("--ui-events", action="store_true",
                    help="print caption events as '@ui {json}' lines and hide the parent app's windows from screen reads "
                         "(used by menubar.py)")
    ap.add_argument("--screen-only", action="store_true",
                    help="the previous app: answer from the current screen only, no memory of past screens")
    args = ap.parse_args()
    if args.screen_only:
        live(args)
    else:
        live_recall(args)                # the shipped app: recall mode (ARCHITECTURE.md §5.9)


if __name__ == "__main__":
    main()
