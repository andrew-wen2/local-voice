#!/usr/bin/env python
"""menubar.py — turn the assistant on and off from the menu bar.

  .venv/bin/python menubar.py                  icon in the menu bar: Turn On / Turn Off / Quit
  .venv/bin/python menubar.py --budget 560     any other arguments are passed to app.py

On runs `app.py` (live mode) as a child process in its own process group. Off asks it to quit, then
ends the whole group (speech, TTS and encoder helpers too), so its memory is freed at once.

Off wipes everything the assistant saw: screens and what it remembers of them live only in that
process's RAM and nothing is written to disk, so ending the process is the wipe. Any screen memory
added to live mode later must keep to this (RAM only), or Off must delete it as well.

Talking is unchanged: hold Right Option in any app. Permissions (Screen Recording, Microphone,
Input Monitoring) belong to the app this is launched from, as with app.py.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"

# state -> (SF Symbol, menu status line)
STATES = {
    "off": ("waveform.slash", "Off"),
    "loading": ("hourglass", "Loading models…"),
    "on": ("waveform", "On: hold Right Option to talk"),
    "stopping": ("hourglass", "Turning off…"),
    "error": ("exclamationmark.triangle", "Stopped with an error (see logs/menubar-*.log)"),
    "no_screen_permission": ("exclamationmark.triangle", "Screen Recording permission is off"),
}


def state_from_line(line: str) -> Optional[str]:
    """Map a line app.py prints to a state, or None if the line doesn't change it."""
    s = line.strip()
    if s.startswith("starting"):
        return "loading"
    if s.startswith("ready in"):
        return "on"
    if s.startswith("Screen Recording permission is off"):
        return "no_screen_permission"
    return None


class Runner:
    """Owns the app.py child process. `on_state(state)` is called from a background thread."""

    def __init__(self, cmd: list[str], on_state: Callable[[str], None], log_dir: Path = LOGS,
                 quit_grace_s: float = 15.0, term_grace_s: float = 5.0):
        self.cmd, self.on_state, self.log_dir = cmd, on_state, log_dir
        self.quit_grace_s, self.term_grace_s = quit_grace_s, term_grace_s
        self.proc: Optional[subprocess.Popen] = None
        self.state = "off"
        self._lock = threading.Lock()
        self._stopping = False

    def _set(self, state: str):
        self.state = state
        self.on_state(state)

    def start(self):
        with self._lock:
            if self.proc and self.proc.poll() is None:
                return
            self.log_dir.mkdir(exist_ok=True)
            log = (self.log_dir / f"menubar-{time.strftime('%Y%m%d-%H%M%S')}.log").open("a")
            self._stopping = False
            self.proc = subprocess.Popen(self.cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                                         start_new_session=True)       # own group: Off ends every helper
            self._set("loading")
            threading.Thread(target=self._read, args=(self.proc, log), daemon=True).start()

    def _read(self, proc: subprocess.Popen, log):
        for line in proc.stdout:
            log.write(line)
            log.flush()
            st = state_from_line(line)
            if st and not self._stopping:
                self._set(st)
        code = proc.wait()
        log.close()
        _kill_group(proc.pid, signal.SIGKILL)              # leftovers, if any, after the leader exits
        if proc is self.proc and not self._stopping:       # exited on its own
            if self.state != "no_screen_permission":
                self._set("off" if code == 0 else "error")

    def stop(self, block: bool = False):
        """Quit app.py and everything it started. Returns at once unless block=True."""
        with self._lock:
            proc = self.proc
            if not proc or proc.poll() is not None:
                self._set("off")
                return
            self._stopping = True
            ready = self.state == "on"
            self._set("stopping")

        def run():
            if ready:                                      # clean quit: app.py reads "q" once it is ready
                try:
                    proc.stdin.write("q\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                _wait(proc, self.quit_grace_s)
            if proc.poll() is None:                        # still loading, or didn't quit in time
                _kill_group(proc.pid, signal.SIGTERM)
                _wait(proc, self.term_grace_s)
            _kill_group(proc.pid, signal.SIGKILL)
            proc.wait()
            self._set("off")

        if block:
            run()
        else:
            threading.Thread(target=run, daemon=True).start()


def _wait(proc: subprocess.Popen, timeout: float):
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        pass


def _kill_group(pgid: int, sig: int):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


# ---------------------------------------------------------------- menu bar (AppKit)

def main():
    import AppKit
    import objc
    from Foundation import NSObject
    from PyObjCTools import AppHelper

    class Controller(NSObject):
        def init(self):
            self = objc.super(Controller, self).init()
            bar = AppKit.NSStatusBar.systemStatusBar()
            self.item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
            menu = AppKit.NSMenu.alloc().init()
            menu.setAutoenablesItems_(False)
            self.status = menu.addItemWithTitle_action_keyEquivalent_("", None, "")
            self.status.setEnabled_(False)
            menu.addItem_(AppKit.NSMenuItem.separatorItem())
            self.toggle = menu.addItemWithTitle_action_keyEquivalent_("Turn On", "toggle:", "")
            self.toggle.setTarget_(self)
            menu.addItem_(AppKit.NSMenuItem.separatorItem())
            q = menu.addItemWithTitle_action_keyEquivalent_("Quit", "quit:", "q")
            q.setTarget_(self)
            self.item.setMenu_(menu)
            cmd = [sys.executable, "-u", str(ROOT / "app.py"), *sys.argv[1:]]
            self.runner = Runner(cmd, lambda st: AppHelper.callAfter(self.show_, st))
            self.show_("off")
            # Python only handles Ctrl-C/SIGTERM when it gets control; the run loop is native code, so wake it
            AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(1.0, self, "tick:", None, True)
            return self

        def tick_(self, _):
            pass

        def show_(self, state):
            symbol, text = STATES[state]
            img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, text)
            img.setTemplate_(True)                         # follows light/dark menu bar
            self.item.button().setImage_(img)
            self.item.button().setToolTip_(f"Local voice: {text}")
            self.status.setTitle_(text)
            running = state in ("loading", "on")
            self.toggle.setTitle_("Turn Off (wipes what it saw)" if running else "Turn On")
            self.toggle.setEnabled_(state != "stopping")

        def toggle_(self, _):
            if self.runner.state in ("loading", "on"):
                self.runner.stop()
            else:
                self.runner.start()

        def quit_(self, _):
            self.runner.stop(block=True)
            AppKit.NSApp.terminate_(None)

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)   # no Dock icon
    ctl = Controller.alloc().init()

    def on_signal(*_):                                     # Ctrl-C / kill in the launching terminal
        ctl.runner.stop(block=True)
        os._exit(0)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    AppHelper.runEventLoop(installInterrupt=False)


if __name__ == "__main__":
    main()
