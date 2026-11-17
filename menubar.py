#!/usr/bin/env python
"""menubar.py — the assistant's small frontend: a menu bar icon and a caption panel.

  .venv/bin/python menubar.py                  icon in the menu bar: Turn On / Turn Off / Captions / Quit
  .venv/bin/python menubar.py --budget 560     any other arguments are passed to app.py

On runs `app.py --ui-events` (recall mode) as a child process in its own process group. Off asks it to quit, then
ends the whole group (speech, TTS and encoder helpers too), so its memory is freed at once.

Captions: while you hold the talk key, a panel at the bottom of the main display shows what it hears, then the answer,
each phrase as it starts playing, and fades out. It is for watching (and for a silent screen recording of a demo);
the answer is still spoken. The panel is left out of the assistant's own screen reads (app.py excludes this process
from its ScreenCaptureKit filter), so it never sees or remembers its own answers. If that exclusion can't be set up
(e.g. the `screencapture` fallback), captions stay off.

Off wipes what the assistant saw: kept screens and stored pieces live in owner-only temp folders that app.py deletes
on a clean quit; app.py also reports them (`dirs` events) and this process deletes them after the child ends, so a
hard kill can't leave them behind. The turn log (logs/turns-recall-*.jsonl: questions, answers, timings) is kept.

Caption events never reach the menu bar log file (they hold your questions and the answers).

Talking is unchanged: hold Right Option in any app. Permissions (Screen Recording, Microphone, Input Monitoring)
belong to the app this is launched from, as with app.py.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
DATA_PREFIXES = ("lv-screens-", "lv-pieces-", "localvoice-cap-")    # app.py's temp folders (screen_store, recall_backend, screen)

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


def ui_event(line: str) -> Optional[dict]:
    """A caption event line ('@ui {json}') -> its dict, or None for any other line."""
    if not line.startswith("@ui "):
        return None
    try:
        ev = json.loads(line[4:])
    except ValueError:
        return None
    return ev if isinstance(ev, dict) else None


def safe_data_dir(path: str) -> Optional[Path]:
    """`path` if it is one of app.py's own temp folders (directly in the temp dir, with a known prefix), else None.
    Paths come from the child's output, so nothing else is ever deleted."""
    try:
        p = Path(os.path.realpath(path))
    except (OSError, ValueError):
        return None
    tmp = Path(os.path.realpath(tempfile.gettempdir()))
    if p.parent != tmp or not p.name.startswith(DATA_PREFIXES):
        return None
    return p


def wipe(paths) -> list[str]:
    """Delete app.py's temp folders among `paths`. Returns those deleted."""
    gone = []
    for path in paths:
        p = safe_data_dir(path)
        if p is not None and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            gone.append(str(p))
    return gone


class Runner:
    """Owns the app.py child process. `on_state(state)` and `on_ui(event)` are called from a background thread."""

    def __init__(self, cmd: list[str], on_state: Callable[[str], None], log_dir: Path = LOGS,
                 quit_grace_s: float = 15.0, term_grace_s: float = 5.0,
                 on_ui: Optional[Callable[[dict], None]] = None):
        self.cmd, self.on_state, self.on_ui, self.log_dir = cmd, on_state, on_ui, log_dir
        self.quit_grace_s, self.term_grace_s = quit_grace_s, term_grace_s
        self.proc: Optional[subprocess.Popen] = None
        self.state = "off"
        self.data_dirs: set[str] = set()
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
            self.data_dirs = set()
            self.proc = subprocess.Popen(self.cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                                         start_new_session=True)       # own group: Off ends every helper
            self._set("loading")
            threading.Thread(target=self._read, args=(self.proc, log), daemon=True).start()

    def _read(self, proc: subprocess.Popen, log):
        for line in proc.stdout:
            ev = ui_event(line)
            if ev is not None:                             # caption events: never logged
                if ev.get("ev") in ("dirs", "ready"):
                    self.data_dirs.update(p for p in (ev.get("paths") or ev.get("dirs") or []) if isinstance(p, str))
                if self.on_ui is not None and not self._stopping:
                    self.on_ui(ev)
                continue
            log.write(line)
            log.flush()
            st = state_from_line(line)
            if st and not self._stopping:
                self._set(st)
        code = proc.wait()
        log.close()
        _kill_group(proc.pid, signal.SIGKILL)              # leftovers, if any, after the leader exits
        wipe(self.data_dirs)
        if proc is self.proc and not self._stopping:       # exited on its own
            if self.state != "no_screen_permission":
                self._set("off" if code == 0 else "error")

    def stop(self, block: bool = False):
        """Quit app.py and everything it started, and delete what it saw. Returns at once unless block=True."""
        with self._lock:
            proc = self.proc
            if not proc or proc.poll() is not None:
                wipe(self.data_dirs)
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
            wipe(self.data_dirs)
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


class Captions:
    """What the caption panel shows, from the event stream (no AppKit, so it is testable).
    Returns from `apply` what to do: "show", "hide" (now) or "linger" (hide after `linger_s()`), or None."""

    def __init__(self):
        self.head = self.question = self.answer = ""

    def apply(self, ev: dict) -> Optional[str]:
        kind = ev.get("ev")
        if kind == "listening":
            self.head, self.question, self.answer = "Listening…", "", ""
            return "show"
        if kind == "heard":
            self.question = (ev.get("text") or "").strip()
            return "show"
        if kind == "thinking":
            self.head = "Thinking…"
            return "show"
        if kind == "say":
            self.head = ""
            self.answer = (self.answer + " " + (ev.get("text") or "").strip()).strip()
            return "show"
        if kind == "done":
            self.head = ""
            if not self.answer:
                err = ev.get("error") or ""
                self.answer = "Didn't catch that." if "empty transcript" in err else ("Something went wrong." if err else "")
            return "linger" if self.answer else "hide"
        if kind == "idle":
            return "hide"
        return None

    def linger_s(self) -> float:
        """Long enough to finish reading after the audio ends."""
        return min(12.0, 3.0 + 0.25 * len(self.answer.split()))


# ---------------------------------------------------------------- menu bar and caption panel (AppKit)

def main():
    import AppKit
    import objc
    from Foundation import NSObject
    from PyObjCTools import AppHelper

    W, PAD, GAP = 560.0, 16.0, 6.0

    def label(size, weight, color):
        f = AppKit.NSTextField.wrappingLabelWithString_("")
        f.setFont_(AppKit.NSFont.systemFontOfSize_weight_(size, weight))
        f.setTextColor_(color)
        f.setSelectable_(False)
        return f

    class Panel:
        """A borderless, click-through, non-activating panel at the bottom centre of the main display."""

        def __init__(self):
            style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
            p = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                AppKit.NSMakeRect(0, 0, W, 80), style, AppKit.NSBackingStoreBuffered, False)
            p.setLevel_(AppKit.NSStatusWindowLevel)
            p.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                     | AppKit.NSWindowCollectionBehaviorStationary
                                     | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary)
            p.setIgnoresMouseEvents_(True)
            p.setOpaque_(False)
            p.setBackgroundColor_(AppKit.NSColor.clearColor())
            p.setHasShadow_(True)
            p.setHidesOnDeactivate_(False)
            p.setAppearance_(AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameVibrantDark))  # subtitle look
            fx = AppKit.NSVisualEffectView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, W, 80))
            fx.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
            fx.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
            fx.setState_(AppKit.NSVisualEffectStateActive)
            fx.setWantsLayer_(True)
            fx.layer().setCornerRadius_(16.0)
            fx.layer().setMasksToBounds_(True)
            p.setContentView_(fx)
            self.head = label(12, AppKit.NSFontWeightSemibold, AppKit.NSColor.secondaryLabelColor())
            self.q = label(14, AppKit.NSFontWeightRegular, AppKit.NSColor.secondaryLabelColor())
            self.a = label(19, AppKit.NSFontWeightMedium, AppKit.NSColor.labelColor())
            for f in (self.head, self.q, self.a):
                fx.addSubview_(f)
            self.win, self.fx = p, fx
            self.gen = 0                                   # bumps on every change; stale hide timers do nothing

        def _height(self, f, w):
            return f.cell().cellSizeForBounds_(AppKit.NSMakeRect(0, 0, w, 10000)).height

        def show(self, c: Captions):
            self.gen += 1
            inner = W - 2 * PAD
            q = f"“{c.question}”" if c.question else ""
            rows = [(f, t) for f, t in ((self.head, c.head), (self.q, q), (self.a, c.answer)) if t]
            for f, t in ((self.head, c.head), (self.q, q), (self.a, c.answer)):
                f.setStringValue_(t)
                f.setHidden_(not t)
            heights = [self._height(f, inner) for f, _ in rows]
            total = 2 * PAD + sum(heights) + GAP * max(0, len(rows) - 1)
            y = total - PAD
            for (f, _), h in zip(rows, heights):           # top to bottom (AppKit's y grows upwards)
                y -= h
                f.setFrame_(AppKit.NSMakeRect(PAD, y, inner, h))
                y -= GAP
            screen = AppKit.NSScreen.screens()[0]          # the display with the menu bar = the one the assistant reads
            vf = screen.visibleFrame()
            x = vf.origin.x + (vf.size.width - W) / 2
            self.win.setFrame_display_(AppKit.NSMakeRect(x, vf.origin.y + 72, W, total), True)
            self.fx.setFrame_(AppKit.NSMakeRect(0, 0, W, total))
            self.win.invalidateShadow()
            if not self.win.isVisible() or self.win.alphaValue() < 1:
                self.win.setAlphaValue_(1.0)
                self.win.orderFrontRegardless()

        def hide(self, after: float, target, selector):
            self.gen += 1
            AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                after, target, selector, self.gen, False)

        def fade_out(self, gen):
            if gen != self.gen or not self.win.isVisible():
                return
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(0.3)
            self.win.animator().setAlphaValue_(0.0)
            AppKit.NSAnimationContext.endGrouping()

    class Controller(NSObject):
        def init(self):
            self = objc.super(Controller, self).init()
            bar = AppKit.NSStatusBar.systemStatusBar()
            self.item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
            menu = AppKit.NSMenu.alloc().init()
            menu.setAutoenablesItems_(False)
            menu.setDelegate_(self)
            self.status = menu.addItemWithTitle_action_keyEquivalent_("", None, "")
            self.status.setEnabled_(False)
            menu.addItem_(AppKit.NSMenuItem.separatorItem())
            self.toggle = menu.addItemWithTitle_action_keyEquivalent_("Turn On", "toggle:", "")
            self.toggle.setTarget_(self)
            self.cap_item = menu.addItemWithTitle_action_keyEquivalent_("Show Captions", "toggleCaptions:", "")
            self.cap_item.setTarget_(self)
            menu.addItem_(AppKit.NSMenuItem.separatorItem())
            q = menu.addItemWithTitle_action_keyEquivalent_("Quit", "quit:", "q")
            q.setTarget_(self)
            self.item.setMenu_(menu)
            self.want_captions = True                      # the user's choice
            self.captions_ok = False                       # app.py confirmed the panel is hidden from its screen reads
            self.on_since: Optional[float] = None
            self.panel, self.cap = Panel(), Captions()
            cmd = [sys.executable, "-u", str(ROOT / "app.py"), "--ui-events", *sys.argv[1:]]
            self.runner = Runner(cmd, lambda st: AppHelper.callAfter(self.show_, st),
                                 on_ui=lambda ev: AppHelper.callAfter(self.ui_, ev))
            self.show_("off")
            # Python only handles Ctrl-C/SIGTERM when it gets control; the run loop is native code, so wake it
            AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(1.0, self, "tick:", None, True)
            return self

        def tick_(self, _):
            pass

        def status_text(self, state):
            text = STATES[state][1]
            if state == "on" and self.on_since is not None:
                mins = int((time.time() - self.on_since) // 60)
                span = "the last hour" if mins >= 60 else f"the last {max(mins, 1)} min"
                text = f"On, remembering {span}: hold Right Option to talk"
            return text

        def menuWillOpen_(self, _):
            self.status.setTitle_(self.status_text(self.runner.state))
            if self.runner.state == "on" and not self.captions_ok:
                self.cap_item.setTitle_("Captions unavailable (couldn't hide them from screen reads)")
            else:
                self.cap_item.setTitle_("Show Captions")

        def show_(self, state):
            symbol, text = STATES[state]
            if state == "on" and self.on_since is None:
                self.on_since = time.time()
            elif state != "on":
                self.on_since = None
            if state in ("off", "stopping", "error", "no_screen_permission"):
                self.captions_ok = False
                self.panel.fade_out(self.panel.gen)
            img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, text)
            img.setTemplate_(True)                         # follows light/dark menu bar
            self.item.button().setImage_(img)
            self.item.button().setToolTip_(f"Local voice: {text}")
            self.status.setTitle_(self.status_text(state))
            running = state in ("loading", "on")
            self.toggle.setTitle_("Turn Off (wipes what it saw)" if running else "Turn On")
            self.toggle.setEnabled_(state != "stopping")
            self.cap_item.setState_(AppKit.NSControlStateValueOn if self.want_captions else AppKit.NSControlStateValueOff)
            self.cap_item.setEnabled_(not (state == "on" and not self.captions_ok))

        def ui_(self, ev):
            if ev.get("ev") == "ready":
                self.captions_ok = bool(ev.get("captions"))
                self.show_(self.runner.state)
                return
            what = self.cap.apply(ev)
            if not (self.want_captions and self.captions_ok) or what is None:
                return
            if what == "show":
                self.panel.show(self.cap)
            elif what == "linger":
                self.panel.show(self.cap)
                self.panel.hide(self.cap.linger_s(), self, "fadeOut:")
            elif what == "hide":
                self.panel.hide(0.0, self, "fadeOut:")

        def fadeOut_(self, timer):
            self.panel.fade_out(timer.userInfo())

        def toggle_(self, _):
            if self.runner.state in ("loading", "on"):
                self.runner.stop()
            else:
                self.runner.start()

        def toggleCaptions_(self, _):
            self.want_captions = not self.want_captions
            if not self.want_captions:
                self.panel.fade_out(self.panel.gen)
            self.show_(self.runner.state)

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
