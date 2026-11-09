"""menubar.Runner against a fake app.py: states follow its output, and Off ends the whole process group."""
import os
import sys
import textwrap
import time

import pytest

from menubar import Runner, state_from_line

FAKE_APP = textwrap.dedent("""
    import subprocess, sys, time
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])  # like the ASR/TTS helpers
    print("helper", helper.pid, flush=True)
    print("starting (first launch can take a few minutes)...", flush=True)
    time.sleep(float(sys.argv[1]))
    print("ready in 0.1 s  {}", flush=True)
    if sys.argv[2] == "ignore_q":
        time.sleep(600)
    for line in sys.stdin:
        if line.strip() == "q":
            break
    helper.terminate()
""")


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for(cond, timeout=10.0):
    t = time.time() + timeout
    while time.time() < t:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def run(tmp_path):
    script = tmp_path / "fake_app.py"
    script.write_text(FAKE_APP)
    runners = []

    def make(load_s="0", mode="quits"):
        states = []
        r = Runner([sys.executable, "-u", str(script), load_s, mode], states.append, log_dir=tmp_path / "logs",
                   quit_grace_s=1.0, term_grace_s=1.0)
        runners.append(r)
        return r, states
    yield make
    for r in runners:
        r.stop(block=True)


def helper_pid(tmp_path):
    for log in (tmp_path / "logs").glob("menubar-*.log"):
        for line in log.read_text().splitlines():
            if line.startswith("helper "):
                return int(line.split()[1])


def test_state_from_line():
    assert state_from_line("starting (first launch can take a few minutes)...") == "loading"
    assert state_from_line("ready in 12.3 s  {...}") == "on"
    assert state_from_line("Screen Recording permission is off for this terminal app.") == "no_screen_permission"
    assert state_from_line("  answered, first audio 480 ms after you stopped") is None


def test_on_then_off_quits_cleanly(run, tmp_path):
    r, states = run()
    r.start()
    assert wait_for(lambda: r.state == "on")
    r.stop(block=True)
    assert r.proc.poll() is not None
    assert states[-1] == "off" and "stopping" in states


@pytest.mark.parametrize("load_s,mode", [("600", "quits"), ("0", "ignore_q")])
def test_off_ends_every_process_even_if_app_does_not_quit(run, tmp_path, load_s, mode):
    """Off while still loading, or when app.py ignores "q": the group is killed, helpers included."""
    r, states = run(load_s, mode)
    r.start()
    assert wait_for(lambda: helper_pid(tmp_path) is not None)
    helper = helper_pid(tmp_path)
    if mode == "ignore_q":
        assert wait_for(lambda: r.state == "on")
    r.stop(block=True)
    assert r.proc.poll() is not None
    assert wait_for(lambda: not alive(helper)), "helper process survived Off"
    assert states[-1] == "off"


def test_app_exiting_on_its_own_with_error(run, tmp_path):
    script = tmp_path / "fails.py"
    script.write_text("import sys; print('starting', flush=True); sys.exit(3)")
    states = []
    r = Runner([sys.executable, "-u", str(script)], states.append, log_dir=tmp_path / "logs")
    r.start()
    assert wait_for(lambda: r.state == "error")
