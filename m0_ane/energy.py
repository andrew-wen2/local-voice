"""Energy per screen read: Gemma 4 vision encoder on the Neural Engine (Core ML) vs the GPU (MLX).

Runs each encoder in a loop for a fixed time while macmon samples power every 250 ms,
with idle periods in between. Energy per frame = (phase avg power - idle avg power) x time per frame.
Both sides compute the same 560-budget encoding of the same screenshot.

  python m0_ane/energy.py --budget 560 --seconds 20
"""
import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "m0_ane/out"
FIELDS = ("all_power", "cpu_power", "gpu_power", "ane_power", "gpu_ram_power", "ram_power", "sys_power")


class Power:
    def __init__(self, interval_ms=250):
        self.samples = []
        self.p = subprocess.Popen(["macmon", "pipe", "-i", str(interval_ms)], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            fans = d.get("fans") or [{}]
            self.samples.append((time.time(), {k: d.get(k) for k in FIELDS}, fans[0].get("rpm")))

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        out = {k: statistics.mean(v[1][k] for v in s if v[1].get(k) is not None) for k in FIELDS if any(v[1].get(k) is not None for v in s)}
        out["fan_rpm_max"] = max((v[2] or 0) for v in s) if s else None
        out["n"] = len(s)
        return out

    def stop(self):
        self.p.terminate()


def loop(fn, seconds):
    fn()  # warm
    n, t0 = 0, time.time()
    start = t0
    while time.time() - start < seconds:
        fn(); n += 1
    t1 = time.time()
    return t0, t1, (t1 - t0) / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=560)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--idle", type=float, default=10)
    a = ap.parse_args()

    # --- load both encoders first (loading is not measured) ---
    import coremltools as ct
    x = np.load(OUT / f"patches_{a.budget}.npy")
    ane = ct.models.MLModel(str(OUT / f"gemma4_vision_{a.budget}_ane.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.utils import prepare_inputs
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load(str(ROOT / "models/gemma-4-e4b-it-4bit"))
    processor.image_processor.max_soft_tokens = a.budget
    img = str(ROOT / "m0_screen/screens/editor.png")
    prompt = apply_chat_template(processor, model.config, "hi", num_images=1)
    inp = prepare_inputs(processor, images=[img], prompts=[prompt])
    pv, pos = inp["pixel_values"], inp.get("image_position_ids")
    gpu_fn = lambda: mx.eval(model.vision_tower(pv, pos))
    ane_fn = lambda: ane.predict({"patches": x})

    pw = Power()
    time.sleep(2)
    phases = {}
    t = time.time(); time.sleep(a.idle); phases["idle_1"] = pw.window(t + 1, time.time())
    t0, t1, per, n = loop(ane_fn, a.seconds); phases["ane"] = {**pw.window(t0 + 1, t1), "s_per_frame": per, "frames": n}
    t = time.time(); time.sleep(a.idle); phases["idle_2"] = pw.window(t + 2, time.time())
    t0, t1, per, n = loop(gpu_fn, a.seconds); phases["gpu"] = {**pw.window(t0 + 1, t1), "s_per_frame": per, "frames": n}
    t = time.time(); time.sleep(a.idle); phases["idle_3"] = pw.window(t + 2, time.time())
    pw.stop()

    idle = statistics.mean(phases[k]["all_power"] for k in ("idle_1", "idle_2", "idle_3"))
    for k in ("ane", "gpu"):
        p = phases[k]
        p["extra_power_w"] = p["all_power"] - idle
        p["energy_per_frame_j"] = p["extra_power_w"] * p["s_per_frame"]
    phases["idle_avg_w"] = idle
    (OUT / f"energy_{a.budget}.json").write_text(json.dumps(phases, indent=1))
    for k in ("idle_1", "ane", "idle_2", "gpu", "idle_3"):
        p = phases[k]
        print(f"{k:7s} total {p['all_power']:.2f} W  cpu {p.get('cpu_power',0):.2f}  gpu {p.get('gpu_power',0):.2f}  "
              f"ane {p.get('ane_power',0):.2f}  fan {p['fan_rpm_max']}"
              + (f"  | {p['s_per_frame']*1000:.0f} ms/frame, +{p['extra_power_w']:.2f} W over idle, {p['energy_per_frame_j']:.2f} J/frame" if k in ("ane", "gpu") else ""))


if __name__ == "__main__":
    main()
