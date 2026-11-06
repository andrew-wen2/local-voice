"""Fan soak test: how much sustained background work before the fans turn on?

Phases (560-budget vision encoder, same frame each time):
  ane      Neural Engine, back to back
  gpu      GPU (MLX), back to back
  gpu_d30  GPU at 30% duty cycle (encode, then sleep ~2.3x the encode time)
Each phase runs until the fans turn on (rpm > 0) or --max-min minutes pass, then cools down
until the fans are off and the GPU is within 3 C of its starting temperature (max --cool-min).
Logs every macmon sample to out/soak.jsonl; prints a summary per phase.

  python m0_ane/soak.py --max-min 8
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


class Mon:
    def __init__(self, log):
        self.last = None
        self.log = log.open("a")
        self.phase = "setup"
        self.p = subprocess.Popen(["macmon", "pipe", "-i", "1000"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = dict(t=time.time(), phase=self.phase, rpm=(d.get("fans") or [{}])[0].get("rpm", 0),
                     all_w=d.get("all_power"), gpu_w=d.get("gpu_power"), ane_w=d.get("ane_power"),
                     cpu_w=d.get("cpu_power"), cpu_c=d["temp"]["cpu_temp_avg"], gpu_c=d["temp"]["gpu_temp_avg"])
            self.last = s
            self.log.write(json.dumps(s) + "\n"); self.log.flush()

    def wait_first(self):
        while self.last is None:
            time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-min", type=float, default=8)
    ap.add_argument("--cool-min", type=float, default=10)
    ap.add_argument("--phases", default="ane,gpu,gpu_d30")
    a = ap.parse_args()

    import coremltools as ct
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    x = np.load(OUT / "patches_560.npy")
    ane = ct.models.MLModel(str(OUT / "gemma4_vision_560_ane.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
    model, processor = load(str(ROOT / "models/gemma-4-e4b-it-4bit"))
    processor.image_processor.max_soft_tokens = 560
    inp = prepare_inputs(processor, images=[str(ROOT / "m0_screen/screens/editor.png")],
                         prompts=[apply_chat_template(processor, model.config, "hi", num_images=1)])
    pv, pos = inp["pixel_values"], inp.get("image_position_ids")

    def gpu():
        mx.eval(model.vision_tower(pv, pos))

    def ane_run():
        ane.predict({"patches": x})

    def idle():
        time.sleep(1.0)

    work = {"ane": (ane_run, 1.0), "ane_d30": (ane_run, 0.3), "gpu": (gpu, 1.0), "gpu_d30": (gpu, 0.3), "idle": (idle, 1.0)}
    mon = Mon(OUT / "soak.jsonl")
    mon.wait_first()

    def cool(base_gpu_c):
        mon.phase = "cool"
        t0 = time.time()
        while time.time() - t0 < a.cool_min * 60:
            s = mon.last
            if s["rpm"] == 0 and (base_gpu_c is None or s["gpu_c"] <= base_gpu_c + 3):
                return round(time.time() - t0)
            time.sleep(5)
        return None

    print("waiting for fans off / baseline...", flush=True)
    cool(None)
    time.sleep(20)
    summary = []
    for name in a.phases.split(","):
        fn, duty = work[name]
        fn()  # warm
        base = mon.last
        mon.phase = name
        t0 = time.time(); frames = 0; fan_on = None
        while time.time() - t0 < a.max_min * 60:
            s0 = time.perf_counter(); fn(); dt = time.perf_counter() - s0; frames += 1
            if duty < 1:
                time.sleep(dt * (1 / duty - 1))
            if mon.last["rpm"] > 0:
                fan_on = time.time() - t0
                break
        dur = time.time() - t0
        rows = [json.loads(l) for l in (OUT / "soak.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r["phase"] == name and r["t"] >= t0]
        res = dict(phase=name, duty=duty, seconds=round(dur), frames=frames,
                   fans_on_after_s=None if fan_on is None else round(fan_on),
                   avg_all_w=round(statistics.mean(r["all_w"] for r in rows), 2),
                   start_gpu_c=round(base["gpu_c"], 1), end_gpu_c=round(rows[-1]["gpu_c"], 1),
                   end_cpu_c=round(rows[-1]["cpu_c"], 1), max_rpm=max(r["rpm"] for r in rows))
        res["cooldown_s"] = cool(base["gpu_c"])
        summary.append(res)
        print(json.dumps(res), flush=True)
        time.sleep(20)
    mon.p.terminate()
    (OUT / "soak_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
