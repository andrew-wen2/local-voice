"""Gemma 4 vision encoder on the Neural Engine (ARCHITECTURE.md §6.1, build step S2b).

The encoder is converted to Core ML in chunks of layers, one fixed-shape set per *bucket* (the
processor's output size H x W; the Neural Engine needs static shapes). Each chunk is one Core ML
call, so a background encode can stop between chunks. Default 1 layer per chunk: ~0.31 s per call
at 1120 and the same total as 4-layer chunks (4.91 vs 4.88 s, identical output), so an encode in
flight when the talk key is pressed is done within ~0.3 s.

  chunk 0:    patches [1, N, 768] -> patch embedding + position embedding + layers 0..L-1 -> x
  chunk i:    x [1, 768, 1, N] -> next L layers -> x
  last chunk: ... -> 3x3 average pooling * sqrt(hidden) -> tokens [1, N/9, 768]

The tokens are the vision tower's output; MLX then applies embed_vision and prefills the screen
turn on the GPU as usual (GemmaAdapter.encode_steps_ane).

  python ane_vision.py convert                    # whole-screen bucket at 1120 (1248 x 1968)
  python ane_vision.py convert --size 1296x1920 --layers-per-chunk 2
  python ane_vision.py list

Converting needs torch (traces the PyTorch wrapper from m0_ane/vision_coreml.py, checked against
Hugging Face's Gemma4VisionModel in M0). Running does not: the app keeps torch out (M3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models/vision-coreml"
PATCH, POOL, HIDDEN = 16, 3, 768


def bucket_dir(h: int, w: int, root: Path = OUT) -> Path:
    return Path(root) / f"{h}x{w}"


# ---------------------------------------------------------------- runtime (no torch)

def patchify(pixel_values: np.ndarray) -> np.ndarray:
    """[1, 3, H, W] in [0, 1] -> [1, N, 768], patches in raster order (x fastest), each flattened
    as (row, col, channel): the same layout as mlx_vlm's VisionPatchEmbedder._patchify."""
    b, c, h, w = pixel_values.shape
    p = pixel_values.reshape(b, c, h // PATCH, PATCH, w // PATCH, PATCH).transpose(0, 2, 4, 3, 5, 1)
    return np.ascontiguousarray(p.reshape(b, (h // PATCH) * (w // PATCH), PATCH * PATCH * c))


class AneVisionEncoder:
    """Loads the converted chunks for every bucket on disk and runs them on the Neural Engine.

    Core ML calls block; run encode_steps on a thread that may block (not the GPU worker's loop)."""

    def __init__(self, root: Path = OUT, compute_units: str = "CPU_AND_NE"):
        self.root, self.compute_units = Path(root), compute_units
        self.buckets: dict[tuple[int, int], list] = {}
        self.layers_per_chunk: dict[tuple[int, int], int] = {}
        self.load_s: dict[str, float] = {}

    def available(self) -> list[tuple[int, int]]:
        out = []
        for d in sorted(self.root.glob("*x*")):
            if (d / "meta.json").exists():
                m = json.loads((d / "meta.json").read_text())
                out.append((m["height"], m["width"]))
        return out

    def load(self, sizes: Optional[list[tuple[int, int]]] = None):
        import coremltools as ct
        cu = getattr(ct.ComputeUnit, self.compute_units)
        for h, w in sizes or self.available():
            d = bucket_dir(h, w, self.root)
            meta = json.loads((d / "meta.json").read_text())
            t0 = time.perf_counter()
            models = []
            for name in meta["chunks"]:
                # Loading an .mlpackage recompiles it every time (~140 s for 4 chunks at 1120); compile
                # once to .mlmodelc next to it and load that.
                compiled = d / name.replace(".mlpackage", ".mlmodelc")
                if not compiled.exists():
                    ct.models.utils.compile_model(str(d / name), str(compiled))
                models.append(ct.models.CompiledMLModel(str(compiled), compute_units=cu))
            self.buckets[(h, w)] = models
            self.layers_per_chunk[(h, w)] = meta["layers_per_chunk"]
            self.load_s[f"{h}x{w}"] = round(time.perf_counter() - t0, 1)
        return self

    def has(self, h: int, w: int) -> bool:
        return (h, w) in self.buckets

    def encode_steps(self, pixel_values: np.ndarray):
        """Generator: one Core ML call per chunk. After each chunk but the last it yields
        (chunks done, chunks, vision layers done, hidden state [1, 768, 1, N] fp16), so an encode cut
        short can continue on the GPU from that layer (GemmaAdapter.encode_steps(resume=...)).
        Returns tokens [1, T, 768] (fp32)."""
        _, _, h, w = pixel_values.shape
        return (yield from self.encode_patches_steps(patchify(pixel_values).astype(np.float16), h, w))

    def encode_patches_steps(self, patches: np.ndarray, h: int, w: int):
        """encode_steps from patches [1, N, 768] fp16 (patchify output) of an H x W image."""
        chunks, lpc = self.buckets[(h, w)], self.layers_per_chunk.get((h, w), 1)
        x = {"patches": patches}
        for i, m in enumerate(chunks):
            y = m.predict(x)
            if i == len(chunks) - 1:
                return np.asarray(y["tokens"], dtype=np.float32)
            x = {"x_in": y["x_out"]}
            yield i + 1, len(chunks), (i + 1) * lpc, y["x_out"]

    def encode(self, pixel_values: np.ndarray) -> np.ndarray:
        g = self.encode_steps(pixel_values)
        while True:
            try:
                next(g)
            except StopIteration as e:
                return e.value


class AneProcess:
    """AneVisionEncoder in a child process (`python ane_vision.py serve`), with the same interface.

    coremltools' predict holds the GIL for the whole call: measured 2026-11-06, other threads ran at
    4% speed during a chunk, and a GPU read took 6.3 s instead of 2.0 s while the Neural Engine
    encoded. A child process has its own GIL. Control messages are JSON lines on stdin/stdout; the
    arrays (patches in, hidden state after each chunk, tokens out) go through one shared-memory block.
    Use from one thread at a time."""

    def __init__(self, root: Path = OUT, compute_units: str = "CPU_AND_NE"):
        self.root, self.compute_units = Path(root), compute_units
        self.buckets: dict[tuple[int, int], dict] = {}
        self.proc = self.shm = None
        self.load_s: dict = {}

    def load(self):
        import subprocess
        from multiprocessing import shared_memory
        metas = []
        for d in sorted(self.root.glob("*x*")):
            if (d / "meta.json").exists():
                metas.append(json.loads((d / "meta.json").read_text()))
        if not metas:
            return self
        size = max(2 * m["patches"] * HIDDEN * 2 + m["tokens"] * HIDDEN * 4 for m in metas)
        self.shm = shared_memory.SharedMemory(create=True, size=size)
        self.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "ane_vision.py"), "serve", "--root", str(self.root),
             "--shm", self.shm.name, "--compute-units", self.compute_units],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        msg = self._recv()
        assert msg["ok"], msg
        for m in metas:
            self.buckets[(m["height"], m["width"])] = m
        self.load_s = msg["load_s"]
        return self

    def _send(self, **msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("ane_vision serve exited")
            if line.startswith("@@"):
                return json.loads(line[2:])

    def _view(self, offset: int, shape, dtype):
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return np.ndarray(shape, dtype=dtype, buffer=self.shm.buf[offset:offset + n])

    def has(self, h: int, w: int) -> bool:
        return (int(h), int(w)) in self.buckets

    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc else None

    def encode_steps(self, pixel_values: np.ndarray):
        _, _, h, w = pixel_values.shape
        m = self.buckets[(int(h), int(w))]
        n = m["patches"]
        self._view(0, (1, n, HIDDEN), np.float16)[...] = patchify(pixel_values).astype(np.float16)
        self._send(cmd="start", h=int(h), w=int(w))
        r = self._recv()
        assert r["ok"], r
        while True:
            self._send(cmd="step")
            r = self._recv()
            if not r["ok"]:
                raise RuntimeError(r.get("error"))
            if r["done"]:
                return self._view(2 * n * HIDDEN * 2, (1, m["tokens"], HIDDEN), np.float32).copy()
            x = self._view(n * HIDDEN * 2, (1, HIDDEN, 1, n), np.float16).copy()
            yield r["chunk"], r["chunks"], r["layers"], x

    def encode(self, pixel_values: np.ndarray) -> np.ndarray:
        return AneVisionEncoder.encode(self, pixel_values)

    def close(self):
        if self.proc is not None:
            try:
                self._send(cmd="stop")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None


def serve(root: Path, shm_name: str, compute_units: str):
    """Child side of AneProcess. Replies are '@@'-prefixed JSON lines on stdout."""
    sys.modules.setdefault("torch", None)         # coremltools treats torch as optional (M3)
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name, track=False)   # the parent owns and unlinks it
    out = sys.stdout

    def reply(**msg):
        out.write("@@" + json.dumps(msg) + "\n")
        out.flush()

    enc = AneVisionEncoder(root, compute_units).load()
    reply(ok=True, load_s=enc.load_s)
    gen, meta = None, None
    for line in sys.stdin:
        msg = json.loads(line)
        try:
            if msg["cmd"] == "stop":
                break
            if msg["cmd"] == "start":
                h, w = msg["h"], msg["w"]
                n = (h // PATCH) * (w // PATCH)
                meta = (n, len(enc.buckets[(h, w)]))
                patches = np.ndarray((1, n, HIDDEN), np.float16, buffer=shm.buf[:n * HIDDEN * 2]).copy()
                gen = enc.encode_patches_steps(patches, h, w)
                reply(ok=True)
            elif msg["cmd"] == "step":
                n, _ = meta
                try:
                    i, total, layers, x = next(gen)
                except StopIteration as e:
                    t = e.value
                    np.ndarray(t.shape, np.float32, buffer=shm.buf[2 * n * HIDDEN * 2:2 * n * HIDDEN * 2 + t.nbytes])[...] = t
                    reply(ok=True, done=True)
                    continue
                np.ndarray((1, HIDDEN, 1, n), np.float16, buffer=shm.buf[n * HIDDEN * 2:2 * n * HIDDEN * 2])[...] = x
                reply(ok=True, done=False, chunk=i, chunks=total, layers=layers)
        except Exception as e:                        # report, keep serving
            reply(ok=False, error=f"{type(e).__name__}: {e}")
    shm.close()


# ---------------------------------------------------------------- conversion (needs torch)

def _chunk_module(full, start: int, end: int, first: bool, last: bool):
    import torch.nn as nn
    import torch.nn.functional as F

    class Chunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.full = full

        def forward(self, inp):
            f = self.full
            if first:   # [1, N, 768] -> [1, 768, 1, N], patch embedding + position embedding
                x = inp.transpose(1, 2)[:, :, None, :]
                x = f.conv(f.input_proj, 2 * (x - 0.5)) + f.pos_c
            else:
                x = inp
            for L in f.layers[start:end]:
                h = f.attn(L, f.rms_c(x, L.input_layernorm.weight))
                x = x + f.rms_c(h, L.post_attention_layernorm.weight)
                m = L.mlp
                h = f.rms_c(x, L.pre_feedforward_layernorm.weight)
                h = f.clin_c(m.down_proj, F.gelu(f.clin_c(m.gate_proj, h), approximate="tanh") * f.clin_c(m.up_proj, h))
                x = x + f.rms_c(h, L.post_feedforward_layernorm.weight)
            if last:
                g = F.avg_pool2d(x.view(1, -1, f.gh, f.gw), POOL)
                return g.flatten(2).transpose(1, 2) * f.root
            return x

    return Chunk().eval()


def convert(h: int, w: int, layers_per_chunk: int = 1, q_block: int = 512, root: Path = OUT):
    os.environ["LV_ALLOW_TORCH"] = "1"
    import coremltools as ct
    import torch
    sys.path.insert(0, str(ROOT / "m0_ane"))
    from vision_coreml import FixedSizeVisionANE, load_hf_vision

    assert h % (PATCH * POOL) == 0 and w % (PATCH * POOL) == 0, (h, w)
    gw, gh = w // PATCH, h // PATCH
    n = gw * gh
    hf, cfg = load_hf_vision()
    full = FixedSizeVisionANE(hf, gw, gh, q_block).eval()
    n_layers = len(full.layers)
    d = bucket_dir(h, w, root)
    d.mkdir(parents=True, exist_ok=True)
    names, x = [], torch.rand(1, n, PATCH * PATCH * 3)
    for ci, start in enumerate(range(0, n_layers, layers_per_chunk)):
        end = min(n_layers, start + layers_per_chunk)
        first, last = start == 0, end == n_layers
        m = _chunk_module(full, start, end, first, last)
        t0 = time.time()
        with torch.no_grad():
            traced = torch.jit.trace(m, x)
            y = m(x)
        inp_name = "patches" if first else "x_in"
        out_name = "tokens" if last else "x_out"
        mlm = ct.convert(
            traced,
            inputs=[ct.TensorType(name=inp_name, shape=tuple(x.shape), dtype=np.float16)],
            outputs=[ct.TensorType(name=out_name, dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        name = f"chunk{ci}_layers{start}-{end - 1}.mlpackage"
        mlm.save(str(d / name))
        names.append(name)
        print(f"{name}: {tuple(x.shape)} -> {tuple(y.shape)} converted in {time.time() - t0:.0f}s", flush=True)
        x = y
    meta = {"height": h, "width": w, "grid": [gh, gw], "patches": n, "tokens": n // POOL ** 2,
            "layers_per_chunk": layers_per_chunk, "q_block": q_block, "chunks": names,
            "converted": time.strftime("%Y-%m-%d %H:%M")}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "list", "serve"])
    ap.add_argument("--size", default="1248x1968", help="processor output HxW (whole screen at 1120: 1248x1968)")
    ap.add_argument("--layers-per-chunk", type=int, default=1)
    ap.add_argument("--q-block", type=int, default=512)
    ap.add_argument("--root", default=str(OUT), help="output folder (one subfolder per bucket)")
    ap.add_argument("--shm", default="", help="serve: shared-memory block name (set by AneProcess)")
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(Path(a.root), a.shm, a.compute_units)
    elif a.cmd == "convert":
        h, w = (int(v) for v in a.size.split("x"))
        convert(h, w, a.layers_per_chunk, a.q_block, Path(a.root))
    else:
        for h, w in AneVisionEncoder(Path(a.root)).available():
            print(f"{h}x{w}", json.loads((bucket_dir(h, w, Path(a.root)) / "meta.json").read_text()))


if __name__ == "__main__":
    main()
