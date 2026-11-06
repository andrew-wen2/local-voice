"""M0 test 3: Gemma 4 E4B vision encoder -> Core ML, run on the Neural Engine.

One fixed-shape Core ML model per image size (the ANE needs static shapes).
Position embeddings, 2-D RoPE tables and the 3x3 pooling are baked in for that
size, so the model input is just the image patches.

  python m0_ane/vision_coreml.py convert --budget 280     # build + check vs PyTorch
  python m0_ane/vision_coreml.py plan    --budget 280     # which ops land on ANE/GPU/CPU
  python m0_ane/vision_coreml.py bench   --budget 280     # latency per compute-unit setting

Reference = Hugging Face transformers' Gemma4VisionModel (fp32, CPU) on the same
patches. Weights come from the MLX checkpoint already on disk (the vision tower
is stored unquantized there).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models/gemma-4-e4b-it-4bit"
OUT = ROOT / "m0_ane/out"
PATCH, K = 16, 3

# image-token budget -> (width_px, height_px) used in the M0 screen test (3:2-ish screenshots).
# Each side is a multiple of 48 px so the patch grid divides by the 3x3 pooling.
SIZES = {
    140: (672, 432),
    280: (960, 624),
    560: (1392, 912),
    1120: (1968, 1248),
}


def load_hf_vision():
    from safetensors import safe_open
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    vc = json.loads((CKPT / "config.json").read_text())["vision_config"]
    vc["rope_parameters"] = {"rope_type": "axial", "rope_theta": vc["rope_parameters"]["rope_theta"]}
    cfg = Gemma4VisionConfig(**{k: v for k, v in vc.items() if k not in ("architectures", "_name_or_path")})
    cfg._attn_implementation = "eager"
    model = Gemma4VisionModel(cfg).eval()
    sd = {}
    with safe_open(str(CKPT / "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("vision_tower."):
                sd[k[len("vision_tower."):]] = f.get_tensor(k).float()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    return model.float(), cfg


def grid(budget):
    w, h = SIZES[budget]
    return w // PATCH, h // PATCH  # patches across, down


def position_ids(gw, gh):
    """Raster order, x fastest: (x, y) per patch."""
    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    return torch.from_numpy(np.stack([xs.ravel(), ys.ravel()], -1)).long()[None]


def patchify(img):
    """img: float32 [H, W, 3] in [0, 1] -> [1, N, 768] patches in raster order, channel-last per pixel."""
    h, w, _ = img.shape
    p = img.reshape(h // PATCH, PATCH, w // PATCH, PATCH, 3).transpose(0, 2, 1, 3, 4)
    return torch.from_numpy(np.ascontiguousarray(p.reshape(1, -1, PATCH * PATCH * 3)))


class FixedSizeVision(nn.Module):
    """Gemma 4 vision tower specialised to one patch grid, ANE-friendly ops only.

    Same math as Gemma4VisionModel with no padding: patch embed + 2-D position
    embedding, 16 bidirectional attention layers with axial RoPE, 3x3 average
    pooling, * sqrt(hidden). Attention is computed in query blocks so no
    N x N score tensor larger than block x N exists.
    """

    def __init__(self, hf, gw, gh, q_block=None):
        super().__init__()
        self.gw, self.gh = gw, gh
        self.layers = hf.encoder.layers
        self.input_proj = hf.patch_embedder.input_proj
        self.heads = hf.config.num_attention_heads
        self.hd = hf.config.head_dim
        self.root = hf.config.hidden_size ** 0.5
        self.q_block = q_block
        pos = position_ids(gw, gh)
        with torch.no_grad():
            pe = hf.patch_embedder._position_embeddings(pos, torch.zeros(pos.shape[:2], dtype=torch.bool))
            cos, sin = hf.encoder.rotary_emb(pe, pos)
        self.register_buffer("pos_emb", pe[0].clone())
        self.register_buffer("cos", cos[0].clone())  # [N, head_dim]
        self.register_buffer("sin", sin[0].clone())
        self.n = gw * gh

    @staticmethod
    def rms(x, w=None, eps=1e-6):
        x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
        return x if w is None else x * w

    @staticmethod
    def clin(m, x):
        x = torch.clamp(x, m.input_min, m.input_max)
        x = m.linear(x)
        return torch.clamp(x, m.output_min, m.output_max)

    def rope(self, x):
        # x: [N, heads, hd]. Axial RoPE: first half of hd uses x-position, second half y-position,
        # each half rotated as rotate_half within itself (matches apply_multidimensional_rope).
        cos, sin = self.cos[:, None, :], self.sin[:, None, :]
        half = self.hd // 2
        parts = []
        for i in range(2):
            xi = x[..., i * half:(i + 1) * half]
            ci, si = cos[..., i * half:(i + 1) * half], sin[..., i * half:(i + 1) * half]
            a, b = xi[..., : half // 2], xi[..., half // 2:]
            parts.append(xi * ci + torch.cat([-b, a], -1) * si)
        return torch.cat(parts, -1)

    def attn(self, layer, x):
        a = layer.self_attn
        n = self.n
        q = self.rope(self.rms(self.clin(a.q_proj, x).view(n, self.heads, self.hd), a.q_norm.weight))
        k = self.rope(self.rms(self.clin(a.k_proj, x).view(n, self.heads, self.hd), a.k_norm.weight))
        v = self.rms(self.clin(a.v_proj, x).view(n, self.heads, self.hd))
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # [heads, N, hd]
        kt = k.transpose(1, 2)
        if self.q_block is None:
            o = torch.softmax(q @ kt, -1) @ v  # scaling = 1.0 in Gemma 4 vision
        else:
            outs = [torch.softmax(q[:, s:s + self.q_block] @ kt, -1) @ v for s in range(0, n, self.q_block)]
            o = torch.cat(outs, 1)
        o = o.transpose(0, 1).reshape(n, self.heads * self.hd)
        return self.clin(a.o_proj, o)

    def forward(self, patches):
        x = self.input_proj(2 * (patches[0] - 0.5)) + self.pos_emb  # [N, C]
        for L in self.layers:
            h = self.attn(L, self.rms(x, L.input_layernorm.weight))
            x = x + self.rms(h, L.post_attention_layernorm.weight)
            m = L.mlp
            h = self.rms(x, L.pre_feedforward_layernorm.weight)
            h = self.clin(m.down_proj, F.gelu(self.clin(m.gate_proj, h), approximate="tanh") * self.clin(m.up_proj, h))
            x = x + self.rms(h, L.post_feedforward_layernorm.weight)
        g = x.view(1, self.gh, self.gw, -1).permute(0, 3, 1, 2)
        g = F.avg_pool2d(g, K)  # [1, C, gh/3, gw/3], raster order matches the HF pooler
        return (g.flatten(2).transpose(1, 2) * self.root)  # [1, tokens, C]


class FixedSizeVisionANE(FixedSizeVision):
    """Same math, laid out the way Apple recommends for the Neural Engine
    (ml-ane-transformers): activations as [1, C, 1, N], linears as 1x1 convs,
    attention split per head with einsum, softmax over the channel axis.
    Input/output shapes are unchanged ([1, N, 768] -> [1, tokens, C]).
    """

    def __init__(self, hf, gw, gh, q_block=None):
        super().__init__(hf, gw, gh, q_block)
        # tables as [1, C, 1, N] / [1, hd, 1, N]
        self.register_buffer("pos_c", self.pos_emb.t()[None, :, None, :].clone())
        self.register_buffer("cos_c", self.cos.t()[None, :, None, :].clone())
        self.register_buffer("sin_c", self.sin.t()[None, :, None, :].clone())

    @staticmethod
    def rms_c(x, w=None, eps=1e-6):  # normalise over dim 1
        x = x * torch.rsqrt((x * x).mean(1, keepdim=True) + eps)
        return x if w is None else x * w[None, :, None, None]

    @staticmethod
    def conv(lin, x):
        return F.conv2d(x, lin.weight[:, :, None, None], lin.bias)

    def clin_c(self, m, x):
        x = torch.clamp(x, m.input_min, m.input_max)
        return torch.clamp(self.conv(m.linear, x), m.output_min, m.output_max)

    def rope_c(self, x):  # x: [1, hd, 1, N] for one head
        half = self.hd // 2
        parts = []
        for i in range(2):
            xi = x[:, i * half:(i + 1) * half]
            ci, si = self.cos_c[:, i * half:(i + 1) * half], self.sin_c[:, i * half:(i + 1) * half]
            a, b = xi[:, : half // 2], xi[:, half // 2:]
            parts.append(xi * ci + torch.cat([-b, a], 1) * si)
        return torch.cat(parts, 1)

    def attn(self, layer, x):
        a = layer.self_attn
        q = self.clin_c(a.q_proj, x).split(self.hd, 1)
        k = self.clin_c(a.k_proj, x).split(self.hd, 1)
        v = self.clin_c(a.v_proj, x).split(self.hd, 1)
        outs = []
        for qi, ki, vi in zip(q, k, v):
            qi = self.rope_c(self.rms_c(qi, a.q_norm.weight))
            ki = self.rope_c(self.rms_c(ki, a.k_norm.weight)).transpose(1, 3)  # [1, N, 1, hd]
            vi = self.rms_c(vi)
            if self.q_block is None:
                w = torch.einsum("bchq,bkhc->bkhq", qi, ki).softmax(1)  # [1, Nk, 1, Nq]
                outs.append(torch.einsum("bkhq,bchk->bchq", w, vi))
            else:
                blk = []
                for s0 in range(0, self.n, self.q_block):
                    w = torch.einsum("bchq,bkhc->bkhq", qi[..., s0:s0 + self.q_block], ki).softmax(1)
                    blk.append(torch.einsum("bkhq,bchk->bchq", w, vi))
                outs.append(torch.cat(blk, 3))
        return self.clin_c(a.o_proj, torch.cat(outs, 1))

    def forward(self, patches):
        x = patches.transpose(1, 2)[:, :, None, :]  # [1, 768, 1, N]
        x = self.conv(self.input_proj, 2 * (x - 0.5)) + self.pos_c
        for L in self.layers:
            h = self.attn(L, self.rms_c(x, L.input_layernorm.weight))
            x = x + self.rms_c(h, L.post_attention_layernorm.weight)
            m = L.mlp
            h = self.rms_c(x, L.pre_feedforward_layernorm.weight)
            h = self.clin_c(m.down_proj, F.gelu(self.clin_c(m.gate_proj, h), approximate="tanh") * self.clin_c(m.up_proj, h))
            x = x + self.rms_c(h, L.post_feedforward_layernorm.weight)
        g = F.avg_pool2d(x.view(1, -1, self.gh, self.gw), K)
        return g.flatten(2).transpose(1, 2) * self.root


def reference(hf, patches, gw, gh):
    with torch.no_grad():
        return hf(pixel_values=patches, pixel_position_ids=position_ids(gw, gh)).last_hidden_state


def test_image(budget, seed=0):
    """A real M0 screenshot resized to this budget's size, if present; else noise."""
    from PIL import Image
    w, h = SIZES[budget]
    shots = sorted((ROOT / "m0_screen/screens").glob("*.png"))
    if shots:
        img = Image.open(shots[0]).convert("RGB").resize((w, h), Image.BICUBIC)
        return np.asarray(img, np.float32) / 255
    return np.random.default_rng(seed).random((h, w, 3), np.float32)


def cos_sim(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float(a @ b / (a.norm() * b.norm()))


def tag(a):
    return f"{a.budget}_{a.layout}" + ("" if a.q_block is None else f"_qb{a.q_block}")


def mlpackage(a):
    return OUT / f"gemma4_vision_{tag(a)}.mlpackage"


def cmd_convert(a):
    import coremltools as ct
    OUT.mkdir(parents=True, exist_ok=True)
    hf, _ = load_hf_vision()
    gw, gh = grid(a.budget)
    m = (FixedSizeVisionANE if a.layout == "ane" else FixedSizeVision)(hf, gw, gh, a.q_block).eval()
    x = patchify(test_image(a.budget))
    t = time.time(); ref = reference(hf, x, gw, gh); t_ref = time.time() - t
    with torch.no_grad():
        mine = m(x)
    print(f"budget {a.budget}: grid {gw}x{gh} = {gw*gh} patches -> {mine.shape[1]} tokens")
    print(f"wrapper vs HF reference: cos {cos_sim(mine, ref):.6f}  (HF fp32 CPU {t_ref:.1f}s)")
    t = time.time()
    traced = torch.jit.trace(m, x)
    mlm = ct.convert(
        traced,
        inputs=[ct.TensorType(name="patches", shape=x.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="tokens", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    path = mlpackage(a)
    mlm.save(str(path))
    print(f"converted in {time.time()-t:.0f}s -> {path.name}")
    np.save(OUT / f"patches_{a.budget}.npy", x.numpy())
    np.save(OUT / f"ref_{a.budget}.npy", ref.numpy())


def cmd_plan(a):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    path = mlpackage(a)
    keep = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)  # holds the compiled copy alive
    compiled = keep.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(path=compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
    fn = plan.model_structure.program.functions["main"]
    counts, cost = {}, {}
    names = {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU", "MLCPUComputeDevice": "CPU"}
    fallback = {}
    for op in fn.block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is None:
            continue
        dev = names.get(type(u.preferred_compute_device).__name__, type(u.preferred_compute_device).__name__)
        counts[dev] = counts.get(dev, 0) + 1
        c = plan.get_estimated_cost_for_mlprogram_operation(op)
        cost[dev] = cost.get(dev, 0) + (c.weight if c else 0)
        if dev != "ANE":
            fallback[op.operator_name] = fallback.get(op.operator_name, 0) + 1
    print(f"{path.name}  (compute units CPU_AND_NE)")
    print("ops by device:", counts)
    print("estimated cost share:", {k: round(v, 3) for k, v in cost.items()})
    print("non-ANE ops by type:", fallback)


def cmd_bench(a):
    import coremltools as ct
    path = mlpackage(a)
    x = np.load(OUT / f"patches_{a.budget}.npy")
    ref = torch.from_numpy(np.load(OUT / f"ref_{a.budget}.npy"))
    res = {}
    for cu in a.units.split(","):
        t = time.time()
        m = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, cu))
        load = time.time() - t
        y = m.predict({"patches": x})["tokens"]
        ts = []
        for _ in range(a.iters):
            t = time.perf_counter(); m.predict({"patches": x}); ts.append(time.perf_counter() - t)
        res[cu] = dict(load_s=round(load, 1), median_ms=round(1000 * float(np.median(ts)), 1),
                       min_ms=round(1000 * min(ts), 1), cos_vs_ref=round(cos_sim(torch.from_numpy(y), ref), 5))
        print(cu, res[cu], flush=True)
    (OUT / f"bench_{tag(a)}.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "plan", "bench"])
    ap.add_argument("--budget", type=int, default=280, choices=sorted(SIZES))
    ap.add_argument("--q-block", type=int, default=None, help="attention query block size (None = full)")
    ap.add_argument("--layout", default="seq", choices=["seq", "ane"])
    ap.add_argument("--units", default="CPU_AND_NE,CPU_AND_GPU,CPU_ONLY")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    {"convert": cmd_convert, "plan": cmd_plan, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
