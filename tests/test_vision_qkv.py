"""kernels/vision_qkv.py must match mlx-vlm's q/k/v norm + 2D RoPE + transpose, to bf16 rounding.

Reference: mlx-vlm's own VisionRMSNorm / VisionRMSNormNoScale / apply_multidimensional_rope, run in fp32.
The kernel computes in fp32 and rounds once, so it should be at least as close to that reference as
mlx-vlm's bf16 path is."""
import mlx.core as mx
import numpy as np
import pytest

mlx_vlm = pytest.importorskip("mlx_vlm")
from mlx_vlm.models.gemma4.vision import (VisionRMSNorm, VisionRMSNormNoScale,  # noqa: E402
                                          apply_multidimensional_rope)

from kernels.vision_qkv import qkv_norm_rope, rope_table  # noqa: E402

H, D, BASE = 12, 64, 100.0


def _inputs(L=391, seed=0):
    rng = np.random.default_rng(seed)
    gw, gh = 23, 17                                           # a 23 x 17 patch grid = 391 tokens
    pos = np.stack(np.meshgrid(np.arange(gw), np.arange(gh)), -1).reshape(-1, 2)[:L]
    q, k, v = (mx.array(rng.normal(0, 3, (1, L, H * D)).astype(np.float32)).astype(mx.bfloat16) for _ in range(3))
    qw, kw = (mx.array(rng.normal(1, 0.3, D).astype(np.float32)).astype(mx.bfloat16) for _ in range(2))
    return q, k, v, qw, kw, mx.array(pos[None].astype(np.int32))


def _reference(q, k, v, qw, kw, pos, dtype):
    qn, kn, vn = VisionRMSNorm(D), VisionRMSNorm(D), VisionRMSNormNoScale()
    qn.weight, kn.weight = qw.astype(dtype), kw.astype(dtype)
    L = q.shape[1]
    q, k, v = (t.astype(dtype).reshape(1, L, H, D) for t in (q, k, v))
    q = apply_multidimensional_rope(qn(q), pos, BASE)
    k = apply_multidimensional_rope(kn(k), pos, BASE)
    v = vn(v)
    return [t.transpose(0, 2, 1, 3).astype(mx.float32) for t in (q, k, v)]


def _rel(a, b):
    return float(mx.abs(a.astype(mx.float32) - b).max() / mx.abs(b).max())


def test_matches_reference_to_rounding():
    q, k, v, qw, kw, pos = _inputs()
    cos_t, sin_t = rope_table(pos, BASE)
    out = qkv_norm_rope(q, k, v, qw, kw, cos_t, sin_t, H)
    ref32 = _reference(q, k, v, qw, kw, pos, mx.float32)
    ref16 = _reference(q, k, v, qw, kw, pos, mx.bfloat16)
    for name, o, r32, r16 in zip("qkv", out, ref32, ref16):
        assert o.shape == (1, H, q.shape[1], D)
        assert o.dtype == mx.bfloat16
        err_kernel, err_mlxvlm = _rel(o, r32), _rel(r16, r32)
        assert err_kernel < 8e-3, (name, err_kernel)          # about two bf16 rounding steps
        assert err_kernel <= err_mlxvlm + 1e-6, (name, err_kernel, err_mlxvlm)


def test_v_has_no_rotation_or_weight():
    q, k, v, qw, kw, pos = _inputs(seed=1)
    cos_t, sin_t = rope_table(pos, BASE)
    _, _, vo = qkv_norm_rope(q, k, v, qw, kw, cos_t, sin_t, H)
    v32 = v.astype(mx.float32).reshape(1, -1, H, D)
    want = (v32 * mx.rsqrt(mx.mean(v32 ** 2, axis=-1, keepdims=True) + 1e-6)).transpose(0, 2, 1, 3)
    assert _rel(vo, want) < 4e-3
