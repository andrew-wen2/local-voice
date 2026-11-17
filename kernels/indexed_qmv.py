"""Custom Metal kernel #2: 4-bit matrix-vector product with indexed group scales.

The idea (lossless). MLX stores each group of 64 weights with its own bf16 scale and bf16 bias: 32 bits of
metadata per 64 four-bit codes, 0.29 GB of Gemma's 2.6 GB of text weights. But each weight tensor uses only
928-5,179 distinct (scale, bias) pairs across up to 163,840 groups (measured).
So store, per tensor, a table of its distinct pairs, and per group a 16-bit index into it: 16 bits instead
of 32, the same values bit for bit.

The kernel. Answering a question runs one token at a time, which is a matrix-vector product per weight
matrix, and it is limited by memory bandwidth (measured: 102 GB/s of the 122 GB/s this chip copies at). This
kernel is MLX's own `qmv_fast` for 4-bit, group size 64 (mlx/backend/metal/kernels/quantized.h), copied
step for step so the floating-point operations happen in the same order, with one change: a thread reads its
group's 16-bit index and looks the (scale, bias) pair up in the tensor's table, instead of reading the two
bf16 values directly. Same values, same order, so the output should be bit-identical to mx.quantized_matmul.

How MLX splits the work (worth knowing):
- A threadgroup is 2 SIMD groups (64 threads) and produces 8 output rows; each SIMD group owns 4 rows.
- Each thread takes 16 consecutive input values (2 packed uint32 words = 16 four-bit codes) per 512-wide
  block, so 32 threads cover 512 inputs, and 4 threads share one group of 64 (one index lookup each).
- The 4-bit trick: instead of shifting each code down, x is pre-divided by 1, 16, 256, 4096, and the code is
  masked in place (w & 0x00f0 is code*16), so x*(w & mask) needs no shift.
- Each thread's partial dot products are added across the SIMD group with simd_sum (hardware, no barrier).
- y = scale * sum(x*q) + bias * sum(x), per group, so the bias costs one multiply per group.

Larger inputs (a screen read's ~1,100 tokens) are limited by compute, not memory; there the pairs are looked
up once per layer into ordinary scales/biases and MLX's matrix kernel runs as usual (indexed_matmul).
"""
import mlx.core as mx
import numpy as np

GROUP = 64
BITS = 4

_SOURCE = """
    constexpr int packs_per_thread = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int pack_factor = 8;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = 16;
    constexpr int block_size = values_per_thread * 32;
    constexpr int group_size = 64;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    typedef float U;

    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size = K;
    const int out_vec_size = N;

    const device uint8_t* ws = (const device uint8_t*)w;
    thread U x_thread[values_per_thread];
    thread U result[results_per_simdgroup] = {0};

    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;
    const int out_row = tid.y * (2 * results_per_simdgroup) + simd_gid * results_per_simdgroup;

    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    const device uint16_t* ix = idx + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* xp = x + tid.x * in_vec_size + simd_lid * values_per_thread;
    device T* yp = y + tid.x * out_vec_size + out_row;

    for (int k = 0; k < in_vec_size; k += block_size) {
        // load_vector<T, U, 16, 4>
        U sum = 0;
        for (int i = 0; i < values_per_thread; i += 4) {
            sum += U(xp[i]) + U(xp[i + 1]) + U(xp[i + 2]) + U(xp[i + 3]);
            x_thread[i] = xp[i];
            x_thread[i + 1] = xp[i + 1] / 16.0f;
            x_thread[i + 2] = xp[i + 2] / 256.0f;
            x_thread[i + 3] = xp[i + 3] / 4096.0f;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            const device uint16_t* wl = (const device uint16_t*)(ws + row * in_vec_size_w);
            const uint g = ix[row * in_vec_size_g];
            U s = table[2 * g];
            U b = table[2 * g + 1];
            // qdot<U, 16, 4>
            U accum = 0;
            for (int i = 0; i < (values_per_thread / 4); i++) {
                accum += (x_thread[4 * i] * (wl[i] & 0x000f) + x_thread[4 * i + 1] * (wl[i] & 0x00f0) +
                          x_thread[4 * i + 2] * (wl[i] & 0x0f00) + x_thread[4 * i + 3] * (wl[i] & 0xf000));
            }
            result[row] += s * accum + sum * b;
        }
        ws += block_size * bytes_per_pack / pack_factor;
        ix += block_size / group_size;
        xp += block_size;
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0) {
            yp[row] = static_cast<T>(result[row]);
        }
    }
"""

_kernel = mx.fast.metal_kernel(name="indexed_qmv_4bit_g64", input_names=["w", "idx", "table", "x"],
                               output_names=["y"], source=_SOURCE)


# ---------------------------------------------------------------- a few rows at once
# Kernel lever 1. For 2+ input rows (a question's last words, a retrieval's question pass, the question prefill),
# indexed_matmul used to rebuild every matrix's full scales and biases from the index on each call (table[idx], ~0.3
# GB of metadata written and read per forward pass) and then run MLX's multi-row kernel, which at these sizes is far
# below the memory limit: a full forward pass took 27 ms for 1 token, 64 ms for 9, 70 ms for 16 (4.6k context).
# This kernel is the one-token kernel above with the weight block loaded once and applied to up to MR rows of x: every
# row's floating-point operations happen in the same order as when that row runs alone, so each output row is
# bit-identical to the one-token kernel's (and to MLX's one-token qmv_fast). It differs from MLX's own multi-row
# kernel only in rounding (MLX's kernels for 1 and 2+ rows already differ from each other that way).
_SOURCE_MULTI = """
    constexpr int packs_per_thread = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int pack_factor = 8;
    constexpr int bytes_per_pack = 4;
    constexpr int values_per_thread = 16;
    constexpr int block_size = values_per_thread * 32;
    constexpr int group_size = 64;
    constexpr int scale_step_per_thread = group_size / values_per_thread;
    typedef float U;

    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size = K;
    const int out_vec_size = N;
    const int m0 = tid.x * MR;

    const device uint8_t* ws = (const device uint8_t*)w;
    thread U x_thread[MR][values_per_thread];
    thread U sums[MR];
    thread U result[MR][results_per_simdgroup];
    for (int m = 0; m < MR; m++) {
        for (int r = 0; r < results_per_simdgroup; r++) { result[m][r] = 0; }
    }

    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / group_size;
    const int out_row = tid.y * (2 * results_per_simdgroup) + simd_gid * results_per_simdgroup;

    ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    const device uint16_t* ix = idx + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* xp = x + m0 * in_vec_size + simd_lid * values_per_thread;

    for (int k = 0; k < in_vec_size; k += block_size) {
        for (int m = 0; m < MR; m++) {
            if (m0 + m < M) {
                const device T* xm = xp + m * in_vec_size;
                U sum = 0;
                for (int i = 0; i < values_per_thread; i += 4) {
                    sum += U(xm[i]) + U(xm[i + 1]) + U(xm[i + 2]) + U(xm[i + 3]);
                    x_thread[m][i] = xm[i];
                    x_thread[m][i + 1] = xm[i + 1] / 16.0f;
                    x_thread[m][i + 2] = xm[i + 2] / 256.0f;
                    x_thread[m][i + 3] = xm[i + 3] / 4096.0f;
                }
                sums[m] = sum;
            }
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            const device uint16_t* wl = (const device uint16_t*)(ws + row * in_vec_size_w);
            const uint g = ix[row * in_vec_size_g];
            U s = table[2 * g];
            U b = table[2 * g + 1];
            // the masked codes as floats, once for all rows of x (the same values the one-token kernel converts)
            U wf[values_per_thread];
            for (int i = 0; i < (values_per_thread / 4); i++) {
                wf[4 * i] = U(wl[i] & 0x000f);
                wf[4 * i + 1] = U(wl[i] & 0x00f0);
                wf[4 * i + 2] = U(wl[i] & 0x0f00);
                wf[4 * i + 3] = U(wl[i] & 0xf000);
            }
            for (int m = 0; m < MR; m++) {
                if (m0 + m < M) {
                    U accum = 0;
                    for (int i = 0; i < (values_per_thread / 4); i++) {
                        accum += (x_thread[m][4 * i] * wf[4 * i] + x_thread[m][4 * i + 1] * wf[4 * i + 1] +
                                  x_thread[m][4 * i + 2] * wf[4 * i + 2] + x_thread[m][4 * i + 3] * wf[4 * i + 3]);
                    }
                    result[m][row] += s * accum + sums[m] * b;
                }
            }
        }
        ws += block_size * bytes_per_pack / pack_factor;
        ix += block_size / group_size;
        xp += block_size;
    }
    for (int m = 0; m < MR; m++) {
        if (m0 + m < M) {
            device T* yp = y + (m0 + m) * out_vec_size + out_row;
            for (int row = 0; row < results_per_simdgroup; row++) {
                U r = simd_sum(result[m][row]);
                if (simd_lid == 0) {
                    yp[row] = static_cast<T>(r);
                }
            }
        }
    }
"""

_kernel_multi = mx.fast.metal_kernel(name="indexed_qmv_multi_4bit_g64", input_names=["w", "idx", "table", "x"],
                                     output_names=["y"], source=_SOURCE_MULTI)
MULTI_MAX = 2                       # used for exactly 2 rows: 18.1 vs 25.0 ms over all 342 matmuls; from 3 rows this
MULTI_MR = 2                        # ALU-bound design is no faster than MLX's matrix kernel


def indexed_qmv_multi(x: mx.array, w: mx.array, idx: mx.array, table: mx.array, mr: int = MULTI_MR) -> mx.array:
    """x [M, K] (2 <= M <= MULTI_MAX) @ dequant(w, table[idx]).T -> [M, N]; each row as indexed_qmv computes it."""
    M, K = x.shape
    N = w.shape[0]
    assert fast_ok(K, N) and w.shape[1] * 8 == K and idx.shape == (N, K // GROUP)
    return _kernel_multi(inputs=[w, idx, table, x],
                         template=[("T", x.dtype), ("K", K), ("N", N), ("M", M), ("MR", mr)],
                         grid=(64 * (-(-M // mr)), N // 8, 1), threadgroup=(64, 1, 1),
                         output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]


def build_index(scales: mx.array, biases: mx.array):
    """(idx uint16 [rows, groups], table [pairs, 2] in the scales' dtype): table[idx] == (scales, biases) exactly.
    Scales and biases must be a 2-byte float type (bf16/fp16), as in the model file."""
    assert scales.dtype.size == 2 and biases.dtype == scales.dtype, scales.dtype
    s = np.asarray(scales.view(mx.uint16)).astype(np.uint32)
    b = np.asarray(biases.view(mx.uint16)).astype(np.uint32)
    pairs, inv = np.unique((s << 16) | b, return_inverse=True)
    if len(pairs) > 65536:
        raise ValueError(f"{len(pairs)} distinct pairs: more than a 16-bit index can address")
    tab = np.stack([(pairs >> 16).astype(np.uint16), (pairs & 0xFFFF).astype(np.uint16)], axis=1)
    table = mx.array(tab).view(scales.dtype)
    idx = mx.array(inv.reshape(scales.shape).astype(np.uint16))
    return idx, table


def fast_ok(K: int, N: int) -> bool:
    """Shapes the kernel handles (as MLX's qmv_fast: K a multiple of 512, N a multiple of 8)."""
    return K % 512 == 0 and N % 8 == 0


def indexed_qmv(x: mx.array, w: mx.array, idx: mx.array, table: mx.array) -> mx.array:
    """x [M, K] (M small) @ dequant(w, table[idx]).T -> [M, N]."""
    M, K = x.shape
    N = w.shape[0]
    assert fast_ok(K, N) and w.shape[1] * 8 == K and idx.shape == (N, K // GROUP)
    return _kernel(inputs=[w, idx, table, x], template=[("T", x.dtype), ("K", K), ("N", N)],
                   grid=(64 * M, N // 8, 1), threadgroup=(64, 1, 1),
                   output_shapes=[(M, N)], output_dtypes=[x.dtype])[0]


def expand(idx: mx.array, table: mx.array):
    """The ordinary (scales, biases) back from the index (for big inputs and odd shapes)."""
    sb = table[idx]                                  # [rows, groups, 2]
    return sb[..., 0], sb[..., 1]


def indexed_matmul(x: mx.array, w: mx.array, idx: mx.array, table: mx.array) -> mx.array:
    """Any shape: the custom kernel for a few rows of x, else MLX's own kernel on the expanded scales."""
    lead, K = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, K)
    if x2.shape[0] == 1 and fast_ok(K, w.shape[0]):     # one token: bit-identical to MLX (checked)
        return indexed_qmv(x2, w, idx, table).reshape(*lead, w.shape[0])
    if x2.shape[0] <= MULTI_MAX and fast_ok(K, w.shape[0]):     # two: each row exactly as it would run alone
        return indexed_qmv_multi(x2, w, idx, table).reshape(*lead, w.shape[0])
    s, b = expand(idx, table)
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GROUP, bits=BITS)


# ---------------------------------------------------------------- drop-in modules

import mlx.nn as nn  # noqa: E402


class IndexedQuantizedLinear(nn.Module):
    """nn.QuantizedLinear (affine, 4-bit, group 64) with (scale, bias) pairs stored as a 16-bit index + table."""

    def __init__(self, ql: "nn.QuantizedLinear", idx=None, table=None):
        super().__init__()
        self.weight = ql["weight"]
        self.idx, self.table = (idx, table) if idx is not None else build_index(ql["scales"], ql["biases"])
        if "bias" in ql:
            self.bias = ql["bias"]
        self.freeze()

    def __call__(self, x):
        y = indexed_matmul(x, self["weight"], self["idx"], self["table"])
        return y + self["bias"] if "bias" in self else y


class IndexedQuantizedEmbedding(nn.Module):
    """nn.QuantizedEmbedding with indexed scales; as_linear is the tied output layer (262,144 rows)."""

    def __init__(self, qe: "nn.QuantizedEmbedding", idx=None, table=None):
        super().__init__()
        self.weight = qe["weight"]
        self.idx, self.table = (idx, table) if idx is not None else build_index(qe["scales"], qe["biases"])
        self.num_embeddings, self.dims = qe.num_embeddings, qe.dims
        self.freeze()

    def __call__(self, x):
        sb = self["table"][self["idx"][x]]
        return mx.dequantize(self["weight"][x], scales=sb[..., 0], biases=sb[..., 1], group_size=GROUP, bits=BITS)

    def as_linear(self, x):
        return indexed_matmul(x, self["weight"], self["idx"], self["table"])


def _convertible(m) -> bool:
    return (isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)) and m.group_size == GROUP and m.bits == BITS
            and getattr(m, "mode", "affine") == "affine" and m.get("biases") is not None
            and m["scales"].dtype.size == 2 and m["biases"].dtype == m["scales"].dtype)   # bf16/fp16 pairs


def index_scales(module: nn.Module, prefix: str = "", sidecar=None) -> dict:
    """Replace every convertible quantized layer under `module` in place. Returns {"layers", "bytes_saved"}.

    sidecar: arrays from build_sidecar's file, keyed "<prefix><layer>.idx/.table". With it, no scale or bias is
    ever read (call this before the model's parameters are evaluated) and no numpy runs at load time; without
    it the index is built here (numpy, ~3 s, and ~0.2 GB of freed numpy memory stays with the process)."""
    from mlx.utils import tree_unflatten
    new, saved = [], 0
    for name, m in module.named_modules():
        if not _convertible(m):
            continue
        key = prefix + name
        before = m["scales"].nbytes + m["biases"].nbytes
        idx, table = (sidecar[key + ".idx"], sidecar[key + ".table"]) if sidecar is not None else (None, None)
        cls = IndexedQuantizedEmbedding if isinstance(m, nn.QuantizedEmbedding) else IndexedQuantizedLinear
        r = cls(m, idx, table)
        saved += before - r["idx"].nbytes - r["table"].nbytes
        new.append((name, r))
    module.update_modules(tree_unflatten(new))
    return {"layers": len(new), "bytes_saved": saved}


def sidecar_path(model_dir) -> "Path":
    from pathlib import Path
    return Path(model_dir) / "indexed_scales.safetensors"


def build_sidecar(model_dir) -> dict:
    """One-off, at setup: every 4-bit group-64 tensor's (idx, table) into <model_dir>/indexed_scales.safetensors.
    Reads the model file's scales/biases directly (CPU); keys are the model file's tensor prefixes."""
    import json
    import struct
    from pathlib import Path
    f = next(p for p in Path(model_dir).glob("*.safetensors") if p.name != "indexed_scales.safetensors")
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    mm = np.memmap(f, dtype=np.uint8, mode="r")
    raw = lambda k: np.frombuffer(mm[8 + n + head[k]["data_offsets"][0]: 8 + n + head[k]["data_offsets"][1]],
                                  dtype=np.uint16).reshape(head[k]["shape"])
    out = {}
    for k in head:
        if not k.endswith(".scales") or "embed_tokens_per_layer" in k or "audio" in k or "vision" in k:
            continue
        p = k[: -len(".scales")]
        if head[k]["dtype"] != "BF16" or head.get(p + ".biases", {}).get("dtype") != "BF16":
            continue
        pairs, inv = np.unique((raw(k).astype(np.uint32) << 16) | raw(p + ".biases"), return_inverse=True)
        if len(pairs) > 65536:
            continue
        out[p + ".idx"] = mx.array(inv.reshape(head[k]["shape"]).astype(np.uint16))
        tab = np.stack([(pairs >> 16).astype(np.uint16), (pairs & 0xFFFF).astype(np.uint16)], axis=1)
        out[p + ".table"] = mx.array(tab).view(mx.bfloat16)
    mx.save_safetensors(str(sidecar_path(model_dir)), out)
    return {"tensors": len(out) // 2, "bytes": sum(v.nbytes for v in out.values())}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    d = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "models/gemma-4-e4b-it-4bit")
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print(build_sidecar(d), "->", sidecar_path(d))
    else:
        print("usage: python -m kernels.indexed_qmv build [MODEL_DIR]")
