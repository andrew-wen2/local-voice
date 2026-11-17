"""Custom Metal kernel #1: the vision encoder's q/k/v prologue, fused (kernel track).

What it replaces. In each of the 16 vision layers, between the q/k/v projections and attention, mlx-vlm runs:

    q = q_norm(q)        # RMSNorm over each head's 64 values, in fp32, times a learned weight
    k = k_norm(k)        # same, own weight
    v = v_norm(v)        # same, no weight
    q = rope2d(q, pos)   # 2D rotary position: channels 0-31 rotated by the patch's x, 32-63 by its y
    k = rope2d(k, pos)
    q, k, v = transpose to [batch, heads, tokens, 64] for the attention kernel

Each line is several separate GPU kernels (casts, squares, means, multiplies, concatenations), and every one
reads and writes the whole tensor (14.7 MB each for q, k and v at 1120). The work is tiny; the time is memory
traffic and kernel launches: ~11 ms per layer measured, against ~1 ms to read q, k, v once and write them
once at this chip's ~120 GB/s.

This kernel does all of it in one pass: each value is read once, normalized, rotated and written straight
into the layout attention wants.

How the work is split (the part worth learning):

- One SIMD group (32 GPU threads that run in lockstep, Apple's equivalent of a CUDA warp) handles one
  (token, head) pair of one of q / k / v: 64 values, so 2 per thread.
- The 2 values a thread holds are exactly the pair RoPE rotates together. RoPE splits the head into an
  x half (channels 0-31) and a y half (32-63); inside each half, channel j is rotated with channel j+16.
  So thread t takes half p = t // 16 and pair j = t % 16: channels p*32 + j and p*32 + j + 16.
- RMSNorm needs the sum of squares over all 64 values, which are spread over the 32 threads. simd_sum()
  adds a value across the SIMD group in hardware, with no shared memory and no barrier.
- cos/sin depend only on the patch position, not on the layer, so they're computed once per image
  (rope_table) and read from a small table here, instead of 16 layers x 2 (q and k) times.
- Math is fp32 throughout and rounded to bf16 once, at the end. mlx-vlm rounds to bf16 after each RoPE
  step, so results differ at the rounding level (checked against an fp32 reference, not for bit equality).
"""
import mlx.core as mx

HEAD = 64          # vision head size
HALF = 16          # RoPE pairs per spatial half (32 channels per half, rotated in pairs j, j+16)

# Metal source for the kernel body. MLX wraps it in a function whose arguments are the inputs and outputs
# named below; thread_position_in_grid etc. are Metal built-ins that MLX passes in when the body uses them.
_SOURCE = """
    uint lane  = thread_position_in_grid.x;          // 0..31: this thread's slot in the SIMD group
    uint row   = thread_position_in_grid.y;          // which (token, head): token * H + head
    uint which = thread_position_in_grid.z;          // 0 = q, 1 = k, 2 = v

    uint H = n_heads;
    uint L = n_tokens;
    if (row >= L * H) return;
    uint tok  = row / H;
    uint head = row % H;

    // The two channels this thread owns: a RoPE pair in the x half (lanes 0-15) or y half (16-31).
    uint p  = lane / 16;
    uint j  = lane % 16;
    uint c1 = p * 32 + j;
    uint c2 = c1 + 16;

    // Inputs are [L, H*64] (the projections' output); read our two values, widen to fp32.
    uint src = tok * (H * 64) + head * 64;
    float x1, x2;
    if (which == 0)      { x1 = float(q[src + c1]); x2 = float(q[src + c2]); }
    else if (which == 1) { x1 = float(k[src + c1]); x2 = float(k[src + c2]); }
    else                 { x1 = float(v[src + c1]); x2 = float(v[src + c2]); }

    // RMSNorm: mean of squares over the head's 64 values = (sum over the 32 lanes of 2 squares) / 64.
    float ss  = simd_sum(x1 * x1 + x2 * x2);
    float inv = metal::rsqrt(ss / 64.0f + eps);
    x1 *= inv;
    x2 *= inv;

    if (which < 2) {
        // Learned scale (q_norm or k_norm), then the 2D rotation from the per-token table.
        x1 *= float(norm_w[which * 64 + c1]);
        x2 *= float(norm_w[which * 64 + c2]);
        uint t = (tok * 2 + p) * 16 + j;               // rope table is [L, 2 halves, 16 pairs]
        float c = cos_t[t];
        float s = sin_t[t];
        float r1 = x1 * c - x2 * s;                    // rotate_half: (x1, x2) -> (-x2, x1)
        float r2 = x2 * c + x1 * s;
        x1 = r1;
        x2 = r2;
    }

    // Write straight into attention's layout [H, L, 64] (batch 1).
    uint dst = (head * L + tok) * 64;
    if (which == 0)      { q_out[dst + c1] = T(x1); q_out[dst + c2] = T(x2); }
    else if (which == 1) { k_out[dst + c1] = T(x1); k_out[dst + c2] = T(x2); }
    else                 { v_out[dst + c1] = T(x1); v_out[dst + c2] = T(x2); }
"""

_kernel = mx.fast.metal_kernel(
    name="vision_qkv_norm_rope",
    input_names=["q", "k", "v", "norm_w", "cos_t", "sin_t", "n_heads", "n_tokens", "eps"],
    output_names=["q_out", "k_out", "v_out"],
    source=_SOURCE,
)


def rope_table(positions: mx.array, base: float):
    """cos/sin for every patch, [L, 2, 16] fp32, as mlx-vlm computes them (before its bf16 cast).
    positions: [1, L, 2] (x, y) patch coordinates. Same for all 16 layers of one image."""
    exps = (2.0 / 32) * mx.arange(0, HALF).astype(mx.float32)
    timescale = mx.power(base, exps)                                   # [16]
    ang = positions[0].astype(mx.float32)[..., None] / timescale       # [L, 2, 16]
    return mx.cos(ang), mx.sin(ang)


def qkv_norm_rope(q, k, v, q_w, k_w, cos_t, sin_t, n_heads: int, eps: float = 1e-6):
    """q, k, v: [1, L, H*64] projection outputs. Returns q, k, v as [1, H, L, 64], normalized and rotated."""
    L = q.shape[1]
    norm_w = mx.concatenate([q_w, k_w]).astype(mx.float32)            # [128]: q weights, then k weights
    outs = _kernel(
        inputs=[q, k, v, norm_w, cos_t, sin_t, n_heads, L, eps],
        template=[("T", q.dtype)],
        grid=(32, L * n_heads, 3),          # threads: 32 per (token, head), for each of q, k, v
        threadgroup=(32, 8, 1),             # 8 SIMD groups per threadgroup
        output_shapes=[(1, n_heads, L, HEAD)] * 3,
        output_dtypes=[q.dtype] * 3,
    )
    return outs[0], outs[1], outs[2]


def attention_call(self, x, positions, mask=None, rope=None):
    """Drop-in for mlx-vlm's VisionAttention.__call__ using the fused kernel. rope: (cos, sin) from
    rope_table(); computed here if not given (it's the same for every layer, so callers should share it)."""
    B, L, _ = x.shape
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)
    cos_t, sin_t = rope if rope is not None else rope_table(positions, self.rope_base_frequency)
    q, k, v = qkv_norm_rope(q, k, v, self.q_norm.weight, self.k_norm.weight, cos_t, sin_t,
                            self.num_heads, self.q_norm.eps)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
    return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
