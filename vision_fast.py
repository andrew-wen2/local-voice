"""Faster Gemma 4 vision encoder (kernel track, ARCHITECTURE.md §4.4). Same outputs, less waste.

1. Patch position embedding as a gather. mlx-vlm builds a one-hot [patches, 2, 10240] float32 tensor
   (~786 MB at 1120) and multiplies it by the position table; a lookup gives bit-identical values
   (each one-hot row selects exactly one table row) in ~1.5 ms instead of ~51 ms.
2. No attention mask when no patch is padding. A fixed-size image (the only kind the engine sends) has
   no padding, so mlx-vlm's dense [1, 1, P, P] mask is all zeros (184 MB at 1120) and only slows the
   fused attention kernel (~12 ms per layer). Dropping it is bit-identical.
3. Optionally, each encoder layer under mx.compile, which fuses the element-wise ops around the
   matmuls (norms, RoPE, clips, GELU). Not bit-identical; checked separately (eval/exactness.py).
4. Optionally, attention's q/k/v norm + 2D RoPE + layout as one custom Metal kernel
   (kernels/vision_qkv.py), with the RoPE table computed once per image. Not bit-identical either.

enable(vision_tower) turns these on for one model (mlx-vlm's classes are wrapped once, per-instance
flag); the engine's own pausable loops call vision_mask() and layer_fn() directly.
"""
import mlx.core as mx

_installed = False
_compiled = {}


def position_embeddings(self, patch_positions: mx.array, padding_positions: mx.array) -> mx.array:
    """Drop-in for VisionPatchEmbedder._position_embeddings: table[0][x] + table[1][y], zero on padding.
    Row order and dtype match the one-hot matmul exactly (bf16 rows summed once, as there)."""
    t = self.position_embedding_table
    pos = mx.maximum(patch_positions, 0)               # padding positions are -1; they are zeroed below
    emb = mx.take(t[0], pos[..., 0], axis=0) + mx.take(t[1], pos[..., 1], axis=0)
    return mx.where(mx.expand_dims(padding_positions, -1), 0.0, emb)


def vision_mask(padding_positions: mx.array, dtype) -> mx.array | None:
    """The encoder's additive attention mask, or None when nothing is padding (then it would be all zeros)."""
    if not bool(mx.any(padding_positions).item()):
        return None
    valid = ~padding_positions
    m = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
    m = mx.where(m, mx.array(0.0, dtype=dtype), mx.array(-1e4, dtype=dtype))
    return mx.expand_dims(m, 1)


def block_forward(layer, h, pos, mask, rope):
    """VisionTransformerBlock.__call__ (mlx_vlm 0.7.12) with attention's q/k/v prologue as one custom
    Metal kernel (kernels/vision_qkv.py). rope: the per-image (cos, sin) table, shared by all 16 layers."""
    from kernels.vision_qkv import attention_call
    attn = attention_call(layer.self_attn, layer.input_layernorm(h), pos, mask, rope)
    h = h + layer.post_attention_layernorm(attn)
    y = layer.mlp(layer.pre_feedforward_layernorm(h))
    return h + layer.post_feedforward_layernorm(y)


def rope_for(vision_tower, positions):
    """The (cos, sin) table for one image, or None when the custom kernel is off."""
    return _rope_from_encoder(vision_tower.encoder, positions)


def layer_fn(layer, compiled: bool, kernel: bool = False):
    """Callable (h, positions, mask, rope=None) -> h for one encoder layer.
    kernel: the fused q/k/v kernel (needs rope from rope_for); compiled: under mx.compile (only when there
    is no mask, the case for every fixed-size image)."""
    def plain(h, pos, mask, rope=None):
        if kernel:
            return block_forward(layer, h, pos, mask, rope)
        return layer(h, pos, mask)
    if not compiled:
        return plain
    key = (id(layer), kernel)
    if key not in _compiled:
        # inputs=layer.state: the layer's weights are inputs read at every call, not constants frozen at the
        # first trace, so weights swapped in and out (ARCHITECTURE.md §3.9) are used and freed correctly.
        if kernel:
            _compiled[key] = mx.compile(lambda h, pos, c, s: block_forward(layer, h, pos, None, (c, s)),
                                        inputs=layer.state)
        else:
            _compiled[key] = mx.compile(lambda h, pos: layer(h, pos, None), inputs=layer.state)
    fn = _compiled[key]

    def call(h, pos, mask, rope=None):
        if mask is not None:
            return plain(h, pos, mask, rope)
        return fn(h, pos, *rope) if kernel else fn(h, pos)
    return call


def install():
    """Wrap mlx-vlm's Gemma 4 vision classes once. The fast paths apply only to models passed to enable(),
    so a model loaded without it (eval/exactness.py's "slowvision") runs mlx-vlm's code unchanged."""
    global _installed
    if _installed:
        return
    from mlx_vlm.models.gemma4 import vision as V
    orig_pos = V.VisionPatchEmbedder._position_embeddings
    orig_enc = V.VisionTransformerModel.__call__
    orig_model = V.VisionModel.__call__

    def pos_call(self, patch_positions, padding_positions):
        if getattr(self, "_lv_fast", False):
            return position_embeddings(self, patch_positions, padding_positions)
        return orig_pos(self, patch_positions, padding_positions)

    def encoder_call(self, hidden_states, positions, mask):
        if not getattr(self, "_lv_fast", False):
            return orig_enc(self, hidden_states, positions, mask)
        rope = _rope_from_encoder(self, positions)
        for layer in self.layers:
            hidden_states = layer_fn(layer, self._lv_compile, self._lv_kernel)(hidden_states, positions, mask, rope)
        return hidden_states

    def model_call(self, pixel_values, pixel_position_ids=None):
        # mlx-vlm builds the dense mask inline; route the fixed-size branch through _fixed_size_call.
        if getattr(self, "_lv_fast", False) and pixel_position_ids is None \
                and isinstance(pixel_values, mx.array) and pixel_values.ndim == 4:
            return _fixed_size_call(self, pixel_values)
        return orig_model(self, pixel_values, pixel_position_ids)

    V.VisionPatchEmbedder._position_embeddings = pos_call
    V.VisionTransformerModel.__call__ = encoder_call
    V.VisionModel.__call__ = model_call
    _installed = True


def enable(vision_tower, compile_layers: bool = False, kernel: bool = False):
    """Turn the fast paths on for one loaded model's vision tower."""
    install()
    for m in (vision_tower, vision_tower.patch_embedder, vision_tower.encoder):
        object.__setattr__(m, "_lv_fast", True)
    object.__setattr__(vision_tower.encoder, "_lv_compile", compile_layers)
    object.__setattr__(vision_tower.encoder, "_lv_kernel", kernel)


def _rope_from_encoder(enc, positions):
    if not getattr(enc, "_lv_kernel", False):
        return None
    from kernels.vision_qkv import rope_table
    return rope_table(positions, enc.layers[0].self_attn.rope_base_frequency)


def _fixed_size_call(vt, pixel_values):
    """VisionModel.__call__'s fixed-size branch (mlx_vlm 0.7.12, gemma4/vision.py) with vision_mask()."""
    import numpy as np
    B, C, H, W = pixel_values.shape
    num_real = (H // vt.patch_size) * (W // vt.patch_size)
    output_length = num_real // (vt.pooling_kernel_size ** 2)
    positions, padding, _ = vt._patch_positions_single(H, W, max_patches=num_real)
    patch_positions = mx.array(np.tile(positions[None], (B, 1, 1)))
    padding_positions = mx.array(np.tile(padding[None], (B, 1)))
    h = vt.patch_embedder(pixel_values, patch_positions, padding_positions)
    h = vt.encoder(h, patch_positions, vision_mask(padding_positions, h.dtype))
    pooled, pool_mask = vt.pooler(h, patch_positions, padding_positions, output_length=output_length)
    valid_mask = pool_mask if pool_mask.shape[1] == output_length else ~pool_mask
    real = [pooled[i, :int(valid_mask[i].astype(mx.int32).sum().item())] for i in range(B)]
    out = mx.concatenate(real, axis=0)[None]
    if vt.config.standardize:
        out = (out - vt.std_bias) * vt.std_scale
    return out
