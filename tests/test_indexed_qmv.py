"""kernels/indexed_qmv.py: the (scale, bias) index is lossless and the one-token kernel matches MLX bit for bit."""
import mlx.core as mx
import mlx.nn as nn

from kernels.indexed_qmv import IndexedQuantizedEmbedding, IndexedQuantizedLinear, build_index, expand, indexed_qmv


def _quantized(n=1024, k=2048, seed=0):
    mx.random.seed(seed)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    return wq, s, b


def test_index_round_trips_exactly():
    _, s, b = _quantized()
    idx, table = build_index(s, b)
    s2, b2 = expand(idx, table)
    assert idx.dtype == mx.uint16 and mx.array_equal(s2, s).item() and mx.array_equal(b2, b).item()


def test_one_token_kernel_is_bit_identical_to_mlx():
    wq, s, b = _quantized()
    idx, table = build_index(s, b)
    x = (mx.random.normal((1, 2048)) * 0.5).astype(mx.bfloat16)
    ref = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
    assert mx.array_equal(indexed_qmv(x, wq, idx, table), ref).item()


def test_drop_in_modules_match_the_originals():
    ql = nn.QuantizedLinear(2048, 1024, bias=False, group_size=64, bits=4)
    ql.set_dtype(mx.bfloat16)                          # bf16 scales/biases, as in the model file
    il = IndexedQuantizedLinear(ql)
    for m in (1, 3):
        x = mx.random.normal((m, 2048)).astype(mx.bfloat16)
        assert mx.array_equal(il(x), ql(x)).item()
    qe = nn.QuantizedEmbedding(4096, 512, group_size=64, bits=4)
    qe.set_dtype(mx.bfloat16)
    ie = IndexedQuantizedEmbedding(qe)
    ids = mx.array([[0, 5, 4095]])
    assert mx.array_equal(ie(ids), qe(ids)).item()
    h = mx.random.normal((1, 512)).astype(mx.float32)
    assert mx.array_equal(ie.as_linear(h), qe.as_linear(h)).item()


def test_two_rows_are_each_bit_identical_to_one_row():
    """2 rows of x share each weight block load (18 vs 25 ms over all 342 matmuls,
    cold weights); every output row must equal the one-token kernel on that row alone, bit for bit."""
    import mlx.core as mx
    import numpy as np
    import kernels.indexed_qmv as Q
    rng = np.random.default_rng(3)
    for N, K in ((2560, 10240), (10240, 2560), (2048, 2560)):
        w = mx.array(rng.integers(0, 2**32, (N, K // 8), dtype=np.uint32))
        # real tensors use 928-5,179 distinct (scale, bias) pairs: draw from a small set, as they do
        scales = mx.array(rng.choice(rng.random(40) * 0.01, (N, K // 64)).astype(np.float32)).astype(mx.bfloat16)
        biases = mx.array(rng.choice(rng.random(40) * 0.01 - 0.005, (N, K // 64)).astype(np.float32)).astype(mx.bfloat16)
        idx, table = Q.build_index(scales, biases)
        x = mx.array(rng.standard_normal((2, K)).astype(np.float32)).astype(mx.bfloat16)
        y = Q.indexed_matmul(x, w, idx, table)
        rows = mx.concatenate([Q.indexed_qmv(x[i:i + 1], w, idx, table) for i in range(2)])
        assert mx.array_equal(y, rows).item(), (N, K)
