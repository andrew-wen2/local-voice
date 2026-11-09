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
