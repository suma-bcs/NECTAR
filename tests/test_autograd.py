"""Finite-difference gradient checks for the autodiff engine and layers."""
import numpy as np
import pytest

from nectar import autograd as ag
from nectar import nn


@pytest.fixture(autouse=True)
def float64():
    ag.set_dtype(np.float64)
    yield
    ag.set_dtype(np.float32)


def numeric_grad(f, x, eps=1e-6):
    g = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x.data[i]
        x.data[i] = old + eps
        fp = f().data.sum()
        x.data[i] = old - eps
        fm = f().data.sum()
        x.data[i] = old
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def check(f, params, tol=1e-5):
    for p in params:
        p.grad = None
    out = f()
    out.sum().backward()
    for p in params:
        num = numeric_grad(f, p)
        assert np.allclose(p.grad, num, atol=tol, rtol=1e-4), (p.name, np.abs(p.grad - num).max())


def P(rng, *shape, name=""):
    return nn.Parameter(rng.normal(size=shape), name=name)


def test_elementwise_and_reductions():
    rng = np.random.default_rng(0)
    a, b = P(rng, 3, 4, name="a"), P(rng, 4, name="b")
    check(lambda: ag.tanh(a * b + a) - ag.sigmoid(a) * 2 + ag.exp(b * 0.1).sum(), [a, b])
    check(lambda: ag.softmax(a, -1) * ag.relu(a), [a])
    check(lambda: ag.logsumexp(a, -1), [a])
    check(lambda: ag.log(ag.sigmoid(a)).mean(axis=0), [a])


def test_matmul_broadcast():
    rng = np.random.default_rng(1)
    a, b = P(rng, 2, 3, 4, name="a"), P(rng, 4, 5, name="b")
    check(lambda: ag.matmul(a, b), [a, b])
    c = P(rng, 2, 5, 4, name="c")
    check(lambda: ag.matmul(a, c.transpose(0, 2, 1)), [a, c])


def test_layer_norm():
    rng = np.random.default_rng(2)
    x, g, b = P(rng, 3, 6, name="x"), P(rng, 6, name="g"), P(rng, 6, name="b")
    w = np.random.default_rng(9).normal(size=(3, 6))
    check(lambda: ag.layer_norm(x, g, b) * w, [x, g, b])


def test_indexing_and_shapes():
    rng = np.random.default_rng(3)
    t = P(rng, 5, 4, name="t")
    idx = np.array([[0, 2, 2], [4, 1, 0]])
    w = np.random.default_rng(8).normal(size=(2, 3, 4))
    check(lambda: ag.getitem(t, idx) * w, [t])
    check(lambda: ag.concat([t[:2], t[3:]], axis=0) * 1.5, [t])
    check(lambda: ag.stack([t[:, 0], t[:, 3]], axis=1) * 2.0, [t])
    x = P(rng, 2, 3, 3, 2, name="x")
    wp = np.random.default_rng(7).normal(size=(2, 5, 5, 2))
    check(lambda: ag.pad_spatial(x, 1) * wp, [x])


def test_layers():
    rng = np.random.default_rng(4)
    x = P(rng, 2, 5, 8, name="x")
    block = nn.TransformerBlock(rng, 8, 2, dropout=0.0)
    pad = np.array([[False, True, True, True, True], [True] * 5])
    w = np.random.default_rng(6).normal(size=(2, 5, 8))
    check(lambda: block(x, pad) * w, [x] + block.parameters()[:3], tol=1e-4)
    gru = nn.GRU(rng, 8, 4)
    check(lambda: gru(x), [x] + gru.parameters(), tol=1e-4)
    conv = nn.ConvLSTM(rng, channels=2)
    seq = [P(rng, 3, 36, 1, name=f"s{i}") for i in range(2)]
    check(lambda: conv(seq), seq + conv.parameters(), tol=1e-4)
