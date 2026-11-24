"""A small reverse-mode automatic differentiation engine on top of NumPy.

Only the operations needed by NECTAR and its baselines are implemented. Every
operation records its parents and a closure that propagates the gradient, and
``Tensor.backward`` runs the closures in reverse topological order.
"""
from __future__ import annotations

import numpy as np

DTYPE = np.float32


def set_dtype(dtype) -> None:
    """Switch the global floating-point type (float64 is used for gradient checks)."""
    global DTYPE
    DTYPE = dtype


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum ``grad`` over the axes that were broadcast to reach its shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "name")

    def __init__(self, data, requires_grad: bool = False, _parents=(), name: str = ""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward = None
        self.name = name

    # ------------------------------------------------------------------ basics
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def _accumulate(self, g: np.ndarray) -> None:
        if self.grad is None:
            self.grad = g.astype(DTYPE, copy=True)
        else:
            self.grad += g

    def backward(self, grad=None) -> None:
        if grad is None:
            grad = np.ones_like(self.data)
        order, seen = [], set()

        def visit(t):
            stack = [(t, False)]
            while stack:
                node, done = stack.pop()
                if done:
                    order.append(node)
                    continue
                if id(node) in seen:
                    continue
                seen.add(id(node))
                stack.append((node, True))
                for p in node._parents:
                    if p.requires_grad and id(p) not in seen:
                        stack.append((p, False))

        visit(self)
        self._accumulate(np.asarray(grad, dtype=DTYPE))
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)
        # free graph references held by intermediate nodes
        for node in order:
            if node._parents:
                node._parents = ()
                node._backward = None

    # --------------------------------------------------------------- operators
    def __add__(self, other):
        return add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return add(self, neg(as_tensor(other)))

    def __rsub__(self, other):
        return add(as_tensor(other), neg(self))

    def __mul__(self, other):
        return mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return mul(self, power(as_tensor(other), -1.0))

    def __neg__(self):
        return neg(self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __getitem__(self, idx):
        return getitem(self, idx)

    def sum(self, axis=None, keepdims=False):
        return tsum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.data.shape[a] for a in np.atleast_1d(axis)])
        return tsum(self, axis, keepdims) * (1.0 / float(n))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return transpose(self, axes)


def as_tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _make(data, parents, backward) -> Tensor:
    req = any(p.requires_grad for p in parents)
    out = Tensor(data, requires_grad=req, _parents=parents if req else ())
    if req:
        out._backward = backward
    return out


# ---------------------------------------------------------------- elementwise
def add(a, b) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)

    def bw(g):
        if a.requires_grad:
            a._accumulate(_unbroadcast(g, a.shape))
        if b.requires_grad:
            b._accumulate(_unbroadcast(g, b.shape))

    return _make(a.data + b.data, (a, b), bw)


def mul(a, b) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)

    def bw(g):
        if a.requires_grad:
            a._accumulate(_unbroadcast(g * b.data, a.shape))
        if b.requires_grad:
            b._accumulate(_unbroadcast(g * a.data, b.shape))

    return _make(a.data * b.data, (a, b), bw)


def neg(a) -> Tensor:
    def bw(g):
        a._accumulate(-g)

    return _make(-a.data, (a,), bw)


def power(a, p: float) -> Tensor:
    a = as_tensor(a)

    def bw(g):
        a._accumulate(g * p * np.power(a.data, p - 1))

    return _make(np.power(a.data, p), (a,), bw)


def exp(a) -> Tensor:
    out = np.exp(a.data)

    def bw(g):
        a._accumulate(g * out)

    return _make(out, (a,), bw)


def log(a) -> Tensor:
    def bw(g):
        a._accumulate(g / a.data)

    return _make(np.log(a.data), (a,), bw)


def tanh(a) -> Tensor:
    out = np.tanh(a.data)

    def bw(g):
        a._accumulate(g * (1.0 - out * out))

    return _make(out, (a,), bw)


def sigmoid(a) -> Tensor:
    out = 1.0 / (1.0 + np.exp(-np.clip(a.data, -40, 40)))

    def bw(g):
        a._accumulate(g * out * (1.0 - out))

    return _make(out, (a,), bw)


def relu(a) -> Tensor:
    mask = a.data > 0

    def bw(g):
        a._accumulate(g * mask)

    return _make(a.data * mask, (a,), bw)


# ------------------------------------------------------------------ reductions
def tsum(a, axis=None, keepdims=False) -> Tensor:
    out = a.data.sum(axis=axis, keepdims=keepdims)

    def bw(g):
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a._accumulate(np.broadcast_to(g, a.shape))

    return _make(out, (a,), bw)


def softmax(a, axis=-1) -> Tensor:
    x = a.data - a.data.max(axis=axis, keepdims=True)
    e = np.exp(x)
    out = e / e.sum(axis=axis, keepdims=True)

    def bw(g):
        a._accumulate(out * (g - (g * out).sum(axis=axis, keepdims=True)))

    return _make(out, (a,), bw)


def logsumexp(a, axis=-1) -> Tensor:
    m = a.data.max(axis=axis, keepdims=True)
    e = np.exp(a.data - m)
    s = e.sum(axis=axis, keepdims=True)
    out = (np.log(s) + m).squeeze(axis)

    def bw(g):
        a._accumulate(np.expand_dims(g, axis) * e / s)

    return _make(out, (a,), bw)


def layer_norm(x, gamma, beta, eps: float = 1e-5) -> Tensor:
    mu = x.data.mean(axis=-1, keepdims=True)
    xc = x.data - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = xc * inv
    out = xhat * gamma.data + beta.data
    n = x.shape[-1]

    def bw(g):
        if gamma.requires_grad:
            gamma._accumulate(_unbroadcast(g * xhat, gamma.shape))
        if beta.requires_grad:
            beta._accumulate(_unbroadcast(g, beta.shape))
        if x.requires_grad:
            gx = g * gamma.data
            dx = (inv / n) * (n * gx - gx.sum(-1, keepdims=True) - xhat * (gx * xhat).sum(-1, keepdims=True))
            x._accumulate(dx)

    return _make(out, (x, gamma, beta), bw)


# ---------------------------------------------------------------- linear algebra
def matmul(a, b) -> Tensor:
    a, b = as_tensor(a), as_tensor(b)

    def bw(g):
        if a.requires_grad:
            if b.ndim == 1:
                ga = np.multiply.outer(g, b.data)
            else:
                ga = g @ np.swapaxes(b.data, -1, -2)
            a._accumulate(_unbroadcast(ga, a.shape))
        if b.requires_grad:
            if a.ndim == 1:
                gb = np.multiply.outer(a.data, g)
            elif b.ndim == 1:
                gb = (np.swapaxes(a.data, -1, -2) @ g[..., None])[..., 0]
                gb = gb.reshape(-1, gb.shape[-1]).sum(0) if gb.ndim > 1 else gb
            else:
                gb = np.swapaxes(a.data, -1, -2) @ g
            b._accumulate(_unbroadcast(gb, b.shape))

    return _make(a.data @ b.data, (a, b), bw)


# ------------------------------------------------------------------ shape ops
def reshape(a, shape) -> Tensor:
    old = a.shape

    def bw(g):
        a._accumulate(g.reshape(old))

    return _make(a.data.reshape(shape), (a,), bw)


def transpose(a, axes) -> Tensor:
    inv = np.argsort(axes)

    def bw(g):
        a._accumulate(np.transpose(g, inv))

    return _make(np.transpose(a.data, axes), (a,), bw)


def getitem(a, idx) -> Tensor:
    if isinstance(idx, Tensor):
        idx = idx.data.astype(np.int64)

    parts = idx if isinstance(idx, tuple) else (idx,)
    advanced = any(isinstance(p, (np.ndarray, list)) for p in parts)

    def bw(g):
        full = np.zeros_like(a.data)
        if advanced:
            np.add.at(full, idx, g)
        else:
            full[idx] += g
        a._accumulate(full)

    return _make(a.data[idx], (a,), bw)


def concat(tensors, axis=-1) -> Tensor:
    tensors = [as_tensor(t) for t in tensors]
    sizes = [t.shape[axis] for t in tensors]
    splits = np.cumsum(sizes)[:-1]

    def bw(g):
        for t, part in zip(tensors, np.split(g, splits, axis=axis)):
            if t.requires_grad:
                t._accumulate(part)

    return _make(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors), bw)


def stack(tensors, axis=0) -> Tensor:
    tensors = [as_tensor(t) for t in tensors]

    def bw(g):
        for i, t in enumerate(tensors):
            if t.requires_grad:
                t._accumulate(np.take(g, i, axis=axis))

    return _make(np.stack([t.data for t in tensors], axis=axis), tuple(tensors), bw)


def pad_spatial(a, p: int) -> Tensor:
    """Zero-pad axes 1 and 2 of a (N, H, W, C) tensor by ``p`` on each side."""

    def bw(g):
        a._accumulate(g[:, p:-p, p:-p, :])

    return _make(np.pad(a.data, ((0, 0), (p, p), (p, p), (0, 0))), (a,), bw)


def dropout(a, rate: float, training: bool, rng: np.random.Generator) -> Tensor:
    if not training or rate <= 0:
        return a
    mask = (rng.random(a.shape) >= rate).astype(DTYPE) / (1.0 - rate)
    return mul(a, Tensor(mask))
