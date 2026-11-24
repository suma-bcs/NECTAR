"""Neural-network building blocks on top of :mod:`nectar.autograd`."""
from __future__ import annotations

import numpy as np

from . import autograd as ag
from .autograd import Tensor


class Parameter(Tensor):
    def __init__(self, data, name: str = ""):
        super().__init__(data, requires_grad=True, name=name)


class Module:
    training: bool = True

    def parameters(self):
        out, seen = [], set()

        def collect(obj):
            if isinstance(obj, Parameter):
                if id(obj) not in seen:
                    seen.add(id(obj))
                    out.append(obj)
            elif isinstance(obj, Module):
                for v in obj.__dict__.values():
                    collect(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    collect(v)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect(v)

        collect(self)
        return out

    def named_parameters(self, prefix=""):
        out = []
        for k, v in self.__dict__.items():
            name = f"{prefix}{k}"
            if isinstance(v, Parameter):
                out.append((name, v))
            elif isinstance(v, Module):
                out.extend(v.named_parameters(name + "."))
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if isinstance(item, Module):
                        out.extend(item.named_parameters(f"{name}.{i}."))
                    elif isinstance(item, Parameter):
                        out.append((f"{name}.{i}", item))
        return out

    def train(self, mode: bool = True):
        def setmode(obj):
            if isinstance(obj, Module):
                obj.training = mode
                for v in obj.__dict__.values():
                    setmode(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    setmode(v)

        setmode(self)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self):
        return {k: v.data.copy() for k, v in self.named_parameters()}

    def load_state_dict(self, state):
        params = dict(self.named_parameters())
        for k, v in state.items():
            params[k].data[...] = v


def glorot(rng, fan_in, fan_out, shape=None):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape or (fan_in, fan_out))


class Linear(Module):
    def __init__(self, rng, d_in, d_out, bias=True):
        self.W = Parameter(glorot(rng, d_in, d_out))
        self.b = Parameter(np.zeros(d_out)) if bias else None

    def __call__(self, x):
        y = ag.matmul(x, self.W)
        return y + self.b if self.b is not None else y


class Embedding(Module):
    def __init__(self, rng, n, d, scale=0.05):
        self.weight = Parameter(rng.normal(0, scale, size=(n, d)))

    def __call__(self, idx):
        return ag.getitem(self.weight, np.asarray(idx, dtype=np.int64))


class LayerNorm(Module):
    def __init__(self, d):
        self.g = Parameter(np.ones(d))
        self.b = Parameter(np.zeros(d))

    def __call__(self, x):
        return ag.layer_norm(x, self.g, self.b)


class CausalSelfAttention(Module):
    """Multi-head causal self-attention with a key padding mask."""

    def __init__(self, rng, d, heads, dropout):
        self.h, self.d, self.dropout = heads, d, dropout
        self.qkv = Linear(rng, d, 3 * d)
        self.out = Linear(rng, d, d)
        self.rng = rng
        self.last_attention = None  # (B, H, L, L) probabilities of the last call

    def __call__(self, x, pad_mask):
        B, L, d = x.shape
        h, dh = self.h, d // self.h
        qkv = self.qkv(x).reshape(B, L, 3, h, dh).transpose(2, 0, 3, 1, 4)  # 3,B,h,L,dh
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = ag.matmul(q, k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(dh))
        causal = np.triu(np.ones((L, L), dtype=bool), 1)
        mask = causal[None, None] | (~pad_mask)[:, None, None, :]
        # keep at least the diagonal so fully padded rows stay finite
        mask = mask & ~np.eye(L, dtype=bool)[None, None]
        scores = scores + Tensor(np.where(mask, -1e9, 0.0))
        att = ag.softmax(scores, axis=-1)
        self.last_attention = att.data
        att = ag.dropout(att, self.dropout, self.training, self.rng)
        y = ag.matmul(att, v).transpose(0, 2, 1, 3).reshape(B, L, d)
        return self.out(y)


class TransformerBlock(Module):
    def __init__(self, rng, d, heads, dropout):
        self.ln1, self.ln2 = LayerNorm(d), LayerNorm(d)
        self.att = CausalSelfAttention(rng, d, heads, dropout)
        self.ff1, self.ff2 = Linear(rng, d, d), Linear(rng, d, d)
        self.dropout, self.rng = dropout, rng

    def __call__(self, x, pad_mask):
        x = x + ag.dropout(self.att(self.ln1(x), pad_mask), self.dropout, self.training, self.rng)
        hdn = self.ff2(ag.relu(self.ff1(self.ln2(x))))
        return x + ag.dropout(hdn, self.dropout, self.training, self.rng)


class GRU(Module):
    """Single-layer GRU unrolled over the time axis (batch-first)."""

    def __init__(self, rng, d_in, d_h):
        self.d_h = d_h
        self.Wx = Linear(rng, d_in, 3 * d_h)
        self.Wh = Linear(rng, d_h, 3 * d_h, bias=False)

    def __call__(self, x):
        B, L, _ = x.shape
        H = self.d_h
        xs = self.Wx(x)
        h = Tensor(np.zeros((B, H)))
        outs = []
        for t in range(L):
            xt = xs[:, t, :]
            hh = self.Wh(h)
            z = ag.sigmoid(xt[:, :H] + hh[:, :H])
            r = ag.sigmoid(xt[:, H:2 * H] + hh[:, H:2 * H])
            n = ag.tanh(xt[:, 2 * H:] + r * hh[:, 2 * H:])
            h = (1.0 - z) * n + z * h
            outs.append(h)
        return ag.stack(outs, axis=1)


def _im2col_index(size=6, k=3):
    rows, cols = [], []
    for i in range(size):
        for j in range(size):
            rows.append([i + a for a in range(k) for _ in range(k)])
            cols.append([j + b for _ in range(k) for b in range(k)])
    return np.array(rows), np.array(cols)


class ConvLSTM(Module):
    """ConvLSTM over a short sequence of single-channel 6x6 maps (layout N, 36, C)."""

    def __init__(self, rng, channels=4, size=6, kernel=3):
        self.C, self.size, self.k = channels, size, kernel
        cin = 1 + channels
        self.W = Parameter(glorot(rng, cin * kernel * kernel, 4 * channels))
        self.b = Parameter(np.zeros(4 * channels))
        self.rows, self.cols = _im2col_index(size, kernel)

    def __call__(self, seq):
        """``seq``: list of tensors (N, 36, 1). Returns the last hidden state (N, 36, C)."""
        N = seq[0].shape[0]
        C, S = self.C, self.size
        h = Tensor(np.zeros((N, S * S, C)))
        c = Tensor(np.zeros((N, S * S, C)))
        for x in seq:
            z = ag.concat([x, h], axis=-1).reshape(N, S, S, 1 + C)
            z = ag.pad_spatial(z, self.k // 2)
            patches = z[:, self.rows, self.cols, :].reshape(N, S * S, -1)  # N,36,9*(1+C)
            gates = ag.matmul(patches, self.W) + self.b
            i = ag.sigmoid(gates[..., :C])
            f = ag.sigmoid(gates[..., C:2 * C])
            o = ag.sigmoid(gates[..., 2 * C:3 * C])
            g = ag.tanh(gates[..., 3 * C:])
            c = f * c + i * g
            h = o * ag.tanh(c)
        return h


class Adam:
    def __init__(self, params, lr=0.002, betas=(0.9, 0.98), eps=1e-8, clip=5.0):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, betas[0], betas[1], eps, clip
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        grads = [p.grad if p.grad is not None else np.zeros_like(p.data) for p in self.params]
        if self.clip:
            norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
            if norm > self.clip:
                grads = [g * (self.clip / norm) for g in grads]
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            p.grad = None
