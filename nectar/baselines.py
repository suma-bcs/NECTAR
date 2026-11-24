"""Non-neural baselines: Pop, Pop-recent, Markov and BPR-MF."""
from __future__ import annotations

import numpy as np

from .data import Dataset


def pop_scorer(d: Dataset):
    counts = np.bincount(d.item[d.split == 0], minlength=d.n_items).astype(np.float64)
    return lambda d_, ev: np.tile(counts, (len(ev), 1))


def pop_recent_scorer(d: Dataset):
    return lambda d_, ev: d_.tau(d_.week[ev])[..., 2].astype(np.float64)


def markov_scorer(d: Dataset):
    def score(d_, ev):
        return d_.mu_rows(d_.item[ev - 1]).astype(np.float64) + 1e-3 * d_.tau(d_.week[ev])[..., 2]
    return score


def train_bpr(d: Dataset, dim=64, lr=0.05, reg=0.002, epochs=30, seed=0, log=print):
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(d.split == 0)
    users, items = d.user[tr], d.item[tr]
    U = rng.normal(0, 0.1, (d.n_users, dim))
    V = rng.normal(0, 0.1, (d.n_items, dim))
    bi = np.zeros(d.n_items)
    seen = {}
    for uu, ii in zip(users, items):
        seen.setdefault(int(uu), set()).add(int(ii))
    for ep in range(epochs):
        order = rng.permutation(len(tr))
        for s in range(0, len(order), 1024):
            o = order[s:s + 1024]
            u, i = users[o], items[o]
            j = rng.integers(1, d.n_items, len(o))
            x = (U[u] * (V[i] - V[j])).sum(1) + bi[i] - bi[j]
            g = 1.0 / (1.0 + np.exp(x))  # d/dx log sigmoid(x)
            gu = g[:, None] * (V[i] - V[j]) - reg * U[u]
            gi = g[:, None] * U[u] - reg * V[i]
            gj = -g[:, None] * U[u] - reg * V[j]
            np.add.at(U, u, lr * gu)
            np.add.at(V, i, lr * gi)
            np.add.at(V, j, lr * gj)
            np.add.at(bi, i, lr * (g - reg * bi[i]))
            np.add.at(bi, j, lr * (-g - reg * bi[j]))
        if log:
            log(f"[BPR-MF] epoch {ep + 1}/{epochs}")

    def score(d_, ev):
        return U[d_.user[ev]] @ V.T + bi[None, :]

    return score


SCORERS = {"Pop": pop_scorer, "Pop-recent": pop_recent_scorer, "Markov": markov_scorer}
