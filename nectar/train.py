"""Training with a sampled softmax over shared uniform negatives."""
from __future__ import annotations

import time

import numpy as np

from . import autograd as ag
from .autograd import Tensor
from .batching import build_batch
from .data import Dataset
from .evaluate import evaluate_model
from .model import ModelConfig, SeqRecModel
from .nn import Adam


def sample_windows(d: Dataset, rng: np.random.Generator, per_user: int = 4, p_random: float = 0.7) -> np.ndarray:
    """End indices of training windows: 70% at a random point, 30% at the latest training event."""
    ends = []
    for u in range(d.n_users):
        start, stop = d.user_ptr[u], d.user_ptr[u + 1]
        tr = np.flatnonzero(d.split[start:stop] == 0)
        if len(tr) < 2:
            continue
        last = start + tr[-1]
        for _ in range(per_user):
            if rng.random() < p_random:
                ends.append(rng.integers(start, last))  # target index e+1 <= last
            else:
                ends.append(last - 1)
    ends = np.array(ends, dtype=np.int64)
    rng.shuffle(ends)
    return ends


def _sampled_softmax(model: SeqRecModel, d: Dataset, o: Tensor, V: Tensor, targets, weeks, prev, negs, train=True):
    cfg = model.cfg
    pos = targets[:, None]
    tau_p = d.tau_pairs(weeks[:, None], pos) if cfg.use_tau else None
    mu_p = d.mu_pairs(prev[:, None], pos, loo_mask=1.0 if train else None) if cfg.use_mu else None
    s_pos = model.score(o, V, pos, tau_p, mu_p)  # N,1
    tau_n = d.tau_pairs(weeks[:, None], negs[None, :].repeat(len(weeks), 0)) if cfg.use_tau else None
    mu_n = (np.log1p(d.trans[prev][:, negs].toarray()) / 2.0).astype(np.float32) if cfg.use_mu else None
    s_neg = model.score(o, V, negs, tau_n, mu_n)  # N,S
    hit = targets[:, None] == negs[None, :]  # remove accidental hits of the positive among negatives
    if hit.any():
        s_neg = s_neg + Tensor(np.where(hit, -1e9, 0.0))
    logits = ag.concat([s_pos, s_neg], axis=1)
    return (ag.logsumexp(logits, axis=1) - s_pos.reshape(-1)).mean()


def loss_on_batch(model: SeqRecModel, d: Dataset, b: dict, negs: np.ndarray):
    cfg = model.cfg
    h, u, aux = model.forward(b)
    V = aux["V"]
    B, L, D = h.shape
    sel = np.flatnonzero(b["tvalid"].ravel())
    targets = b["targets"].ravel()[sel]
    weeks = b["req_week"].ravel()[sel]
    prev = b["items"].ravel()[sel]
    o = h.reshape(B * L, D)[sel]
    loss = _sampled_softmax(model, d, o, V, targets, weeks, prev, negs)
    if u is not None:
        P = u.shape[1]
        tv = b["tvalid"][:, L - P:].ravel()
        selu = np.flatnonzero(tv)
        if len(selu):
            ou = u.reshape(B * P, D)[selu]
            loss = loss + _sampled_softmax(model, d, ou, V, b["targets"][:, L - P:].ravel()[selu],
                                           b["req_week"][:, L - P:].ravel()[selu],
                                           b["items"][:, L - P:].ravel()[selu], negs)
    if "v_hat" in aux and b["val_mask"].any():
        m = b["val_mask"].astype(np.float32)
        diff = aux["v_hat"] - Tensor(b["val_target"])
        loss = loss + cfg.lam * ((diff * diff) * Tensor(m)).sum() * (1.0 / m.sum())
    return loss


def train_model(cfg: ModelConfig, d: Dataset, val_events: np.ndarray, seed: int = 0, max_epochs: int = 25,
                patience: int = 4, batch_size: int = 128, negatives: int = 256, lr: float = 0.002,
                windows_per_user: int = 4, log=print):
    rng = np.random.default_rng(seed + 1000)
    model = SeqRecModel(cfg, d, seed=seed)
    opt = Adam(model.parameters(), lr=lr)
    best, best_state, bad, history = -1.0, model.state_dict(), 0, []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        ends = sample_windows(d, rng, windows_per_user)
        losses = []
        for s in range(0, len(ends), batch_size):
            b = build_batch(d, ends[s:s + batch_size], cfg.L)
            negs = rng.integers(1, d.n_items, negatives)
            loss = loss_on_batch(model, d, b, negs)
            loss.backward()
            opt.step()
            losses.append(float(loss.data))
        per_event, _, _ = evaluate_model(model, d, val_events)
        val = float(np.mean(per_event["NDCG@10"]))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_ndcg10": val,
                        "minutes": (time.time() - t0) / 60})
        log(f"[{cfg.name} seed={seed}] epoch {epoch:2d} loss {np.mean(losses):.4f} val NDCG@10 {val:.4f}")
        if val > best:
            best, best_state, bad = val, model.state_dict(), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history, (time.time() - t0) / 60
