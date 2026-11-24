"""Turn interaction indices into padded model batches.

A batch is described by ``ends``: for every row, the global index of the last
history interaction. The row predicts the interaction that follows each history
position; for evaluation only the final target (``ends + 1``) is used.
"""
from __future__ import annotations

import numpy as np

from .data import Dataset


def build_batch(d: Dataset, ends: np.ndarray, L: int, removed: list | None = None) -> dict:
    """Build a left-padded batch of length ``L``.

    ``removed`` optionally lists, per row, global indices to drop from the
    history (used by the explanation-fidelity test).
    """
    B = len(ends)
    idx = np.full((B, L), -1, dtype=np.int64)
    for r, e in enumerate(ends):
        u = d.user[e]
        start = d.user_ptr[u]
        if removed is not None and removed[r] is not None and len(removed[r]):
            hist = np.setdiff1d(np.arange(start, e + 1), removed[r])[-L:]
        else:
            hist = np.arange(max(start, e - L + 1), e + 1)
        idx[r, L - len(hist):] = hist
    valid = idx >= 0
    safe = np.where(valid, idx, 0)
    # the target of position k is the next interaction of the same user
    nxt = np.where(valid, np.r_[safe[:, 1:].T, (np.asarray(ends) + 1)[None, :]].T, 0)
    nxt = np.minimum(nxt, len(d.item) - 1)
    users = d.user[np.asarray(ends)]
    same_user = d.user[nxt] == users[:, None]
    tvalid = valid & same_user

    tagged = d.tag_flag[safe] * valid
    tag_share = np.cumsum(tagged, axis=1) / np.maximum(np.cumsum(valid, axis=1), 1)
    b = {
        "idx": idx,
        "valid": valid,
        "items": np.where(valid, d.item[safe], 0),
        "gap": np.where(valid, d.gap[safe], 0.0),
        "behaviour": d.behaviour[safe] * valid[..., None],
        "tag_vec": d.tag_vec[safe] * valid[..., None],
        "tag_flag": tagged,
        "targets": np.where(tvalid, d.item[nxt], 0),
        "tvalid": tvalid,
        "req_nodes": d.ctx_nodes[nxt],
        "req_gap": np.where(tvalid, d.gap[nxt], 0.0),
        "req_week": d.week[nxt],
        "val_target": d.valence[nxt],
        "val_mask": (d.tag_flag[nxt] > 0) & tvalid,
        "tag_share": tag_share.astype(np.float32),
        "abs_dev": d.behaviour[safe, 1] * valid,
        "users": users,
        "target_idx": nxt,
    }
    return b


def gap_features(gap: np.ndarray) -> np.ndarray:
    """Eight sinusoidal features of log(1 + gap in seconds)."""
    x = np.log1p(np.maximum(gap, 0.0))[..., None]
    freqs = np.array([0.25, 0.5, 1.0, 2.0])
    return np.concatenate([np.sin(x * freqs), np.cos(x * freqs)], axis=-1).astype(np.float32)
