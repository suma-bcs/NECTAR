"""Situation slices, fusion-weight analysis, bandit simulation and explanation fidelity."""
from __future__ import annotations

import numpy as np

from .batching import build_batch
from .data import Dataset
from .evaluate import _mask_seen

# --------------------------------------------------------------------- slices


def event_slices(d: Dataset, events: np.ndarray, L: int = 30) -> dict:
    """Boolean masks for the situation slices used in the paper."""
    gap = d.gap[events]
    slices = {
        "gap<=30min": gap <= 1800,
        "30min<gap<=30d": (gap > 1800) & (gap <= 30 * 86400),
        "gap>30d": gap > 30 * 86400,
    }
    unusual, tag_user = np.zeros(len(events), bool), np.zeros(len(events), bool)
    for r, e in enumerate(events):
        start = d.user_ptr[d.user[e]]
        blocks = d.hour[start:e] // 4
        share = np.mean(blocks == d.hour[e] // 4) if e > start else 1.0
        unusual[r] = share < 0.10
        tag_user[r] = d.tag_flag[start:e].any()
    slices.update({"unusual_hour": unusual, "usual_hour": ~unusual,
                   "tag_user": tag_user, "no_tag_user": ~tag_user})
    return slices


def slice_table(per_event_by_model: dict, slices: dict, metric: str = "NDCG@10") -> dict:
    return {name: {"n": int(mask.sum()),
                   **{m: float(np.mean(pe[metric][mask])) if mask.any() else float("nan")
                      for m, pe in per_event_by_model.items()}}
            for name, mask in slices.items()}

# --------------------------------------------------------------- fusion weights


def fusion_summary(d: Dataset, events: np.ndarray, analysis: dict, L: int = 30) -> dict:
    mods = list(analysis.get("modalities", []))
    if "alpha" not in analysis:
        return {}
    alpha, gamma = analysis["alpha"], analysis.get("gamma")
    out = {"modalities": mods, "alpha_mean": alpha.mean(0).tolist()}
    if gamma is not None:
        out["gamma_mean"] = gamma.mean(0).tolist()
        comb = alpha * gamma
        gap = d.gap[events]
        edges = [0, 60, 600, 3600, 86400, 7 * 86400, np.inf]
        labels = ["<1m", "1-10m", "10-60m", "1-24h", "1-7d", ">7d"]
        out["combined_by_gap"] = {lab: comb[(gap >= lo) & (gap < hi)].mean(0).tolist()
                                  for lab, lo, hi in zip(labels, edges[:-1], edges[1:])
                                  if ((gap >= lo) & (gap < hi)).any()}
        has_tag = np.array([d.tag_flag[max(d.user_ptr[d.user[e]], e - L):e].any() for e in events])
        out["combined_by_tag"] = {"with_tags": comb[has_tag].mean(0).tolist() if has_tag.any() else None,
                                  "without_tags": comb[~has_tag].mean(0).tolist() if (~has_tag).any() else None}
    if "M" in analysis:
        gap = d.gap[events]
        short, long_ = gap <= 660, gap > 18 * 3600
        if short.any() and long_.any():
            out["eccm_short"] = analysis["M"][short].mean(0).tolist()
            out["eccm_long"] = analysis["M"][long_].mean(0).tolist()
    return out

# ---------------------------------------------------------------------- bandit


def _user_state(model, d: Dataset, events: np.ndarray):
    model.eval()
    b = build_batch(d, events - 1, model.cfg.L)
    h, u, aux = model.forward(b, P=1)
    o = (u[:, -1, :] if u is not None else h[:, -1, :])
    sc = model.score(o, aux["V"], np.arange(d.n_items), d.tau(d.week[events]), d.mu_rows(d.item[events - 1])).data
    model.train()
    return o.data, aux["V"].data, sc


def bandit_simulation(model, d: Dataset, max_users: int = 3000, rounds: int = 5, top: int = 20,
                      alpha_ucb: float = 0.5, ts_scale: float = 0.05, eps: float = 0.1, seeds=(0, 1, 2)):
    """Offline re-ranking of the top-``top`` list with held-out ratings as rewards."""
    test = np.flatnonzero(d.split == 2)
    first_test = {}
    liked = {}
    for e in test:
        u = int(d.user[e])
        first_test.setdefault(u, e)
        if d.rating[e] >= 4:
            liked.setdefault(u, set()).add(int(d.item[e]))
    users = [u for u in first_test if u in liked and first_test[u] > d.user_ptr[u]]
    rng0 = np.random.default_rng(0)
    if len(users) > max_users:
        users = list(rng0.choice(users, max_users, replace=False))
    events = np.array([first_test[u] for u in users])
    cands, feats, rewards = [], [], []
    for s in range(0, len(events), 256):
        ev = events[s:s + 256]
        o, V, sc = _user_state(model, d, ev)
        _mask_seen(d, ev, sc)
        top_idx = np.argsort(-sc, axis=1)[:, :top]
        for r, e in enumerate(ev):
            items = top_idx[r]
            z = np.concatenate([o[r][None, :] * V[items] / 8.0, sc[r, items][:, None] / 10.0], axis=1)
            cands.append(items)
            feats.append(z)
            rewards.append(np.array([i in liked[int(d.user[e])] for i in items], dtype=float))
    dim = feats[0].shape[1]

    def run(policy, seed):
        rng = np.random.default_rng(seed)
        Ainv, bvec = np.eye(dim), np.zeros(dim)
        shown = [np.zeros(top, bool) for _ in users]
        hits, curve = 0, []
        for _ in range(rounds):
            for n in range(len(users)):
                avail = np.flatnonzero(~shown[n])
                Z = feats[n][avail]
                if policy == "greedy":
                    pick = avail[0]
                elif policy == "random":
                    pick = rng.choice(avail)
                else:
                    theta = Ainv @ bvec
                    if policy == "linucb":
                        val = Z @ theta + alpha_ucb * np.sqrt(np.einsum("ij,jk,ik->i", Z, Ainv, Z))
                        pick = avail[int(np.argmax(val))]
                    elif policy == "thompson":
                        th = rng.multivariate_normal(theta, ts_scale * Ainv)
                        pick = avail[int(np.argmax(Z @ th))]
                    elif policy == "egreedy":
                        pick = rng.choice(avail) if rng.random() < eps else avail[int(np.argmax(Z @ theta))]
                    else:
                        raise ValueError(policy)
                shown[n][pick] = True
                r = rewards[n][pick]
                z = feats[n][pick]
                Az = Ainv @ z
                Ainv -= np.outer(Az, Az) / (1.0 + z @ Az)  # Sherman-Morrison update of A^-1
                bvec += r * z
                hits += r
                curve.append(hits)
        return np.array(curve)

    out = {"users": len(users), "recommendations": len(users) * rounds}
    for policy in ("greedy", "linucb", "egreedy", "thompson", "random"):
        runs = [run(policy, s) for s in (seeds if policy in ("egreedy", "thompson", "random") else seeds[:1])]
        finals = [float(c[-1]) for c in runs]
        out[policy] = {"hits_mean": float(np.mean(finals)), "hits_sd": float(np.std(finals)),
                       "curve": np.mean(runs, 0)[:: max(1, len(runs[0]) // 200)].tolist()}
    return out

# ---------------------------------------------------------- explanation fidelity


def explanation_fidelity(model, d: Dataset, n_users: int = 600, ks=(1, 2, 3, 5, 10), seed: int = 0):
    """Score drop of the top recommendation after deleting k history movies."""
    assert model.cfg.backbone == "attn", "attention-based selection needs the attention backbone"
    rng = np.random.default_rng(seed)
    test = np.flatnonzero(d.split == 2)
    first = {}
    for e in test:
        first.setdefault(int(d.user[e]), e)
    events = np.array([e for e in first.values() if e - d.user_ptr[d.user[e]] >= max(ks) + 2])
    if len(events) > n_users:
        events = np.sort(rng.choice(events, n_users, replace=False))
    L = model.cfg.L
    model.eval()
    b = build_batch(d, events - 1, L)
    h, u, aux = model.forward(b, P=1)
    o = u[:, -1, :] if u is not None else h[:, -1, :]
    sc = model.score(o, aux["V"], np.arange(d.n_items), d.tau(d.week[events]), d.mu_rows(d.item[events - 1])).data
    _mask_seen(d, events, sc)
    top1 = sc.argmax(1)
    base = sc[np.arange(len(events)), top1]
    att = model.blocks[-1].att.last_attention[:, :, -1, :].mean(1)  # N, L
    att = np.where(b["valid"], att, -np.inf)
    out = {}
    for strategy in ("attention", "recency", "random"):
        drops = {}
        for k in ks:
            removed = []
            for r in range(len(events)):
                pos = np.flatnonzero(b["valid"][r])
                if strategy == "attention":
                    chosen = pos[np.argsort(-att[r, pos])[:k]]
                elif strategy == "recency":
                    chosen = pos[-k:]
                else:
                    chosen = rng.choice(pos, min(k, len(pos)), replace=False)
                removed.append(b["idx"][r, chosen])
            b2 = build_batch(d, events - 1, L, removed=removed)
            h2, u2, aux2 = model.forward(b2, P=1)
            o2 = u2[:, -1, :] if u2 is not None else h2[:, -1, :]
            prev2 = b2["items"][:, -1]
            cand = top1[:, None]
            tau = d.tau_pairs(d.week[events][:, None], cand)
            mu = d.mu_pairs(prev2[:, None], cand)
            new = model.score(o2, aux2["V"], cand, tau, mu).data[:, 0]
            diff = base - new
            ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
            drops[k] = {"mean": float(diff.mean()), "ci95": float(ci)}
        out[strategy] = drops
    model.train()
    out["users"] = int(len(events))
    return out
