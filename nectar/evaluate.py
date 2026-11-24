"""Successive, full-catalogue evaluation.

Each sampled event is predicted from all of the user's earlier interactions
(which may include earlier validation or test events, never later ones), and
every movie the user has not yet rated is ranked.
"""
from __future__ import annotations

import numpy as np

from .batching import build_batch
from .data import Dataset

KS = (5, 10, 20)


def sample_events(d: Dataset, split: int, n: int, seed: int = 0) -> np.ndarray:
    cand = np.flatnonzero(d.split == split)
    cand = cand[cand > 0]
    cand = cand[d.user[cand - 1] == d.user[cand]]
    rng = np.random.default_rng(seed)
    if len(cand) > n:
        cand = rng.choice(cand, n, replace=False)
    return np.sort(cand)


def _mask_seen(d: Dataset, events: np.ndarray, scores: np.ndarray) -> None:
    scores[:, 0] = -np.inf
    for r, e in enumerate(events):
        start = d.user_ptr[d.user[e]]
        scores[r, d.item[start:e]] = -np.inf


def ranks_from_scores(d: Dataset, events: np.ndarray, scores: np.ndarray) -> np.ndarray:
    _mask_seen(d, events, scores)
    target = d.item[events]
    ts = scores[np.arange(len(events)), target]
    return 1 + (scores > ts[:, None]).sum(axis=1)


def metrics_from_ranks(ranks: np.ndarray) -> dict:
    out = {}
    for k in KS:
        hit = ranks <= k
        out[f"HR@{k}"] = hit.astype(np.float64)
        out[f"NDCG@{k}"] = np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)
    out["MRR"] = 1.0 / ranks
    return out


def summarize(per_event: dict) -> dict:
    return {k: float(np.mean(v)) for k, v in per_event.items()}


def neural_scores(model, d: Dataset, events: np.ndarray, batch: int = 256, keep_analysis: bool = False):
    """Full-catalogue scores for a neural model. Returns (scores, analysis or None)."""
    model.eval()
    all_items = np.arange(d.n_items)
    out = np.zeros((len(events), d.n_items), dtype=np.float32)
    analysis = {"alpha": [], "gamma": [], "M": [], "attention": []} if keep_analysis else None
    for s in range(0, len(events), batch):
        ev = events[s:s + batch]
        b = build_batch(d, ev - 1, model.cfg.L)
        h, u, aux = model.forward(b, P=1)
        o = u[:, -1, :] if u is not None else h[:, -1, :]
        weeks = d.week[ev]
        prev = d.item[ev - 1]
        sc = model.score(o, aux["V"], all_items, d.tau(weeks), d.mu_rows(prev))
        out[s:s + len(ev)] = sc.data
        if keep_analysis:
            for k in ("alpha", "gamma", "M"):
                if k in aux:
                    analysis[k].append(aux[k][:, -1])
            if model.cfg.backbone == "attn":
                analysis["attention"].append(model.blocks[-1].att.last_attention[:, :, -1, :].mean(1))
    if keep_analysis:
        analysis = {k: np.concatenate(v) for k, v in analysis.items() if v}
        analysis["modalities"] = aux.get("modalities", [])
    model.train()
    return out, analysis


def evaluate_model(model, d: Dataset, events: np.ndarray, batch: int = 256, keep_analysis: bool = False):
    scores, analysis = neural_scores(model, d, events, batch, keep_analysis)
    ranks = ranks_from_scores(d, events, scores)
    return metrics_from_ranks(ranks), ranks, analysis


def evaluate_scorer(scorer, d: Dataset, events: np.ndarray, batch: int = 512):
    """Evaluate a non-neural scorer: ``scorer(d, events) -> (N, n_items)`` scores."""
    ranks = []
    for s in range(0, len(events), batch):
        ev = events[s:s + batch]
        ranks.append(ranks_from_scores(d, ev, scorer(d, ev).astype(np.float64)))
    ranks = np.concatenate(ranks)
    return metrics_from_ranks(ranks), ranks
