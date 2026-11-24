"""Paired significance tests and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, running, out = len(keys), 0.0, {}
    for rank, k in enumerate(keys):
        adj = min(1.0, (m - rank) * pvals[k])
        running = max(running, adj)
        out[k] = running
    return out


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a) - np.asarray(b)
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compare_to_reference(ref: np.ndarray, others: dict, n_boot: int = 1000) -> dict:
    """Per-event NDCG@10 of a reference model against several others, Holm-corrected."""
    raw = {k: paired_wilcoxon(ref, v) for k, v in others.items()}
    adj = holm(raw)
    out = {}
    for k, v in others.items():
        diff = ref - v
        out[k] = {
            "other_mean": float(np.mean(v)),
            "diff": float(diff.mean()),
            "rel_change_pct": float(100 * (ref.mean() - v.mean()) / max(v.mean(), 1e-12)),
            "ci95": bootstrap_ci(diff, n_boot),
            "p_raw": raw[k],
            "p_holm": adj[k],
        }
    return out
