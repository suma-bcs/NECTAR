"""Synthetic data in MovieLens 25M format, used for tests and smoke runs."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import GENRES

WORDS_POS = ["great", "funny", "classic", "beautiful", "masterpiece", "clever"]
WORDS_NEG = ["boring", "dark", "violent", "slow", "overrated", "sad"]
WORDS_NEU = ["visually", "space", "twist", "family", "music", "war", "heist", "robots"]


def make_movielens_like(out_dir: str, n_users=300, n_items=400, seed=0, start=1_200_000_000,
                        years=8.0, mean_sessions=12, genome=True) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    k = 8
    U = rng.normal(size=(n_users, k))
    V = rng.normal(size=(n_items, k))
    item_genres = [rng.choice(19, size=rng.integers(1, 4), replace=False) for _ in range(n_items)]
    pop = rng.zipf(1.6, size=n_items).astype(float)
    pop = np.minimum(pop, 200) / 200.0
    span = years * 365 * 86400
    rows, tag_rows = [], []
    for u in range(n_users):
        n_sess = max(2, rng.poisson(mean_sessions))
        t = start + rng.uniform(0, span * 0.6)
        seen = set()
        prev = None
        for _ in range(n_sess):
            t += rng.exponential(span / (n_sess * 3))
            for _ in range(rng.integers(1, 12)):
                t += rng.exponential(90)
                logits = U[u] @ V.T + 2 * pop
                if prev is not None:
                    logits = logits + 1.5 * (np.arange(n_items) % 50 == (prev + 1) % 50)
                logits[list(seen)] = -1e9
                p = np.exp(logits - logits.max())
                p /= p.sum()
                i = int(rng.choice(n_items, p=p))
                seen.add(i)
                prev = i
                score = U[u] @ V[i]
                rating = float(np.clip(np.round((3.5 + score / 3 + rng.normal(0, 0.7)) * 2) / 2, 0.5, 5.0))
                rows.append((u + 1, i + 1, rating, int(t)))
                if rng.random() < 0.08:
                    words = list(rng.choice(WORDS_POS if rating >= 4 else WORDS_NEG if rating <= 2 else WORDS_NEU, 2))
                    tag_rows.append((u + 1, i + 1, " ".join(words), int(t + rng.uniform(0, 3600))))
                if len(seen) >= n_items - 5:
                    break
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(
        os.path.join(out_dir, "ratings.csv"), index=False)
    pd.DataFrame(tag_rows, columns=["userId", "movieId", "tag", "timestamp"]).to_csv(
        os.path.join(out_dir, "tags.csv"), index=False)
    titles = [f"Movie {i + 1} ({1950 + int(rng.integers(0, 70))})" for i in range(n_items)]
    genres = ["|".join(GENRES[g] for g in gs) for gs in item_genres]
    pd.DataFrame({"movieId": np.arange(1, n_items + 1), "title": titles, "genres": genres}).to_csv(
        os.path.join(out_dir, "movies.csv"), index=False)
    if genome:
        n_tags = 40
        rel = 1 / (1 + np.exp(-(V @ rng.normal(size=(k, n_tags)))))
        mid, tid = np.meshgrid(np.arange(1, n_items + 1), np.arange(1, n_tags + 1), indexing="ij")
        pd.DataFrame({"movieId": mid.ravel(), "tagId": tid.ravel(), "relevance": rel.ravel()}).to_csv(
            os.path.join(out_dir, "genome-scores.csv"), index=False)
