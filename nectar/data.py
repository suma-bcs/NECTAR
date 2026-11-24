"""MovieLens 25M preprocessing: global chronological split, filtering and features.

The output of :func:`prepare` is a single ``.npz`` file used by every model.
All features of an interaction are computed from that user's earlier events
(or from training data only, for fitted transforms), so nothing leaks from the
future into a prediction.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)",
]
GAP_EDGES = np.array([60, 300, 1800, 3600, 6 * 3600, 86400, 7 * 86400, 30 * 86400, 180 * 86400])  # 10 bins
SESSION_GAP = 1800  # seconds; a new session starts after 30 minutes of inactivity
N_HOUR, N_WD, N_GAP, N_POS, N_GEN = 24, 7, 10, 6, len(GENRES)
N_CTX_NODES = N_HOUR + N_WD + N_GAP + N_POS + N_GEN  # 67
WEEK = 7 * 86400


@dataclass
class PrepConfig:
    raw_dir: str
    out_file: str
    train_quantile: float = 0.85
    val_quantile: float = 0.925
    min_train_user: int = 10
    min_train_item: int = 10
    tag_window: int = 86400
    tag_svd_dim: int = 24
    genome_pca_dim: int = 32
    max_users: int | None = None  # optional cap for quick experiments
    seed: int = 0


def session_pos_bin(pos: np.ndarray) -> np.ndarray:
    return np.digitize(pos, [1, 2, 3, 6, 11])  # 0,1,2,3-5,6-10,11+


def gap_bin(gap: np.ndarray) -> np.ndarray:
    return np.digitize(gap, GAP_EDGES)


def _valence_fn():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        return lambda text: sia.polarity_scores(text)["compound"]
    except ImportError:  # small fallback lexicon so the pipeline still runs
        print("[warn] vaderSentiment not installed: using a small fallback lexicon")
        pos = {"good", "great", "funny", "beautiful", "love", "amazing", "masterpiece", "fun", "best",
               "classic", "excellent", "brilliant", "awesome", "touching", "cute", "clever"}
        neg = {"bad", "boring", "awful", "worst", "stupid", "sad", "violent", "dark", "disturbing",
               "overrated", "slow", "predictable", "terrible", "depressing", "gore", "weak"}

        def score(text):
            words = re.findall(r"[a-z]+", text.lower())
            s = sum(w in pos for w in words) - sum(w in neg for w in words)
            return float(np.tanh(s / 2.0))

        return score


def load_raw(raw_dir: str):
    r = pd.read_csv(os.path.join(raw_dir, "ratings.csv"))
    m = pd.read_csv(os.path.join(raw_dir, "movies.csv"))
    t = pd.read_csv(os.path.join(raw_dir, "tags.csv"))
    gpath = os.path.join(raw_dir, "genome-scores.csv")
    g = pd.read_csv(gpath) if os.path.exists(gpath) else None
    return r, m, t, g


def prepare(cfg: PrepConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    ratings, movies, tags, genome = load_raw(cfg.raw_dir)

    # ---- global cut-offs over all rating timestamps
    cut1 = int(np.quantile(ratings.timestamp.values, cfg.train_quantile))
    cut2 = int(np.quantile(ratings.timestamp.values, cfg.val_quantile))

    # ---- eligible users: >= min_train_user ratings before cut1 and >= 1 after
    before = ratings[ratings.timestamp < cut1].groupby("userId").size()
    after = ratings[ratings.timestamp >= cut1].groupby("userId").size()
    eligible = before[before >= cfg.min_train_user].index.intersection(after.index)
    if cfg.max_users is not None and len(eligible) > cfg.max_users:
        eligible = pd.Index(rng.choice(eligible.values, cfg.max_users, replace=False))
    df = ratings[ratings.userId.isin(eligible)].copy()

    # ---- items with >= min_train_item training interactions
    item_train = df[df.timestamp < cut1].groupby("movieId").size()
    keep_items = item_train[item_train >= cfg.min_train_item].index
    df = df[df.movieId.isin(keep_items)]
    ntr = df[df.timestamp < cut1].groupby("userId").size()
    df = df[df.userId.isin(ntr[ntr >= 2].index)]
    df = df.sort_values(["userId", "timestamp", "movieId"], kind="mergesort").reset_index(drop=True)

    users = np.sort(df.userId.unique())
    items = np.sort(df.movieId.unique())
    uid = {u: i for i, u in enumerate(users)}
    iid = {m: i + 1 for i, m in enumerate(items)}  # 0 is padding
    df["u"] = df.userId.map(uid).astype(np.int64)
    df["i"] = df.movieId.map(iid).astype(np.int64)
    df["split"] = np.where(df.timestamp < cut1, 0, np.where(df.timestamp < cut2, 1, 2)).astype(np.int8)
    n_items = len(items) + 1

    # ---- tags written within tag_window after the rating
    tg = tags[tags.userId.isin(eligible) & tags.movieId.isin(keep_items)][["userId", "movieId", "tag", "timestamp"]]
    tg = tg.merge(df[["userId", "movieId", "timestamp"]], on=["userId", "movieId"], suffixes=("_tag", ""))
    tg = tg[(tg.timestamp_tag >= tg.timestamp) & (tg.timestamp_tag <= tg.timestamp + cfg.tag_window)]
    tg["tag"] = tg["tag"].astype(str)
    tagtext = tg.groupby(["userId", "movieId"])["tag"].apply(lambda s: " ".join(s)).rename("tagtext")
    df = df.merge(tagtext, left_on=["userId", "movieId"], right_index=True, how="left")
    df["tagtext"] = df["tagtext"].fillna("")
    has_tag = (df.tagtext.str.len() > 0).values

    valence = np.zeros(len(df), dtype=np.float32)
    vfn = _valence_fn()
    valence[has_tag] = [vfn(t) for t in df.tagtext.values[has_tag]]

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    tag_vec = np.zeros((len(df), cfg.tag_svd_dim), dtype=np.float32)
    train_tag = has_tag & (df.split.values == 0)
    if train_tag.sum() > cfg.tag_svd_dim + 1:
        vec = TfidfVectorizer(min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform(df.tagtext.values[train_tag])
        k = min(cfg.tag_svd_dim, Xtr.shape[1] - 1)
        svd = TruncatedSVD(n_components=k, random_state=cfg.seed).fit(Xtr)
        Z = svd.transform(vec.transform(df.tagtext.values[has_tag]))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        tag_vec[has_tag, :k] = Z

    # ---- item content: genres, year, genome PCA, genome flag
    mv = movies.set_index("movieId").loc[items]
    gen = np.zeros((n_items, N_GEN), dtype=np.float32)
    for row, gs in enumerate(mv.genres.values, start=1):
        for gname in str(gs).split("|"):
            if gname in GENRES:
                gen[row, GENRES.index(gname)] = 1.0
    years = mv.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float).values
    years = np.where(np.isnan(years), np.nanmedian(years), years)
    year = np.zeros((n_items, 1), dtype=np.float32)
    year[1:, 0] = (years - 1990.0) / 30.0
    gpca = np.zeros((n_items, cfg.genome_pca_dim), dtype=np.float32)
    gflag = np.zeros((n_items, 1), dtype=np.float32)
    if genome is not None:
        genome = genome[genome.movieId.isin(items)]
        if len(genome):
            gm = genome.pivot(index="movieId", columns="tagId", values="relevance").fillna(0.0)
            from sklearn.decomposition import PCA

            k = min(cfg.genome_pca_dim, gm.shape[0] - 1, gm.shape[1])
            comp = PCA(n_components=k, random_state=cfg.seed).fit_transform(gm.values)
            comp /= comp.std(axis=0, keepdims=True) + 1e-8
            rows = np.array([iid[m] for m in gm.index])
            gpca[rows, :k] = comp
            gflag[rows, 0] = 1.0
    content = np.concatenate([gen, year, gpca, gflag], axis=1)

    # ---- per-interaction context and behaviour (from the user's earlier events only)
    u = df.u.values
    ts = df.timestamp.values.astype(np.int64)
    first = np.r_[True, u[1:] != u[:-1]]
    gap = np.r_[0, np.diff(ts)].astype(np.float64)
    gap[first] = 10 * 365 * 86400.0
    new_sess = first | (gap > SESSION_GAP)
    sess_id = np.cumsum(new_sess)
    sess_start = np.maximum.accumulate(np.where(new_sess, np.arange(len(df)), 0))
    sess_pos = np.arange(len(df)) - sess_start
    dt = pd.to_datetime(ts, unit="s", utc=True)
    hour = dt.hour.values.astype(np.int64)
    wday = dt.weekday.values.astype(np.int64)

    r = df.rating.values.astype(np.float64)
    g = df.groupby("u")
    csum = g.rating.cumsum().values - r
    csq = (df.rating ** 2).groupby(df.u).cumsum().values - r ** 2
    cnt = g.cumcount().values.astype(np.float64)
    mean_prev = np.where(cnt > 0, csum / np.maximum(cnt, 1), 3.5)
    var_prev = np.where(cnt > 1, csq / np.maximum(cnt, 1) - mean_prev ** 2, 1.0)
    dev = (r - mean_prev) / 2.0
    vol = np.sqrt(np.maximum(var_prev, 0.0)) / 2.0
    gi = gen[df.i.values]
    prev_g = np.vstack([np.zeros((1, N_GEN)), gi[:-1]])
    inter = (gi * prev_g).sum(1)
    union = np.maximum(((gi + prev_g) > 0).sum(1), 1)
    gswitch = np.where(first, 0.0, 1.0 - inter / union)
    behaviour = np.stack([
        dev, np.abs(dev), vol, np.log1p(gap) / 15.0, gswitch,
        np.log1p(sess_pos) / 3.0, has_tag.astype(float), valence,
    ], axis=1).astype(np.float32)

    # ---- request-time memory: weekly item counts and training transitions
    t0 = int(ts.min()) // WEEK
    week = (ts // WEEK - t0).astype(np.int64)
    n_weeks = int(week.max()) + 2
    weekly = np.zeros((n_weeks, n_items), dtype=np.float32)
    np.add.at(weekly, (week, df.i.values), 1.0)
    cum = np.vstack([np.zeros((1, n_items), dtype=np.float32), np.cumsum(weekly, axis=0)])  # cum[w] = weeks < w

    tr = df.split.values == 0
    same_user_next = np.r_[u[1:] == u[:-1], False]
    tr_pair = tr & same_user_next & np.r_[tr[1:], False]
    src = df.i.values[:-1][tr_pair[:-1]]
    dst = df.i.values[1:][tr_pair[:-1]]
    trans = sparse.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n_items, n_items))
    trans.sum_duplicates()

    # ---- context graph over 67 nodes from training interactions
    offs = np.cumsum([0, N_HOUR, N_WD, N_GAP, N_POS])
    nodes = np.stack([hour + offs[0], wday + offs[1], gap_bin(gap) + offs[2], session_pos_bin(sess_pos) + offs[3]], 1)[tr]
    A = np.zeros((N_CTX_NODES, N_CTX_NODES), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            if a != b:
                np.add.at(A, (nodes[:, a], nodes[:, b]), 1.0)
    gtr = gi[tr]
    for a in range(4):
        cnts = np.zeros((N_CTX_NODES, N_GEN))
        np.add.at(cnts, nodes[:, a], gtr)
        A[:, offs[4]:offs[4] + N_GEN] += cnts
        A[offs[4]:offs[4] + N_GEN, :] += cnts.T
    A = np.log1p(A)
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    A_hat = (dinv[:, None] * A * dinv[None, :]).astype(np.float32)

    user_ptr = np.r_[0, np.flatnonzero(np.diff(u)) + 1, len(u)].astype(np.int64)

    out = dict(
        user=u, item=df.i.values.astype(np.int64), ts=ts, rating=r.astype(np.float32),
        split=df.split.values, hour=hour, wday=wday, gap=gap.astype(np.float64),
        gap_bin=gap_bin(gap), sess_pos=session_pos_bin(sess_pos), sess_id=sess_id,
        behaviour=behaviour, tag_flag=has_tag.astype(np.float32), valence=valence,
        tag_vec=tag_vec, content=content, genres=gen, week=week, pop_cum=cum,
        trans_data=trans.data, trans_indices=trans.indices, trans_indptr=trans.indptr,
        A_hat=A_hat, user_ptr=user_ptr, n_items=np.int64(n_items),
        cutoffs=np.array([cut1, cut2], dtype=np.int64), movie_ids=items,
    )
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out_file)), exist_ok=True)
    np.savez_compressed(cfg.out_file, **out)
    stats = dict(
        users=int(len(users)), movies=int(n_items - 1), interactions=int(len(df)),
        train=int((df.split == 0).sum()), val=int((df.split == 1).sum()), test=int((df.split == 2).sum()),
        cut_train=str(pd.to_datetime(cut1, unit="s")), cut_val=str(pd.to_datetime(cut2, unit="s")),
        tagged_interactions=int(has_tag.sum()), users_with_tags=int(df.u[has_tag].nunique()),
        movies_with_genome=int(gflag.sum()),
    )
    with open(os.path.splitext(cfg.out_file)[0] + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


class Dataset:
    """In-memory view of a prepared ``.npz`` file."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        for k in z.files:
            setattr(self, k, z[k])
        self.n_items = int(self.n_items)
        self.trans = sparse.csr_matrix((self.trans_data, self.trans_indices, self.trans_indptr),
                                       shape=(self.n_items, self.n_items))
        self.n_users = len(self.user_ptr) - 1
        # context node ids of each interaction (as a request)
        offs = np.cumsum([0, N_HOUR, N_WD, N_GAP, N_POS])
        self.ctx_nodes = np.stack([self.hour + offs[0], self.wday + offs[1],
                                   self.gap_bin + offs[2], self.sess_pos + offs[3]], 1).astype(np.int64)

    def tau(self, week: np.ndarray, items=None) -> np.ndarray:
        """Recent-popularity features log(1+n)/4 for windows of 1, 4 and 13 weeks before ``week``.

        Returns shape (len(week), n_items or len(items), 3)."""
        w = np.asarray(week)
        cols = slice(None) if items is None else items
        feats = []
        for span in (1, 4, 13):
            lo = np.maximum(w - span + 0, 0)
            c = self.pop_cum[w][:, cols] - self.pop_cum[lo][:, cols]
            feats.append(np.log1p(c) / 4.0)
        return np.stack(feats, axis=-1).astype(np.float32)

    def tau_pairs(self, week: np.ndarray, items: np.ndarray) -> np.ndarray:
        """tau for matching arrays of weeks and items (same shape). Returns (..., 3)."""
        feats = []
        for span in (1, 4, 13):
            lo = np.maximum(week - span, 0)
            c = self.pop_cum[week, items] - self.pop_cum[lo, items]
            feats.append(np.log1p(c) / 4.0)
        return np.stack(feats, axis=-1).astype(np.float32)

    def mu_rows(self, prev_items: np.ndarray) -> np.ndarray:
        """Transition prior log(1+c)/2 from each previous item to all items (dense rows)."""
        return (np.log1p(self.trans[prev_items].toarray()) / 2.0).astype(np.float32)

    def mu_pairs(self, prev_items, items, loo_mask=None) -> np.ndarray:
        """Transition prior for matching (prev, item) arrays; ``loo_mask`` removes one count."""
        c = np.asarray(self.trans[prev_items.ravel(), items.ravel()]).reshape(items.shape)
        if loo_mask is not None:
            c = np.maximum(c - loo_mask, 0)
        return (np.log1p(c) / 2.0).astype(np.float32)
