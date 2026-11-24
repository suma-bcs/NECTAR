"""Figures for the paper, drawn from the JSON results written by the experiment runner."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"NECTAR": "#2a6fdb", "other": "#9aa5b1"}


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=200)
    plt.close(fig)


def fig_data(d, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    q = (d.ts // (91 * 86400))
    for s, lab, c in ((0, "train", "#2a6fdb"), (1, "validation", "#f2a541"), (2, "test", "#d64545")):
        qs, cnt = np.unique(q[d.split == s], return_counts=True)
        ax[0].bar(qs, cnt, color=c, label=lab, width=1.0)
    for cut in d.cutoffs:
        ax[0].axvline(cut // (91 * 86400), color="k", ls="--", lw=0.8)
    ax[0].set_xlabel("quarter (index)")
    ax[0].set_ylabel("ratings")
    ax[0].legend()
    ax[0].set_title("(a) ratings per quarter")
    g = d.gap[d.gap < 5 * 365 * 86400]
    ax[1].hist(np.log10(np.maximum(g, 1)), bins=60, color="#2a6fdb")
    ax[1].set_xlabel("log10 gap between consecutive interactions (s)")
    ax[1].set_title("(b) inter-interaction gaps")
    _save(fig, out_dir, "fig_data.png")


def fig_overall(summary, out_dir):
    names = sorted(summary, key=lambda k: summary[k]["NDCG@10"]["mean"])
    vals = [summary[n]["NDCG@10"]["mean"] for n in names]
    lo = [summary[n]["NDCG@10"]["mean"] - summary[n]["NDCG@10"]["ci95"][0] for n in names]
    hi = [summary[n]["NDCG@10"]["ci95"][1] - summary[n]["NDCG@10"]["mean"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(names, vals, xerr=[lo, hi], color=[COLORS["NECTAR"] if n == "NECTAR" else COLORS["other"] for n in names])
    ax.set_xlabel("test NDCG@10 (95% bootstrap CI)")
    _save(fig, out_dir, "fig_overall.png")


def fig_curves(histories, out_dir):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name, hist in histories.items():
        ax.plot([h["epoch"] for h in hist], [h["val_ndcg10"] for h in hist], marker="o", ms=3, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NDCG@10")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "fig_curves.png")


def fig_ablation(ablation, out_dir):
    names = list(ablation)
    vals = [ablation[n]["NDCG@10"] for n in names]
    full = ablation[names[0]]["NDCG@10"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(names[::-1], vals[::-1], color=[COLORS["NECTAR"] if i == len(names) - 1 else COLORS["other"] for i in range(len(names))])
    for i, (n, v) in enumerate(zip(names[::-1], vals[::-1])):
        if n != names[0]:
            ax.text(v, i, f" {100 * (v - full) / full:+.1f}%", va="center", fontsize=8)
    ax.set_xlabel("test NDCG@10 (seed 0)")
    _save(fig, out_dir, "fig_ablation.png")


def fig_slices(slices, out_dir, models):
    groups = [("gap<=30min", "30min<gap<=30d", "gap>30d"), ("usual_hour", "unusual_hour"), ("no_tag_user", "tag_user")]
    titles = ["(a) gap before request", "(b) hour of request", "(c) tag history"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, grp, title in zip(axes, groups, titles):
        x = np.arange(len(grp))
        w = 0.8 / len(models)
        for j, m in enumerate(models):
            ax.bar(x + j * w, [slices[g][m] for g in grp], width=w, label=m)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([f"{g}\n(n={slices[g]['n']})" for g in grp], fontsize=7)
        ax.set_title(title)
    axes[0].set_ylabel("NDCG@10")
    axes[-1].legend(fontsize=7)
    _save(fig, out_dir, "fig_slices.png")


def fig_fusion(fusion, out_dir):
    if "combined_by_gap" not in fusion:
        return
    mods = fusion["modalities"]
    bins = list(fusion["combined_by_gap"])
    arr = np.array([fusion["combined_by_gap"][b] for b in bins])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, m in enumerate(mods):
        ax[0].plot(bins, arr[:, j], marker="o", label=m)
    ax[0].set_ylabel("mean alpha * gamma")
    ax[0].set_title("(a) by gap before request")
    ax[0].legend(fontsize=7)
    t = fusion["combined_by_tag"]
    x = np.arange(len(mods))
    for j, key in enumerate(("without_tags", "with_tags")):
        if t[key] is not None:
            ax[1].bar(x + 0.4 * j, t[key], width=0.4, label=key.replace("_", " "))
    ax[1].set_xticks(x + 0.2)
    ax[1].set_xticklabels(mods)
    ax[1].set_title("(b) by tag history")
    ax[1].legend(fontsize=7)
    _save(fig, out_dir, "fig_fusion.png")


def fig_eccm(fusion, out_dir):
    if "eccm_short" not in fusion:
        return
    a, b = np.array(fusion["eccm_short"]), np.array(fusion["eccm_long"])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for k, (m, t) in enumerate(((a, "(a) within 11 minutes"), (b, "(b) after > 18 hours"), (b - a, "(c) difference"))):
        im = ax[k].imshow(m, cmap="RdBu_r" if k == 2 else "Blues")
        ax[k].set_title(t)
        ax[k].set_xlabel("affect factor")
        ax[k].set_ylabel("context factor")
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    _save(fig, out_dir, "fig_eccm.png")


def fig_bandit(bandit, out_dir):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for p in ("greedy", "linucb", "egreedy", "thompson", "random"):
        c = bandit[p]["curve"]
        ax.plot(np.linspace(0, bandit["recommendations"], len(c)), c, label=p)
    ax.set_xlabel("recommendations")
    ax.set_ylabel("cumulative hits")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_bandit.png")


def fig_fidelity(fid, out_dir):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for s in ("attention", "recency", "random"):
        ks = sorted(int(k) for k in fid[s])
        m = [fid[s][str(k)]["mean"] if str(k) in fid[s] else fid[s][k]["mean"] for k in ks]
        e = [fid[s][str(k)]["ci95"] if str(k) in fid[s] else fid[s][k]["ci95"] for k in ks]
        ax.errorbar(ks, m, yerr=e, marker="o", capsize=3, label=s)
    ax.set_xlabel("history movies removed (k)")
    ax.set_ylabel("score drop of top recommendation")
    ax.legend()
    _save(fig, out_dir, "fig_fidelity.png")
