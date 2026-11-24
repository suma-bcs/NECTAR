#!/usr/bin/env python
"""Run every experiment in the paper and write tables, figures and a Markdown report.

Usage:
    python scripts/run_experiments.py --config configs/paper.yaml
    python scripts/run_experiments.py --config configs/smoke.yaml        # quick check
    python scripts/run_experiments.py --config configs/paper.yaml --stages train report

Runs are cached in <out>/runs/, so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nectar import plots  # noqa: E402
from nectar.analysis import (bandit_simulation, event_slices, explanation_fidelity,  # noqa: E402
                             fusion_summary, slice_table)
from nectar.baselines import SCORERS, train_bpr  # noqa: E402
from nectar.data import Dataset  # noqa: E402
from nectar.evaluate import evaluate_model, evaluate_scorer, sample_events, summarize  # noqa: E402
from nectar.model import SeqRecModel, make_config  # noqa: E402
from nectar.stats import bootstrap_ci, compare_to_reference, paired_wilcoxon  # noqa: E402
from nectar.train import train_model  # noqa: E402

TRAIN_KEYS = {"lr", "negatives", "batch_size", "windows_per_user"}
ALL_STAGES = ["baselines", "tune", "train", "ablation", "analysis", "bandit", "fidelity", "report"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out = cfg["out"]
        self.runs = os.path.join(self.out, "runs")
        os.makedirs(self.runs, exist_ok=True)
        self.d = Dataset(cfg["data"])
        self.val = sample_events(self.d, 1, cfg["n_val_events"], seed=11)
        self.tune_val = sample_events(self.d, 1, cfg["n_tune_events"], seed=22)
        self.test = sample_events(self.d, 2, cfg["n_test_events"], seed=33)
        log(f"data: {self.d.n_users} users, {self.d.n_items - 1} movies; "
            f"{len(self.val)} val / {len(self.tune_val)} tuning / {len(self.test)} test events")

    # -------------------------------------------------------------- helpers
    def _path(self, tag):
        return os.path.join(self.runs, tag)

    def _done(self, tag):
        return os.path.exists(self._path(tag) + ".json")

    def _save(self, tag, per_event, info, model=None):
        np.savez_compressed(self._path(tag) + ".npz", **per_event)
        with open(self._path(tag) + ".json", "w") as f:
            json.dump(info, f, indent=2)
        if model is not None:
            np.savez_compressed(self._path(tag) + "_weights.npz", **model.state_dict())

    def load(self, tag):
        z = np.load(self._path(tag) + ".npz")
        with open(self._path(tag) + ".json") as f:
            info = json.load(f)
        return {k: z[k] for k in z.files}, info

    def load_model(self, preset, seed, overrides=None):
        tag = f"{preset}_seed{seed}"
        _, info = self.load(tag)
        model = SeqRecModel(make_config(preset, **info.get("model_overrides", {})), self.d, seed=seed)
        w = np.load(self._path(tag) + "_weights.npz")
        model.load_state_dict({k: w[k] for k in w.files})
        return model

    def best_overrides(self, preset):
        path = os.path.join(self.out, "tuning.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            tuning = json.load(f)
        return tuning.get(preset, {}).get("best", {})

    def train_one(self, preset, seed, overrides=None, events=None, tag=None):
        overrides = dict(overrides or {})
        tag = tag or f"{preset}_seed{seed}"
        if self._done(tag):
            return
        model_kw = {k: v for k, v in overrides.items() if k not in TRAIN_KEYS}
        train_kw = {k: v for k, v in overrides.items() if k in TRAIN_KEYS}
        cfg = make_config(preset, **model_kw)
        model, hist, minutes = train_model(cfg, self.d, self.val, seed=seed, max_epochs=self.cfg["max_epochs"],
                                           patience=self.cfg["patience"], log=log, **train_kw)
        ev = self.test if events is None else events
        per_event, ranks, _ = evaluate_model(model, self.d, ev)
        info = {"model": preset, "seed": seed, "overrides": overrides, "model_overrides": model_kw,
                "summary": summarize(per_event), "history": hist, "train_minutes": minutes,
                "events": "test" if events is None else "tuning"}
        self._save(tag, per_event, info, model if events is None else None)
        log(f"{tag}: NDCG@10 {info['summary']['NDCG@10']:.4f} ({minutes:.1f} min)")

    # --------------------------------------------------------------- stages
    def stage_baselines(self):
        for name, make in SCORERS.items():
            tag = f"{name}_seed0"
            if not self._done(tag):
                per_event, _ = evaluate_scorer(make(self.d), self.d, self.test)
                self._save(tag, per_event, {"model": name, "seed": 0, "summary": summarize(per_event)})
                log(f"{name}: NDCG@10 {np.mean(per_event['NDCG@10']):.4f}")
        if not self._done("BPR-MF_seed0"):
            t = time.time()
            scorer = train_bpr(self.d, epochs=self.cfg.get("bpr_epochs", 30), log=None)
            per_event, _ = evaluate_scorer(scorer, self.d, self.test)
            self._save("BPR-MF_seed0", per_event, {"model": "BPR-MF", "seed": 0, "summary": summarize(per_event),
                                                   "train_minutes": (time.time() - t) / 60})
            log(f"BPR-MF: NDCG@10 {np.mean(per_event['NDCG@10']):.4f}")

    def stage_tune(self):
        results = {}
        for preset, trials in self.cfg["tuning"].items():
            scores = []
            for j, ov in enumerate(trials):
                tag = f"tune_{preset}_{j}"
                self.train_one(preset, 0, ov, events=self.tune_val, tag=tag)
                _, info = self.load(tag)
                scores.append({"overrides": ov, "val_ndcg10": info["summary"]["NDCG@10"]})
            best = max(scores, key=lambda s: s["val_ndcg10"])
            results[preset] = {"trials": scores, "best": best["overrides"]}
            log(f"tuning {preset}: best {best}")
        with open(os.path.join(self.out, "tuning.json"), "w") as f:
            json.dump(results, f, indent=2)

    def stage_train(self):
        for preset in self.cfg["multi_seed_models"]:
            for seed in self.cfg["seeds"]:
                self.train_one(preset, seed, self.best_overrides(preset))
        for preset in self.cfg["single_seed_models"]:
            self.train_one(preset, 0)

    def stage_ablation(self):
        for preset in self.cfg["ablations"]:
            self.train_one(preset, 0, self.best_overrides("NECTAR"))

    def stage_analysis(self):
        model = self.load_model("NECTAR", 0)
        per_event, _, analysis = evaluate_model(model, self.d, self.test, keep_analysis=True)
        fusion = fusion_summary(self.d, self.test, analysis)
        with open(os.path.join(self.out, "fusion.json"), "w") as f:
            json.dump(fusion, f, indent=2)

    def stage_bandit(self):
        model = self.load_model("NECTAR", 0)
        res = bandit_simulation(model, self.d, max_users=self.cfg["bandit_users"])
        with open(os.path.join(self.out, "bandit.json"), "w") as f:
            json.dump(res, f, indent=2)
        log(f"bandit: " + ", ".join(f"{p} {res[p]['hits_mean']:.0f}" for p in ("greedy", "linucb", "egreedy", "thompson", "random")))

    def stage_fidelity(self):
        model = self.load_model("NECTAR", 0)
        res = explanation_fidelity(model, self.d, n_users=self.cfg["fidelity_users"])
        with open(os.path.join(self.out, "fidelity.json"), "w") as f:
            json.dump(res, f, indent=2)

    # --------------------------------------------------------------- report
    def _model_events(self, name):
        """Per-event metrics averaged over the seeds that were run."""
        seeds = self.cfg["seeds"] if name in self.cfg["multi_seed_models"] else [0]
        runs = [self.load(f"{name}_seed{s}") for s in seeds if self._done(f"{name}_seed{s}")]
        if not runs:
            return None, None
        keys = runs[0][0].keys()
        avg = {k: np.mean([r[0][k] for r in runs], axis=0) for k in keys}
        return avg, runs

    def stage_report(self):
        names = (self.cfg["multi_seed_models"] + self.cfg["single_seed_models"] + list(SCORERS) + ["BPR-MF"])
        summary, per_event_all, histories, minutes = {}, {}, {}, {}
        for n in names:
            avg, runs = self._model_events(n)
            if avg is None:
                continue
            per_event_all[n] = avg
            row = {}
            for k in avg:
                vals = [r[1]["summary"][k] for r in runs]
                row[k] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)) if len(vals) > 1 else None,
                          "ci95": bootstrap_ci(avg[k], self.cfg.get("n_boot", 1000)) if k == "NDCG@10" else None}
            row["seeds"] = len(runs)
            summary[n] = row
            if "history" in runs[0][1]:
                histories[n] = runs[0][1]["history"]
                minutes[n] = float(np.mean([r[1]["train_minutes"] for r in runs]))
        ref = per_event_all["NECTAR"]["NDCG@10"]
        comparisons = compare_to_reference(ref, {n: v["NDCG@10"] for n, v in per_event_all.items() if n != "NECTAR"},
                                           self.cfg.get("n_boot", 1000))
        # memory-term effect on SASRec
        memory = {n: summary[n]["NDCG@10"]["mean"] for n in ("SASRec", "SASRec+T", "SASRec+T+M") if n in summary}
        # ablation (seed 0, paired against full seed 0)
        ablation = {}
        full0, _ = self.load("NECTAR_seed0")
        ablation["NECTAR (full, seed 0)"] = {"NDCG@10": float(full0["NDCG@10"].mean()), "HR@10": float(full0["HR@10"].mean())}
        for preset in self.cfg["ablations"]:
            if self._done(f"{preset}_seed0"):
                pe, _ = self.load(f"{preset}_seed0")
                ablation[preset] = {"NDCG@10": float(pe["NDCG@10"].mean()), "HR@10": float(pe["HR@10"].mean()),
                                    "change_pct": float(100 * (pe["NDCG@10"].mean() - full0["NDCG@10"].mean()) / full0["NDCG@10"].mean()),
                                    "p": paired_wilcoxon(full0["NDCG@10"], pe["NDCG@10"])}
        sl = event_slices(self.d, self.test)
        slice_models = [m for m in ("NECTAR", "Ctx-SASRec+T+M", "SASRec+T+M", "GRU4Rec+T+M", "Markov", "Pop-recent") if m in per_event_all]
        slices = slice_table({m: per_event_all[m] for m in slice_models}, sl)

        def jload(name):
            p = os.path.join(self.out, name)
            return json.load(open(p)) if os.path.exists(p) else None

        fusion, bandit, fidelity, tuning = jload("fusion.json"), jload("bandit.json"), jload("fidelity.json"), jload("tuning.json")
        stats_path = os.path.splitext(self.cfg["data"])[0] + "_stats.json"
        bench = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
        results = dict(benchmark=bench, summary=summary, comparisons=comparisons, memory=memory, ablation=ablation,
                       slices=slices, fusion=fusion, bandit=bandit, fidelity=fidelity, tuning=tuning,
                       train_minutes=minutes)
        with open(os.path.join(self.out, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        fig_dir = os.path.join(self.out, "figures")
        plots.fig_data(self.d, fig_dir)
        plots.fig_overall(summary, fig_dir)
        plots.fig_curves({k: histories[k] for k in histories if k in ("NECTAR", "Ctx-SASRec+T+M", "SASRec+T+M", "GRU4Rec+T+M")}, fig_dir)
        plots.fig_ablation(ablation, fig_dir)
        plots.fig_slices(slices, fig_dir, slice_models)
        if fusion:
            plots.fig_fusion(fusion, fig_dir)
            plots.fig_eccm(fusion, fig_dir)
        if bandit:
            plots.fig_bandit(bandit, fig_dir)
        if fidelity:
            plots.fig_fidelity(fidelity, fig_dir)
        write_markdown(results, os.path.join(self.out, "report.md"))
        log(f"report written to {os.path.join(self.out, 'report.md')}")


def write_markdown(r, path):
    L = ["# NECTAR results\n"]
    if r["benchmark"]:
        L.append("## Benchmark (paper Table 4)\n")
        L += [f"- {k}: {v}" for k, v in r["benchmark"].items()]
        L.append("")
    L.append("## Test results (paper Table 7)\n")
    cols = ["HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20", "MRR"]
    L.append("| Model | " + " | ".join(cols) + " |")
    L.append("|---" * (len(cols) + 1) + "|")
    for n, row in sorted(r["summary"].items(), key=lambda kv: -kv[1]["NDCG@10"]["mean"]):
        cells = [f"{row[c]['mean']:.4f}" + (f" ± {row[c]['sd']:.4f}" if row[c]["sd"] is not None else "") for c in cols]
        L.append(f"| {n} ({row['seeds']} seed{'s' if row['seeds'] > 1 else ''}) | " + " | ".join(cells) + " |")
    L.append("\n## NECTAR vs. each model (per-event NDCG@10, Wilcoxon, Holm-corrected)\n")
    L.append("| Model | its NDCG@10 | NECTAR change | 95% CI of difference | Holm p |")
    L.append("|---|---|---|---|---|")
    for n, c in r["comparisons"].items():
        L.append(f"| {n} | {c['other_mean']:.4f} | {c['rel_change_pct']:+.1f}% | [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] | {c['p_holm']:.3g} |")
    if r["memory"]:
        L.append("\n## Request-time memory (Section 6.1)\n")
        L += [f"- {k}: {v:.4f}" for k, v in r["memory"].items()]
    L.append("\n## Ablation (paper Table 8)\n")
    L.append("| Variant | NDCG@10 | HR@10 | Change | p |")
    L.append("|---|---|---|---|---|")
    for n, a in r["ablation"].items():
        L.append(f"| {n} | {a['NDCG@10']:.4f} | {a['HR@10']:.4f} | "
                 + (f"{a['change_pct']:+.1f}% | {a['p']:.3g} |" if "p" in a else "– | – |"))
    L.append("\n## Situation slices, NDCG@10 (Section 6.4)\n")
    models = [k for k in next(iter(r["slices"].values())) if k != "n"]
    L.append("| Slice | n | " + " | ".join(models) + " |")
    L.append("|---" * (len(models) + 2) + "|")
    for s, row in r["slices"].items():
        L.append(f"| {s} | {row['n']} | " + " | ".join(f"{row[m]:.4f}" for m in models) + " |")
    if r["fusion"]:
        f = r["fusion"]
        L.append("\n## Fusion weights (Section 6.5)\n")
        L.append(f"- modalities: {f['modalities']}")
        L.append(f"- mean attention alpha: {[round(x, 2) for x in f['alpha_mean']]}")
        if "gamma_mean" in f:
            L.append(f"- mean gate gamma: {[round(x, 2) for x in f['gamma_mean']]}")
    if r["bandit"]:
        b = r["bandit"]
        L.append(f"\n## Bandit simulation (Section 6.7): {b['users']} users, {b['recommendations']} recommendations\n")
        for p in ("greedy", "linucb", "egreedy", "thompson", "random"):
            L.append(f"- {p}: {b[p]['hits_mean']:.0f} ± {b[p]['hits_sd']:.0f} hits")
    if r["fidelity"]:
        fi = r["fidelity"]
        L.append(f"\n## Explanation fidelity (Section 6.8): {fi['users']} users\n")
        for s in ("attention", "recency", "random"):
            L.append(f"- {s}: " + ", ".join(f"k={k}: {v['mean']:.3f}" for k, v in fi[s].items()))
    if r["tuning"]:
        L.append("\n## Tuning (paper Table 5)\n")
        for m, t in r["tuning"].items():
            for tr in t["trials"]:
                L.append(f"- {m} {tr['overrides'] or 'default'}: {tr['val_ndcg10']:.4f}")
    if r["train_minutes"]:
        L.append("\n## Training time per run (minutes)\n")
        L += [f"- {k}: {v:.1f}" for k, v in r["train_minutes"].items()]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="*", default=ALL_STAGES, choices=ALL_STAGES)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    runner = Runner(cfg)
    for stage in ALL_STAGES:
        if stage in args.stages:
            log(f"== stage: {stage}")
            getattr(runner, f"stage_{stage}")()


if __name__ == "__main__":
    main()
