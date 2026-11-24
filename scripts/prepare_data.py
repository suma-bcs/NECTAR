#!/usr/bin/env python
"""Preprocess MovieLens 25M (or synthetic data) into a single .npz file.

    python scripts/prepare_data.py --raw data/ml-25m --out data/ml25m.npz
    python scripts/prepare_data.py --synthetic --out data/synthetic.npz
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nectar.data import PrepConfig, prepare  # noqa: E402
from nectar.synthetic import make_movielens_like  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/ml-25m", help="folder with ratings.csv, tags.csv, movies.csv, genome-scores.csv")
    ap.add_argument("--out", default="data/ml25m.npz")
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic MovieLens-format data first")
    ap.add_argument("--max-users", type=int, default=None, help="optional cap on eligible users for quick runs")
    args = ap.parse_args()
    raw, min_item = args.raw, 10
    if args.synthetic:
        raw = os.path.join(os.path.dirname(args.out) or ".", "synthetic_raw")
        make_movielens_like(raw, n_users=600, n_items=500, seed=0, years=6.0, mean_sessions=14)
        min_item = 3
    stats = prepare(PrepConfig(raw_dir=raw, out_file=args.out, max_users=args.max_users, min_train_item=min_item))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
