# NECTAR

**NECTAR: Dissecting Graph–Transformer Fusion of Context, Affect and Temporal Memory for Leakage-Free Sequential Movie Recommendation**

This repository contains the full implementation and experiment pipeline for the NECTAR testbed: a modular sequential recommender whose stages can be switched off one at a time, so that the contribution of each stage can be measured under a leakage-free protocol on MovieLens 25M.

| Stage | Component | What it does |
|---|---|---|
| Backbone | time-aware causal transformer | item history with gap encoding |
| 1 | HCE = GCE + TCT | graph propagation over 67 context nodes + temporal context transformer |
| 2 | BEEN = TIPA + SE + ELM | latent behavioural affect state from 8 behaviour features and tag text |
| 3 | ECCM + TDM | 6×6 emotion–context map and bidirectional ConvLSTM drift model |
| 4 | NFAM + GMAU | attention over modalities with reliability gates |
| 5 | TADRN | dot-product score + recent popularity (T) + transition prior (M) |

Everything is written in plain NumPy with a small reverse-mode automatic-differentiation engine (`nectar/autograd.py`); no deep-learning framework is needed and all experiments run on a CPU.

## Installation

```bash
git clone https://github.com/<your-account>/nectar.git
cd nectar
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                     # 20 tests, about 5 seconds
```

Python 3.10 or newer is required.

## Data

MovieLens 25M is not redistributed here (its licence does not allow it). Download it from GroupLens and unzip it into `data/`:

```bash
mkdir -p data && cd data
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip && cd ..
python scripts/prepare_data.py --raw data/ml-25m --out data/ml25m.npz
```

Preprocessing applies the protocol of the paper: global cut-offs at the 85th and 92.5th percentiles of all rating timestamps; users with at least ten ratings before the first cut-off and at least one after it; movies with at least ten training interactions; tags attached when written within one day after the rating. Statistics are written to `data/ml25m_stats.json`.

## Quick check (synthetic data, a few minutes)

```bash
python scripts/prepare_data.py --synthetic --out data/synthetic.npz
python scripts/run_experiments.py --config configs/smoke.yaml
```

This runs every stage of the pipeline on a small synthetic dataset in MovieLens format and writes `results/smoke/report.md`. The numbers are meaningless; the run only shows that everything works.

## Reproducing the paper

```bash
python scripts/run_experiments.py --config configs/paper.yaml
```

Stages can be run separately and are cached in `results/paper/runs/`, so an interrupted run resumes where it stopped:

```bash
python scripts/run_experiments.py --config configs/paper.yaml --stages baselines tune
python scripts/run_experiments.py --config configs/paper.yaml --stages train ablation
python scripts/run_experiments.py --config configs/paper.yaml --stages analysis bandit fidelity report
```

| Stage | Output | Paper |
|---|---|---|
| `baselines` | Pop, Pop-recent, Markov, BPR-MF | Table 7 |
| `tune` | 6 trials each for NECTAR and Ctx-SASRec+T+M on 6,000 validation events | Table 5 |
| `train` | NECTAR and Ctx-SASRec+T+M with 3 seeds; other neural baselines with seed 0 | Table 7, Figs. 4–5 |
| `ablation` | five ablated variants (seed 0) | Table 8, Fig. 6 |
| `analysis` | fusion weights and ECCM averages | Figs. 8–9 |
| `bandit` | LinUCB, linear Thompson sampling, ε-greedy, greedy, random | Fig. 10 |
| `fidelity` | attention / recency / random deletion test | Fig. 11 |
| `report` | `results.json`, `report.md`, figures, Wilcoxon + Holm tests, bootstrap CIs, slices | all |

The full run trains 27 neural models on about 3.3 million interactions, so it takes many hours on a CPU. Training time per run is recorded in the report.

### Note on reproducibility

Results depend on the random seeds, the library versions and the hardware. Please report the numbers produced by this code (in `results/paper/report.md`) in any paper that uses it.

## Repository structure

```
nectar/
  autograd.py     reverse-mode autodiff on NumPy
  nn.py           Linear, Embedding, LayerNorm, attention, GRU, ConvLSTM, Adam
  data.py         MovieLens preprocessing, features, memory terms, context graph
  batching.py     padded batches with request-time features
  model.py        NECTAR (all stages, ablation flags) and the neural baselines
  baselines.py    Pop, Pop-recent, Markov, BPR-MF
  train.py        sampled-softmax training with shared negatives, early stopping
  evaluate.py     successive full-catalogue evaluation, HR/NDCG/MRR
  stats.py        Wilcoxon signed-rank with Holm correction, bootstrap CIs
  analysis.py     situation slices, fusion weights, bandit simulation, explanation fidelity
  plots.py        figures
  synthetic.py    synthetic MovieLens-format data for tests
scripts/
  prepare_data.py
  run_experiments.py
configs/
  paper.yaml      full reproduction
  smoke.yaml      quick end-to-end check
tests/            gradient checks and end-to-end tests
```

## Model presets

`nectar.model.PRESETS` defines every model in the paper: `NECTAR`, `NECTAR-noHCE`, `NECTAR-noBEEN`, `NECTAR-noTDM`, `NECTAR-noGMAU`, `Backbone+T+M`, `SASRec`, `SASRec+T`, `SASRec+T+M`, `TiSASRec-style+T+M`, `GRU4Rec+T+M` and `Ctx-SASRec+T+M`. To train one model directly:

```python
from nectar.data import Dataset
from nectar.evaluate import sample_events, evaluate_model, summarize
from nectar.model import make_config
from nectar.train import train_model

d = Dataset("data/ml25m.npz")
val, test = sample_events(d, 1, 1500, seed=11), sample_events(d, 2, 20000, seed=33)
model, history, minutes = train_model(make_config("NECTAR"), d, val, seed=0)
per_event, ranks, _ = evaluate_model(model, d, test)
print(summarize(per_event))
```

## Citation

If you use this code, please cite the paper (details will be added on publication) and the MovieLens dataset:

> F. M. Harper and J. A. Konstan. The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems* 5(4), 2015.

## Licence

MIT (see `LICENSE`). The MovieLens data are subject to the GroupLens licence.
