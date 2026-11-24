"""End-to-end checks on synthetic data: preprocessing, gradients, training, evaluation."""
import numpy as np
import pytest

from nectar import autograd as ag
from nectar.baselines import SCORERS, train_bpr
from nectar.batching import build_batch
from nectar.data import Dataset, PrepConfig, prepare
from nectar.evaluate import evaluate_model, evaluate_scorer, sample_events
from nectar.model import PRESETS, SeqRecModel, make_config
from nectar.synthetic import make_movielens_like
from nectar.train import loss_on_batch, sample_windows


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    make_movielens_like(str(root / "raw"), n_users=200, n_items=150, seed=1)
    prepare(PrepConfig(raw_dir=str(root / "raw"), out_file=str(root / "data.npz"), min_train_item=3))
    return Dataset(str(root / "data.npz"))


def test_no_future_leakage(data):
    d = data
    # every feature of an interaction must be computable from that user's earlier events
    u = d.user
    first = np.r_[True, u[1:] != u[:-1]]
    assert np.all(d.ts[1:][~first[1:]] >= d.ts[:-1][~first[1:]])
    # popularity features at week w only count weeks before w
    w = int(d.week.max())
    assert np.allclose(d.pop_cum[w + 1] - d.pop_cum[w], np.bincount(d.item[d.week == w], minlength=d.n_items))
    # transitions come from training pairs only
    assert d.trans.sum() <= (d.split == 0).sum()


def test_full_model_gradient(data):
    ag.set_dtype(np.float64)
    try:
        cfg = make_config("NECTAR", d=8, L=6, P=3, heads=2, dropout=0.0, tipa_dim=4, asv_dim=4)
        model = SeqRecModel(cfg, data, seed=0)
        rng = np.random.default_rng(0)
        ends = sample_windows(data, rng, 1)[:4]
        b = build_batch(data, ends, cfg.L)
        negs = rng.integers(1, data.n_items, 5)
        params = [model.Wa.W, model.gates[1].W, model.Wr.W, model.conv_f.W, model.ctx_nodes.weight]
        for p in model.parameters():
            p.grad = None
        loss_on_batch(model, data, b, negs).backward()
        for p in params:
            idx = tuple(np.unravel_index(np.argmax(np.abs(p.grad)), p.shape))
            old = p.data[idx]
            p.data[idx] = old + 1e-6
            lp = float(loss_on_batch(model, data, b, negs).data)
            p.data[idx] = old - 1e-6
            lm = float(loss_on_batch(model, data, b, negs).data)
            p.data[idx] = old
            assert abs((lp - lm) / 2e-6 - p.grad[idx]) < 1e-5 * max(1.0, abs(p.grad[idx]))
    finally:
        ag.set_dtype(np.float32)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_every_model_runs(data, preset):
    cfg = make_config(preset, d=16, L=10, P=4)
    model = SeqRecModel(cfg, data, seed=0)
    rng = np.random.default_rng(0)
    b = build_batch(data, sample_windows(data, rng, 1)[:8], cfg.L)
    loss = loss_on_batch(model, data, b, rng.integers(1, data.n_items, 16))
    loss.backward()
    assert np.isfinite(loss.data)
    ev = sample_events(data, 2, 20)
    per_event, ranks, _ = evaluate_model(model, data, ev)
    assert ranks.min() >= 1 and np.all(np.isfinite(per_event["NDCG@10"]))


def test_baselines(data):
    ev = sample_events(data, 2, 50)
    for name, make in SCORERS.items():
        per_event, ranks = evaluate_scorer(make(data), data, ev)
        assert ranks.min() >= 1
    per_event, _ = evaluate_scorer(train_bpr(data, epochs=2, log=None), data, ev)
    assert np.isfinite(per_event["MRR"]).all()
