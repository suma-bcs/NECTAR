"""NECTAR and the neural baselines (one configurable class).

Stages (each can be switched off for ablation):
  backbone  time-aware causal transformer (or GRU for GRU4Rec)
  Stage 1   HCE  = GCE (graph propagation over 67 context nodes) + TCT
  Stage 2   BEEN = TIPA (GRU over behaviour) + SE (tag text) + ELM (affect state, valence head)
  Stage 3   ECCM (6x6 bilinear map) + TDM (bidirectional ConvLSTM, window 3)
  Stage 4   NFAM attention + GMAU reliability gates -> CUE
  Stage 5   TADRN score with temporal-popularity (T) and transition-prior (M) terms
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import autograd as ag
from . import nn
from .autograd import Tensor
from .batching import gap_features
from .data import N_CTX_NODES, Dataset


@dataclass
class ModelConfig:
    name: str = "NECTAR"
    backbone: str = "attn"  # "attn" or "gru"
    d: int = 64
    L: int = 30
    blocks: int = 2
    heads: int = 2
    dropout: float = 0.2
    gap_enc: bool = True
    use_tau: bool = True
    use_mu: bool = True
    early_fusion: bool = False  # Ctx-SASRec: side information added to the inputs
    fusion: bool = True  # False -> plain sequence model (u = h)
    hce: bool = True
    been: bool = True
    tdm: bool = True
    gmau: bool = True
    P: int = 8  # positions trained through the fusion head
    tipa_dim: int = 32
    asv_dim: int = 32
    eccm: int = 6
    conv_channels: int = 4
    window: int = 3
    lam: float = 0.1
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "NECTAR": dict(),
    "NECTAR-noHCE": dict(name="NECTAR-noHCE", hce=False),
    "NECTAR-noBEEN": dict(name="NECTAR-noBEEN", been=False),
    "NECTAR-noTDM": dict(name="NECTAR-noTDM", tdm=False),
    "NECTAR-noGMAU": dict(name="NECTAR-noGMAU", gmau=False),
    "Backbone+T+M": dict(name="Backbone+T+M", fusion=False),
    "SASRec": dict(name="SASRec", fusion=False, gap_enc=False, use_tau=False, use_mu=False),
    "SASRec+T": dict(name="SASRec+T", fusion=False, gap_enc=False, use_mu=False),
    "SASRec+T+M": dict(name="SASRec+T+M", fusion=False, gap_enc=False),
    "TiSASRec-style+T+M": dict(name="TiSASRec-style+T+M", fusion=False, gap_enc=True),
    "GRU4Rec+T+M": dict(name="GRU4Rec+T+M", fusion=False, backbone="gru", gap_enc=False),
    "Ctx-SASRec+T+M": dict(name="Ctx-SASRec+T+M", fusion=False, gap_enc=True, early_fusion=True),
}


def make_config(preset: str, **overrides) -> ModelConfig:
    kw = dict(PRESETS[preset])
    kw.update(overrides)
    return ModelConfig(**kw)


class SeqRecModel(nn.Module):
    def __init__(self, cfg: ModelConfig, data: Dataset, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.rng = rng
        d = cfg.d
        self.content = data.content.astype(np.float32)
        self.A_hat = data.A_hat.astype(np.float32)
        self.n_items = data.n_items
        # items
        self.item_emb = nn.Embedding(rng, data.n_items, d)
        self.Wf = nn.Linear(rng, self.content.shape[1], d, bias=False)
        self.Wf.W.data *= 0.1  # start content contributions small relative to the ID embedding
        self.pos_emb = nn.Embedding(rng, cfg.L, d)
        self.Wt = nn.Linear(rng, 8, d, bias=False)
        # backbone
        if cfg.backbone == "attn":
            self.blocks = [nn.TransformerBlock(rng, d, cfg.heads, cfg.dropout) for _ in range(cfg.blocks)]
        else:
            self.gru = nn.GRU(rng, d, d)
        self.ln_out = nn.LayerNorm(d)
        # early fusion (Ctx-SASRec)
        if cfg.early_fusion:
            self.ctx_plain = nn.Embedding(rng, N_CTX_NODES, d)
            self.Wb_in = nn.Linear(rng, 8, d, bias=False)
            self.Wz_in = nn.Linear(rng, data.tag_vec.shape[1], d, bias=False)
        # stage 1
        if cfg.fusion and cfg.hce:
            self.ctx_nodes = nn.Embedding(rng, N_CTX_NODES, d)
            self.Wt_ctx = nn.Linear(rng, 8, d, bias=False)
            self.tct = nn.TransformerBlock(rng, d, cfg.heads, cfg.dropout)
            self.ln_ctx = nn.LayerNorm(d)
        # stage 2
        if cfg.fusion and cfg.been:
            self.tipa = nn.GRU(rng, 8, cfg.tipa_dim)
            self.Ws = nn.Linear(rng, data.tag_vec.shape[1], cfg.tipa_dim, bias=False)
            self.Wa = nn.Linear(rng, 2 * cfg.tipa_dim, cfg.asv_dim)
            self.wv = nn.Linear(rng, cfg.asv_dim, 1)
            self.Wp = nn.Linear(rng, cfg.asv_dim, d, bias=False)
        # stage 3
        if cfg.fusion and cfg.tdm:
            self.s0 = nn.Parameter(rng.normal(0, 0.05, size=(d,)))
            self.a0 = nn.Parameter(rng.normal(0, 0.05, size=(cfg.asv_dim,)))
            self.Wc = nn.Linear(rng, d, cfg.eccm, bias=False)
            self.We = nn.Linear(rng, cfg.asv_dim, cfg.eccm, bias=False)
            self.conv_f = nn.ConvLSTM(rng, cfg.conv_channels, cfg.eccm)
            self.conv_b = nn.ConvLSTM(rng, cfg.conv_channels, cfg.eccm)
            self.Wr = nn.Linear(rng, 2 * cfg.conv_channels, d)
        # stage 4
        if cfg.fusion:
            self.Wh = nn.Linear(rng, d, d, bias=False)
            self.Wq = nn.Linear(rng, d, d)
            self.w_att = nn.Linear(rng, d, 1, bias=False)
            self.gates = [nn.Linear(rng, 2 * d + 3, d) for _ in range(4)]
            self.ln_cue = nn.LayerNorm(d)
        # stage 5
        self.W_tau = nn.Linear(rng, d, 3)
        self.w_mu = nn.Linear(rng, d, 1)
        self.analysis = {}

    # ------------------------------------------------------------------ pieces
    def item_table(self) -> Tensor:
        return self.item_emb.weight + self.Wf(Tensor(self.content))

    def _backbone(self, b, V):
        cfg = self.cfg
        B, L = b["items"].shape
        x = ag.getitem(V, b["items"]) + self.pos_emb.weight
        if cfg.gap_enc:
            x = x + self.Wt(Tensor(gap_features(b["gap"])))
        if cfg.early_fusion:
            ctx = self.ctx_plain(b["req_nodes"]).mean(axis=2)
            x = x + ctx + self.Wb_in(Tensor(b["behaviour"])) + self.Wz_in(Tensor(b["tag_vec"]))
        x = x * Tensor(b["valid"][..., None].astype(np.float32))
        x = ag.dropout(x, cfg.dropout, self.training, self.rng)
        if cfg.backbone == "attn":
            for blk in self.blocks:
                x = blk(x, b["valid"])
        else:
            x = self.gru(x)
        return self.ln_out(x)

    def _context(self, b):
        A = Tensor(self.A_hat)
        G0 = self.ctx_nodes.weight
        AG = ag.matmul(A, G0)
        G = (G0 + AG + ag.matmul(A, AG)) * (1.0 / 3.0)
        c = ag.getitem(G, b["req_nodes"]).mean(axis=2) + self.Wt_ctx(Tensor(gap_features(b["req_gap"])))
        return self.ln_ctx(self.tct(c, b["valid"]))

    def _affect(self, b):
        q = self.tipa(Tensor(b["behaviour"]))
        m = self.Ws(Tensor(b["tag_vec"])) * Tensor(b["tag_flag"][..., None])
        a = ag.tanh(self.Wa(ag.concat([q, m], axis=-1)))
        v_hat = self.wv(a).reshape(a.shape[0], a.shape[1])
        return a, v_hat

    def _drift(self, s, a, P):
        """DUR for the last P positions, using a window of ``w`` ECCMs ending at each."""
        cfg = self.cfg
        B, L, _ = s.shape
        w, E = cfg.window, cfg.eccm
        span = P + w - 1
        s_t, a_t = s[:, L - span:, :], a[:, L - span:, :]
        M = ag.sigmoid(ag.matmul(self.Wc(s_t).reshape(B, span, E, 1), self.We(a_t).reshape(B, span, 1, E)))
        Mf = M.reshape(B, span, E * E, 1)
        frames = [Mf[:, j:j + P].reshape(B * P, E * E, 1) for j in range(w)]
        hf = self.conv_f(frames)
        hb = self.conv_b(frames[::-1])
        r = self.Wr(ag.concat([hf.mean(axis=1), hb.mean(axis=1)], axis=-1)).reshape(B, P, -1)
        return r, M[:, w - 1:]

    # ----------------------------------------------------------------- forward
    def forward(self, b, P=None):
        """Returns (h for all positions, u for the last P positions or None, aux dict)."""
        cfg = self.cfg
        V = self.item_table()
        h = self._backbone(b, V)
        aux = {"V": V}
        if not cfg.fusion:
            return h, None, aux
        P = P or cfg.P
        B, L, d = h.shape
        hk = h[:, L - P:, :]
        mods, rho = [hk], [np.tile(np.array([0.0, 0.0, 1.0], np.float32), (B, P, 1))]
        s = a = None
        if cfg.hce:
            s = self._context(b)
            mods.append(s[:, L - P:, :])
            lg = (np.log1p(b["req_gap"][:, L - P:]) / 15.0).astype(np.float32)
            rho.append(np.stack([lg, np.zeros_like(lg), np.ones_like(lg)], -1))
        if cfg.been:
            a, v_hat = self._affect(b)
            aux["v_hat"] = v_hat
            mods.append(self.Wp(a[:, L - P:, :]))
            rho.append(np.stack([b["tag_share"][:, L - P:], b["abs_dev"][:, L - P:],
                                 np.ones((B, P), np.float32)], -1).astype(np.float32))
        if cfg.tdm:
            s_in = s if s is not None else self.s0 * Tensor(np.ones((B, L, 1), np.float32))
            a_in = a if a is not None else self.a0 * Tensor(np.ones((B, L, 1), np.float32))
            r, M = self._drift(s_in, a_in, P)
            mods.append(r)
            mm = M.data.mean(axis=(2, 3)).astype(np.float32)
            rho.append(np.stack([mm, np.zeros_like(mm), np.ones_like(mm)], -1))
            aux["M"] = M.data
        Hm = ag.stack(mods, axis=2)  # B,P,M,d
        score = self.w_att(ag.tanh(self.Wh(Hm) + self.Wq(hk).reshape(B, P, 1, d))).reshape(B, P, len(mods))
        alpha = ag.softmax(score, axis=-1)  # B,P,M
        fused = []
        gam = []
        for m_i in range(len(mods)):
            hm = Hm[:, :, m_i, :]
            if cfg.gmau:
                g = ag.sigmoid(self.gates[m_i](ag.concat([hm, hk, Tensor(rho[m_i])], axis=-1)))
                gam.append(g.data.mean(-1))
                hm = g * hm
            fused.append(hm * alpha[:, :, m_i:m_i + 1])
        total = fused[0]
        for f in fused[1:]:
            total = total + f
        u = self.ln_cue(hk + total)
        aux["alpha"] = alpha.data
        if gam:
            aux["gamma"] = np.stack(gam, -1)
        aux["modalities"] = ["sequence"] + (["context"] if cfg.hce else []) + \
                            (["affect"] if cfg.been else []) + (["drift"] if cfg.tdm else [])
        return h, u, aux

    # ------------------------------------------------------------------ scoring
    def score(self, o: Tensor, V: Tensor, cand: np.ndarray, tau: np.ndarray | None, mu: np.ndarray | None) -> Tensor:
        """Scores for candidate items. ``o``: (N, d); ``cand``: (N, K) or (K,) shared."""
        cfg = self.cfg
        if cand.ndim == 1:
            s = ag.matmul(o, ag.getitem(V, cand).transpose(1, 0))
        else:
            s = (o.reshape(o.shape[0], 1, o.shape[1]) * ag.getitem(V, cand)).sum(axis=-1)
        if cfg.use_tau and tau is not None:
            gt = self.W_tau(o)  # N,3
            s = s + (gt.reshape(gt.shape[0], 1, 3) * Tensor(tau)).sum(axis=-1)
        if cfg.use_mu and mu is not None:
            gm = self.w_mu(o)  # N,1
            s = s + gm * Tensor(mu)
        return s
