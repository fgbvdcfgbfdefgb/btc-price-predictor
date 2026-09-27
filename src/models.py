"""Evolvable multi-horizon forecasting backbones.

Every model maps a window (batch, lookback, n_features) -> (batch, n_horizons)
of predicted *standardised log returns*.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    def __init__(self, n_feat, lookback, n_out, hidden, depth, dropout, act):
        super().__init__()
        A = ACTS[act]
        layers, d = [nn.Flatten()], n_feat * lookback
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), A(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GRUNet(nn.Module):
    def __init__(self, n_feat, lookback, n_out, hidden, depth, dropout, act):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=depth, batch_first=True,
                          dropout=dropout if depth > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), ACTS[act](),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1])


class _TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, dil, dropout, act):
        super().__init__()
        self.pad = (k - 1) * dil                      # causal padding
        self.conv = nn.Conv1d(cin, cout, k, dilation=dil)
        self.norm = nn.GroupNorm(1, cout)
        self.act = ACTS[act]()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return self.drop(self.act(self.norm(y))) + self.res(x)


class TCN(nn.Module):
    def __init__(self, n_feat, lookback, n_out, hidden, depth, dropout, act):
        super().__init__()
        self.blocks = nn.ModuleList([
            _TCNBlock(n_feat if i == 0 else hidden, hidden, 3, 2 ** i, dropout, act)
            for i in range(depth)])
        self.head = nn.Sequential(nn.Flatten(), nn.LayerNorm(hidden),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x):
        h = x.transpose(1, 2)
        for b in self.blocks:
            h = b(h)
        return self.head(h[:, :, -1])


class AttnNet(nn.Module):
    """Small causal transformer encoder over the lookback window."""

    def __init__(self, n_feat, lookback, n_out, hidden, depth, dropout, act):
        super().__init__()
        heads = max(1, min(8, hidden // 32))
        while hidden % heads:
            heads -= 1
        self.inp = nn.Linear(n_feat, hidden)
        self.pos = nn.Parameter(torch.randn(1, lookback, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=hidden * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_out))
        self.register_buffer(
            "mask", torch.triu(torch.ones(lookback, lookback, dtype=torch.bool), 1))

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos, mask=self.mask)
        return self.head(h[:, -1])


BACKBONES = {"mlp": MLP, "gru": GRUNet, "tcn": TCN, "attn": AttnNet}


def build_model(genome: dict, n_feat: int, n_out: int) -> nn.Module:
    cls = BACKBONES[genome["backbone"]]
    return cls(n_feat, genome["lookback"], n_out,
               genome["hidden"], genome["depth"],
               genome["dropout"], genome["act"])


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
