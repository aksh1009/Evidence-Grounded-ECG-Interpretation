"""Step 4 - 1D xresnet for multi-label ECG classification (PTB-XL benchmark family).

Fully convolutional + adaptive pooling, so it accepts any input length (train on crops,
evaluate on the full 10 s). Input: (N, 12, T) standardised 100 Hz signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn(ci, co, k=5, stride=1, act=True, zero_bn=False):
    bn = nn.BatchNorm1d(co)
    nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
    layers = [nn.Conv1d(ci, co, k, stride=stride, padding=k // 2, bias=False), bn]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """Bottleneck residual block (expansion 4) with xresnet-style avg-pool shortcut."""

    def __init__(self, ci, nh, stride=1, k=5, expansion=4):
        super().__init__()
        co = nh * expansion
        self.convs = nn.Sequential(
            conv_bn(ci, nh, 1),
            conv_bn(nh, nh, k, stride=stride),
            conv_bn(nh, co, 1, act=False, zero_bn=True),
        )
        sc = []
        if stride != 1:
            sc.append(nn.AvgPool1d(stride, stride, ceil_mode=True))
        if ci != co:
            sc.append(conv_bn(ci, co, 1, act=False))
        self.shortcut = nn.Sequential(*sc)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.convs(x) + self.shortcut(x))


class XResNet1d(nn.Module):
    def __init__(self, n_classes: int, n_leads: int = 12, layers=(3, 4, 6, 3), width=(64, 128, 256, 512),
                 stem=(32, 32, 64), k=5, expansion=4, dropout=0.5):
        super().__init__()
        s = [conv_bn(n_leads if i == 0 else stem[i - 1], c, k, stride=2 if i == 0 else 1) for i, c in enumerate(stem)]
        self.stem = nn.Sequential(*s, nn.MaxPool1d(3, 2, 1))
        blocks, ci = [], stem[-1]
        for i, (n, w) in enumerate(zip(layers, width)):
            nh = w // expansion
            for j in range(n):
                blocks.append(ResBlock(ci, nh, stride=2 if (j == 0 and i > 0) else 1, k=k, expansion=expansion))
                ci = nh * expansion
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2 * ci), nn.Dropout(dropout), nn.Linear(2 * ci, ci // 2), nn.ReLU(inplace=True),
            nn.BatchNorm1d(ci // 2), nn.Dropout(dropout), nn.Linear(ci // 2, n_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def features(self, x):
        return self.body(self.stem(x))

    def forward(self, x):
        f = self.features(x)
        f = torch.cat([f.mean(-1), f.amax(-1)], dim=1)
        return self.head(f)
