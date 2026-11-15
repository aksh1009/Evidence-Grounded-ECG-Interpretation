"""Step 4b - Integrated Gradients attribution: which leads / which beat drove a CNN prediction.

Turns a class prediction into a citation: top leads + the time window (a detected beat when
available) with the most attribution mass.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import CANONICAL_LEADS


def integrated_gradients(model, x: torch.Tensor, class_idx: list[int], steps: int = 24,
                         baseline: torch.Tensor | None = None) -> np.ndarray:
    """x: (12, T) standardised input on the model's device. Returns (len(class_idx), 12, T)."""
    model.eval()
    x0 = torch.zeros_like(x) if baseline is None else baseline
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=x.device)[:, None, None]
    path = (x0 + alphas * (x - x0)).detach().requires_grad_(True)       # (steps, 12, T)
    out = []
    for c in class_idx:
        if path.grad is not None:
            path.grad = None
        logits = model(path)[:, c]
        g, = torch.autograd.grad(logits.sum(), path)
        out.append(((x - x0) * g.mean(0)).detach().cpu().numpy())
    return np.stack(out) if out else np.zeros((0,) + tuple(x.shape))


def summarize_attribution(attr: np.ndarray, fs: int, beats: list[dict] | None = None,
                          top_k_leads: int = 3) -> dict:
    """attr: (12, T) for one class -> {top_leads, lead_share, window_s, window_kind}."""
    a = np.abs(attr)
    lead_mass = a.sum(axis=1)
    share = lead_mass / (lead_mass.sum() + 1e-12)
    order = np.argsort(-share)[:top_k_leads]
    prof = a.sum(axis=0)
    T = len(prof)
    window, kind = None, "time"
    if beats:
        best, best_m = None, -1.0
        for b in beats:
            lo = b["qrs_s"][0] - 0.25                                     # include P wave
            hi = b["t_s"][1] if b["t_s"][1] is not None else b["qrs_s"][1] + 0.4
            i0, i1 = max(0, int(lo * fs)), min(T, int(hi * fs))
            if i1 - i0 < 5:
                continue
            m = prof[i0:i1].sum() / (i1 - i0) ** 0.5                     # mild length normalisation
            if m > best_m:
                best, best_m = (round(max(0.0, lo), 2), round(min(T / fs, hi), 2)), m
        if best:
            window, kind = best, "beat"
    if window is None:
        w = fs
        cs = np.convolve(prof, np.ones(w), mode="valid")
        i = int(np.argmax(cs))
        window = (round(i / fs, 2), round((i + w) / fs, 2))
    return {
        "top_leads": [CANONICAL_LEADS[i] for i in order],
        "lead_share": {CANONICAL_LEADS[i]: round(float(share[i]), 3) for i in order},
        "window_s": list(window),
        "window_kind": kind,
    }


@torch.no_grad()
def deletion_check(model, x: torch.Tensor, class_idx: int, lead_order: list[int]) -> dict:
    """Faithfulness probe: zero (= train mean) the top lead vs. a random lead, compare prob drop."""
    model.eval()
    p0 = torch.sigmoid(model(x[None]))[0, class_idx].item()
    xt = x.clone()
    xt[lead_order[0]] = 0
    p_top = torch.sigmoid(model(xt[None]))[0, class_idx].item()
    rnd = [i for i in range(12) if i != lead_order[0]][np.random.randint(11)]
    xr = x.clone()
    xr[rnd] = 0
    p_rnd = torch.sigmoid(model(xr[None]))[0, class_idx].item()
    return {"p": p0, "drop_top": p0 - p_top, "drop_random": p0 - p_rnd}
