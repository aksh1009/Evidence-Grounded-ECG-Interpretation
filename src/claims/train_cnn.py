"""Step 4 - CNN training, cross-fitted out-of-fold predictions, threshold tuning."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

from .model import XResNet1d
from .signal import Standardizer, prepare_batch_for_cnn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- data
def build_cnn_input_cache(sig_uv: np.ndarray, path: Path, fs_out: int = 100, chunk: int = 512) -> np.ndarray:
    """(N, 5000, 12) int16 uV -> (N, 12, 1000) float32 mV band-passed at 100 Hz, cached."""
    if path.exists():
        return np.load(path, mmap_mode="r")
    n = len(sig_uv)
    T = sig_uv.shape[1] * fs_out // 500
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, 12, T))
    for i in range(0, n, chunk):
        arr[i:i + chunk] = prepare_batch_for_cnn(sig_uv[i:i + chunk], 500, fs_out)
    arr.flush()
    return np.load(path, mmap_mode="r")


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, dict]:
    per = {}
    for j in range(y.shape[1]):
        if 0 < y[:, j].sum() < len(y):
            per[j] = roc_auc_score(y[:, j], p[:, j])
    return float(np.mean(list(per.values()))), per


# --------------------------------------------------------------------------- training
def _augment(x: torch.Tensor, crop: int) -> torch.Tensor:
    n, _, t = x.shape
    if crop < t:
        st = torch.randint(0, t - crop + 1, (n,), device=x.device)
        idx = st[:, None] + torch.arange(crop, device=x.device)[None]
        x = torch.gather(x, 2, idx[:, None, :].expand(-1, x.shape[1], -1))
    scale = 1 + 0.1 * torch.randn(n, 1, 1, device=x.device)
    return x * scale + 0.02 * torch.randn_like(x)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            out.append(torch.sigmoid(model(x[i:i + bs]).float()).cpu().numpy())
    return np.concatenate(out)


def train_model(x_tr, y_tr, x_va, y_va, cfg, log=print, tag="model", resume_path: Path | None = None
                ) -> tuple[nn.Module, dict]:
    """x_*: standardised float32 torch tensors already on DEVICE.

    If ``resume_path`` is given, the full training state is saved there after every epoch and
    training resumes from it when the file exists (survives crashes / sleep).
    """
    c = cfg.cnn
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    model = XResNet1d(n_classes=y_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    steps = c.epochs * math.ceil(len(x_tr) / c.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c.lr, total_steps=steps, pct_start=0.3)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    y_tr_t = torch.as_tensor(y_tr, device=DEVICE)
    best = {"auc": -1, "epoch": -1, "state": None}
    hist = []
    crop = int(0.9 * x_tr.shape[-1])
    start = 1
    if resume_path is not None and Path(resume_path).exists():
        ck = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, hist, start = ck["best"], ck["hist"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng_cpu"])
        log(f"[{tag}] resumed after epoch {ck['epoch']} (best val auc {best['auc']:.4f})")
    for ep in range(start, c.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        perm = torch.randperm(len(x_tr), device=DEVICE)
        for i in range(0, len(perm), c.batch_size):
            b = perm[i:i + c.batch_size]
            if len(b) < 2:
                continue
            xb = _augment(x_tr[b], crop)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = lossf(model(xb).float(), y_tr_t[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(b)
        pv = predict(model, x_va)
        auc, _ = macro_auc(y_va, pv)
        vloss = float(nn.functional.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6),
                                                         torch.tensor(y_va)).item())
        hist.append({"epoch": ep, "train_loss": tot / len(x_tr), "val_loss": vloss, "val_macro_auc": auc})
        log(f"[{tag}] ep {ep:02d} train_loss {tot/len(x_tr):.4f} val_loss {vloss:.4f} val_auc {auc:.4f} ({time.time()-t0:.0f}s)")
        if auc > best["auc"]:
            best = {"auc": auc, "epoch": ep, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if resume_path is not None:
            tmp = Path(str(resume_path) + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "best": best, "hist": hist, "epoch": ep,
                        "rng_cpu": torch.get_rng_state()}, tmp)
            tmp.replace(resume_path)          # atomic: never leaves a half-written checkpoint
    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "best_val_auc": best["auc"], "history": hist}


def tune_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-class F1-optimal threshold on the validation fold (fallback 0.5 when no positives)."""
    th = np.full(y.shape[1], 0.5)
    grid = np.linspace(0.02, 0.9, 45)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        f = [f1_score(y[:, j], p[:, j] >= g, zero_division=0) for g in grid]
        th[j] = grid[int(np.argmax(f))]
    return th


def save_checkpoint(path: Path, model, classes, std: Standardizer, info: dict, thresholds=None):
    torch.save({"state_dict": model.state_dict(), "classes": classes, "standardizer": std.state(),
                "thresholds": None if thresholds is None else np.asarray(thresholds).tolist(),
                "info": {k: v for k, v in info.items() if k != "history"}}, path)
    with open(path.with_suffix(".history.json"), "w") as f:
        json.dump(info.get("history", []), f, indent=1)


def load_checkpoint(path: Path) -> tuple[XResNet1d, dict]:
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = XResNet1d(n_classes=len(ck["classes"])).to(DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck
