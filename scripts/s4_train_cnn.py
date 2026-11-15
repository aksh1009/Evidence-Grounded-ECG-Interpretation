"""Step 4: train the diagnostic CNN.

  python scripts/s4_train_cnn.py final        # train on folds 1-8, select on fold 9, predict val+test
  python scripts/s4_train_cnn.py oof          # cross-fitted out-of-fold predictions for folds 1-8
  --epochs N / --limit N for quick smoke runs
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from claims.config import load_config
from claims.data import cnn_targets, load_metadata, load_scp_statements, load_signals, label_vocab, build_labels
from claims.signal import Standardizer
from claims.utils import keep_awake
from claims.train_cnn import (DEVICE, build_cnn_input_cache, macro_auc, predict, save_checkpoint, train_model,
                              tune_thresholds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["final", "oof", "smoke"])
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    keep_awake()

    cfg = load_config()
    if a.epochs:
        cfg["cnn"]["epochs"] = a.epochs
    meta = load_metadata(cfg)
    scp = load_scp_statements(cfg)
    labels = build_labels(meta, scp, cfg.data.diag_min_likelihood)
    y, classes = cnn_targets(labels, label_vocab(scp))
    sig = load_signals(cfg, meta)
    X = np.asarray(build_cnn_input_cache(sig, cfg.path("cache") / "cnn100.npy"))
    fold = meta["strat_fold"].to_numpy()
    out = cfg.path("artifacts") / "cnn"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"train_{a.mode}.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    va = fold == cfg.data.val_fold
    te = fold == cfg.data.test_fold

    def fit(tr_mask, tag):
        std = Standardizer().fit(X[tr_mask])
        xt = torch.as_tensor(std(X[tr_mask]), device=DEVICE)
        xv = torch.as_tensor(std(X[va]), device=DEVICE)
        model, info = train_model(xt, y[tr_mask], xv, y[va], cfg, log=log, tag=tag,
                                  resume_path=out / f"resume_{tag.replace(' ', '').replace(',', '-')}.pt")
        del xt
        return model, std, info

    def pred(model, std, mask):
        return predict(model, torch.as_tensor(std(X[mask]), device=DEVICE))

    if a.mode == "smoke":
        tr = np.isin(fold, [1])
        cfg["cnn"]["epochs"] = a.epochs or 2
        fit(tr, "smoke")
        return

    if a.mode == "final":
        tr = np.isin(fold, cfg.data.train_folds)
        model, std, info = fit(tr, "final")
        pv, pt = pred(model, std, va), pred(model, std, te)
        th = tune_thresholds(y[va], pv)
        save_checkpoint(out / "cnn_final.pt", model, classes, std, info, th)
        np.save(out / "pred_val.npy", pv)
        np.save(out / "pred_test.npy", pt)
        np.save(out / "ids_val.npy", meta.index[va].to_numpy())
        np.save(out / "ids_test.npy", meta.index[te].to_numpy())
        auc_v, _ = macro_auc(y[va], pv)
        log(f"[final] best epoch {info['best_epoch']}  val macro-AUC {auc_v:.4f}")
        json.dump({"classes": classes, "thresholds": th.tolist()}, open(out / "thresholds.json", "w"), indent=1)

    if a.mode == "oof":
        oof = np.full((len(meta), len(classes)), np.nan, dtype=np.float32)
        for g in cfg.cnn.oof_groups:
            m = np.isin(fold, g)
            gfile = out / f"pred_oof_{'-'.join(map(str, g))}.npy"
            if gfile.exists():                       # group already finished in an earlier run
                oof[m] = np.load(gfile)
                continue
            tr = np.isin(fold, cfg.data.train_folds) & ~np.isin(fold, g)
            model, std, info = fit(tr, f"oof{g}")
            oof[m] = pred(model, std, m)
            np.save(gfile, oof[m])
            auc, _ = macro_auc(y[m], oof[m])
            log(f"[oof{g}] held-out folds {g} macro-AUC {auc:.4f}")
            torch.cuda.empty_cache()
        trm = np.isin(fold, cfg.data.train_folds)
        np.save(out / "pred_oof_train.npy", oof[trm])
        np.save(out / "ids_oof_train.npy", meta.index[trm].to_numpy())
        auc, _ = macro_auc(y[trm], oof[trm])
        log(f"[oof] all train folds out-of-fold macro-AUC {auc:.4f}")


if __name__ == "__main__":
    main()
