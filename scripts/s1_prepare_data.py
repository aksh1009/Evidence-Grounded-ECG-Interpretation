"""Step 1: labels, splits and signal cache. Prints sanity statistics."""
import json
import time

import numpy as np

from claims.config import load_config
from claims.data import (build_labels, build_signal_cache, cnn_targets, label_vocab, load_metadata,
                         load_scp_statements, load_signals, split_ids)


def main():
    cfg = load_config()
    meta = load_metadata(cfg)
    scp = load_scp_statements(cfg)
    vocab = label_vocab(scp)
    labels = build_labels(meta, scp, cfg.data.diag_min_likelihood)
    y, classes = cnn_targets(labels, vocab)
    splits = split_ids(meta, cfg)

    print(f"records: {len(meta)}  patients: {meta.patient_id.nunique()}")
    print({k: len(v) for k, v in splits.items()})
    print({k: len(v) for k, v in vocab.items()})

    # patient leakage check across splits
    pid = meta["patient_id"]
    sets = {k: set(pid.loc[v]) for k, v in splits.items()}
    print("patient overlap train/val:", len(sets["train"] & sets["val"]),
          "train/test:", len(sets["train"] & sets["test"]), "val/test:", len(sets["val"] & sets["test"]))

    no_diag = (labels["subclass"].str.len() == 0).sum()
    print(f"records without any diagnostic label: {no_diag}")
    counts = {c: int(y[:, i].sum()) for i, c in enumerate(classes)}
    print("CNN class counts:", json.dumps(counts))

    out = cfg.path("cache")
    labels.to_pickle(out / "labels.pkl")
    with open(out / "vocab.json", "w") as f:
        json.dump({**vocab, "cnn_classes": classes}, f, indent=1)

    t = time.time()
    p = build_signal_cache(cfg, meta)
    sig = load_signals(cfg, meta)
    print(f"signal cache {p.name}: {sig.shape} {sig.dtype} in {time.time()-t:.0f}s")
    zero_rows = int((np.abs(sig[:, :, :]).reshape(len(sig), -1).max(1) == 0).sum())
    print("all-zero records:", zero_rows)


if __name__ == "__main__":
    main()
