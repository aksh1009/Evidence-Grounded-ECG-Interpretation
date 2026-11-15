"""Step 1 - PTB-XL metadata, cardiologist labels, fold splits and a signal cache.

Labels come exclusively from the cardiologist-assigned ``scp_codes`` in ptbxl_database.csv.
CNN targets = 23 diagnostic subclasses + 12 rhythm statements (multi-label).
"""
from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# --------------------------------------------------------------------------- metadata
def load_metadata(cfg: Config) -> pd.DataFrame:
    root = cfg.path("ptbxl")
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)  # dict code -> likelihood (0-100)
    n = int(cfg.data.get("dev_subset", 0) or 0)
    if n > 0:
        df = df.iloc[:n]
    return df


def load_scp_statements(cfg: Config) -> pd.DataFrame:
    scp = pd.read_csv(cfg.path("ptbxl") / "scp_statements.csv", index_col=0)
    for c in ("diagnostic", "form", "rhythm"):
        scp[c] = scp[c].fillna(0).astype(int)
    return scp


# --------------------------------------------------------------------------- labels
def label_vocab(scp: pd.DataFrame) -> dict[str, list[str]]:
    diag = scp[scp.diagnostic == 1]
    return {
        "superclass": SUPERCLASSES,
        "subclass": sorted(diag.diagnostic_subclass.dropna().unique().tolist()),
        "diagnostic": sorted(diag.index.tolist()),
        "rhythm": sorted(scp[scp.rhythm == 1].index.tolist()),
        "form": sorted(scp[scp.form == 1].index.tolist()),
    }


def build_labels(meta: pd.DataFrame, scp: pd.DataFrame, diag_min_likelihood: float = 0) -> pd.DataFrame:
    """Per-record label lists at every level of the PTB-XL hierarchy.

    * diagnostic codes: present if likelihood >= diag_min_likelihood
      (likelihood 0 means "unknown" in PTB-XL, not "absent").
    * rhythm / form codes: present whenever listed (their likelihood is typically 0 by design).
    """
    diag = scp[scp.diagnostic == 1]
    rhythm = set(scp[scp.rhythm == 1].index)
    form = set(scp[scp.form == 1].index)

    rows = []
    for ecg_id, codes in meta["scp_codes"].items():
        d = [c for c, lik in codes.items() if c in diag.index and lik >= diag_min_likelihood]
        rows.append({
            "ecg_id": ecg_id,
            "diagnostic": sorted(d),
            "subclass": sorted({diag.loc[c, "diagnostic_subclass"] for c in d}),
            "superclass": sorted({diag.loc[c, "diagnostic_class"] for c in d}),
            "rhythm": sorted(c for c in codes if c in rhythm),
            "form": sorted(c for c in codes if c in form),
        })
    return pd.DataFrame(rows).set_index("ecg_id")


def cnn_targets(labels: pd.DataFrame, vocab: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Multi-hot matrix over 23 diagnostic subclasses + 12 rhythm codes."""
    classes = vocab["subclass"] + vocab["rhythm"]
    idx = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for r, (sub, rhy) in enumerate(zip(labels["subclass"], labels["rhythm"])):
        for c in list(sub) + list(rhy):
            y[r, idx[c]] = 1.0
    return y, classes


def superclass_of_subclass(scp: pd.DataFrame) -> dict[str, str]:
    diag = scp[scp.diagnostic == 1]
    return diag.groupby("diagnostic_subclass")["diagnostic_class"].first().to_dict()


# --------------------------------------------------------------------------- splits
def split_ids(meta: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    f = meta["strat_fold"]
    return {
        "train": meta.index[f.isin(cfg.data.train_folds)].to_numpy(),
        "val": meta.index[f == cfg.data.val_fold].to_numpy(),
        "test": meta.index[f == cfg.data.test_fold].to_numpy(),
    }


# --------------------------------------------------------------------------- signals
def _read_one(args: tuple[str, int]) -> np.ndarray:
    import wfdb

    path, n = args
    rec = wfdb.rdrecord(path, physical=False)
    order = [rec.sig_name.index(name) for name in _file_lead_names(rec.sig_name)]
    gains = np.asarray(rec.adc_gain, dtype=np.float64)[order]
    base = np.asarray(rec.baseline, dtype=np.float64)[order]
    sig = (rec.d_signal[:, order].astype(np.float64) - base) / gains  # mV
    out = np.zeros((n, 12), dtype=np.int16)
    m = min(n, sig.shape[0])
    out[:m] = np.round(sig[:m] * 1000).clip(-32768, 32767).astype(np.int16)  # microvolts
    return out


def _file_lead_names(sig_name: list[str]) -> list[str]:
    """Map canonical lead names to the (case-variant) names used in the files, e.g. AVR -> aVR."""
    lookup = {s.upper(): s for s in sig_name}
    return [lookup[c.upper()] for c in CANONICAL_LEADS]


def signal_cache_path(cfg: Config) -> Path:
    n = int(cfg.data.get("dev_subset", 0) or 0)
    return cfg.path("cache") / (f"signals500_dev{n}.npy" if n else "signals500.npy")


def build_signal_cache(cfg: Config, meta: pd.DataFrame, workers: int = 12) -> Path:
    """All records as int16 microvolts, shape (N, 5000, 12), row order == meta.index order."""
    out = signal_cache_path(cfg)
    ids_path = out.with_suffix(".ids.npy")
    if out.exists() and ids_path.exists() and np.array_equal(np.load(ids_path), meta.index.to_numpy()):
        return out
    root = cfg.path("ptbxl")
    n = int(cfg.data.n_samples)
    jobs = [(str(root / fn), n) for fn in meta["filename_hr"]]
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.int16, shape=(len(jobs), n, 12))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, sig in enumerate(ex.map(_read_one, jobs, chunksize=64)):
            arr[i] = sig
    arr.flush()
    del arr
    np.save(ids_path, meta.index.to_numpy())
    return out


def load_signals(cfg: Config, meta: pd.DataFrame) -> np.ndarray:
    """Memory-mapped int16 microvolt array aligned with meta.index. Use ``to_mv`` to convert."""
    path = signal_cache_path(cfg)
    ids = np.load(path.with_suffix(".ids.npy"))
    assert np.array_equal(ids, meta.index.to_numpy()), "signal cache out of sync with metadata"
    return np.load(path, mmap_mode="r")


def to_mv(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) / 1000.0
