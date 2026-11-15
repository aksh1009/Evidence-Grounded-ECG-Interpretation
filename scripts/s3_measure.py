"""Step 3: run the measurement engine.

  python scripts/s3_measure.py validate --n 300 [--split train]   # compare to 12SL on a sample
  python scripts/s3_measure.py all                                 # measure every record, save parquet+json
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from claims.config import load_config
from claims.data import load_metadata, load_signals, split_ids, to_mv
from claims.measure import measure
from claims.signal import clean_for_measurement
from claims.validate_measure import compare, flatten, load_12sl
from claims.utils import keep_awake

_SIG = None
_CFG = None


def _init():
    global _SIG, _CFG
    _CFG = load_config()
    _SIG = load_signals(_CFG, load_metadata(_CFG))


def _work(args):
    row, ecg_id, sex, params = args
    s = _CFG.signal
    x = clean_for_measurement(to_mv(_SIG[row]), 500, s.highpass_hz, s.lowpass_hz, s.notch_hz)
    try:
        m = measure(x, 500, sex=sex, params=params).to_dict()
    except Exception as e:  # never silently drop: record the failure
        m = {"ok": False, "reason": f"error: {e}"}
    return ecg_id, m


def run(rows, ids, sexes, params=None, workers=6):
    jobs = [(int(r), int(i), int(s), params) for r, i, s in zip(rows, ids, sexes)]
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        return dict(ex.map(_work, jobs, chunksize=16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "all"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", default="train")
    ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    keep_awake()

    cfg = load_config()
    meta = load_metadata(cfg)
    refs = {n: load_12sl(cfg.path("ptbxl_plus") / f"{n}_features.csv") for n in ("12sl", "unig")}
    pos = pd.Series(np.arange(len(meta)), index=meta.index)
    t = time.time()

    if a.mode == "validate":
        ids = split_ids(meta, cfg)[a.split]
        ids = np.random.default_rng(0).choice(ids, min(a.n, len(ids)), replace=False)
    else:
        ids = meta.index.to_numpy()
    res = run(pos.loc[ids].values, ids, meta.loc[ids, "sex"].values, json.loads(a.params))
    flat = pd.DataFrame([flatten(i, res[i]) for i in ids])
    print(f"measured {len(ids)} in {time.time()-t:.0f}s, failures: {(~flat.ok).sum()}")
    pd.set_option("display.width", 200)
    for n, ref in refs.items():
        print(f"--- agreement vs {n}")
        print(compare(flat[flat.ok], ref).round(3).to_string(index=False))

    if a.mode == "all":
        out = cfg.path("artifacts") / "measurements"
        out.mkdir(exist_ok=True, parents=True)
        flat.to_parquet(out / "measurements_flat.parquet")
        with open(out / "measurements_full.json", "w") as f:
            json.dump({str(k): v for k, v in res.items()}, f)
        # held-out agreement report (fold 10 only), saved for the final evaluation
        test_ids = split_ids(meta, cfg)["test"]
        for name, reference in refs.items():
            comp = compare(flat[flat.ok & flat.ecg_id.isin(test_ids)], reference)
            comp.to_csv(out / f"agreement_vs_{name}_test.csv", index=False)
            print(f"\nTEST-fold agreement vs {name}:\n", comp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
