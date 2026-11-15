"""Grid-search measurement thresholds against 12SL on TRAINING-fold records only."""
import itertools
import time

import numpy as np
import pandas as pd

from claims.config import load_config
from claims.data import load_metadata, split_ids
from claims.validate_measure import compare, flatten, load_12sl

import s3_measure as S


def main():
    cfg = load_config()
    meta = load_metadata(cfg)
    ref = load_12sl(cfg.path("ptbxl_plus") / "12sl_features.csv")
    pos = pd.Series(np.arange(len(meta)), index=meta.index)
    ids = np.random.default_rng(1).choice(split_ids(meta, cfg)["train"], 600, replace=False)
    grid = {
        "qrs_on_frac": [0.03, 0.05, 0.07],
        "qrs_off_frac": [0.04, 0.06, 0.08],
        "tend_percentile": [75, 90, 100],
        "p_edge_frac": [0.15, 0.25],
    }
    rows = []
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid.keys(), vals))
        t = time.time()
        res = S.run(pos.loc[ids].values, ids, meta.loc[ids, "sex"].values, p)
        flat = pd.DataFrame([flatten(i, res[i]) for i in ids])
        c = compare(flat[flat.ok], ref).set_index("measure")
        rows.append({**p, **{f"{m}_mae": c.loc[m, "mae"] for m in ["pr_ms", "qrs_ms", "qt_ms"]},
                     **{f"{m}_bias": c.loc[m, "bias"] for m in ["pr_ms", "qrs_ms", "qt_ms"]},
                     "stj_mae": c.loc["st_j (all leads)", "mae"], "p_agree": c.loc["p_present agreement", "corr"]})
        print(rows[-1], f"{time.time()-t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["score"] = df.pr_ms_mae / 20 + df.qrs_ms_mae / 10 + df.qt_ms_mae / 20 + df.stj_mae / 0.02
    df.sort_values("score").to_csv(cfg.path("artifacts") / "measure_calibration.csv", index=False)
    print(df.sort_values("score").head(8).round(3).to_string())


if __name__ == "__main__":
    main()
