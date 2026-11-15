"""Calibrate the P-wave beat-to-beat consistency threshold: AFib vs sinus rhythm, TRAINING folds only."""
import numpy as np
import pandas as pd

import s3_measure as S
from claims.config import load_config
from claims.data import load_metadata, split_ids
from claims.validate_measure import flatten


def main():
    cfg = load_config()
    meta = load_metadata(cfg)
    lab = pd.read_pickle(cfg.path("cache") / "labels.pkl")
    tr = split_ids(meta, cfg)["train"]
    rng = np.random.default_rng(0)
    af = [e for e in tr if "AFIB" in lab.at[e, "rhythm"]]
    sr = [e for e in tr if lab.at[e, "rhythm"] == ["SR"]]
    ids = np.concatenate([rng.choice(af, 400, replace=False), rng.choice(sr, 800, replace=False)])
    pos = pd.Series(np.arange(len(meta)), index=meta.index)
    res = S.run(pos.loc[ids].values, ids, meta.loc[ids, "sex"].values, {"p_min_consistency": -1})
    d = pd.DataFrame([flatten(i, res[i]) for i in ids])
    d = d[d.ok]
    d["af"] = d.ecg_id.isin(af)
    c = d.p_consistency.fillna(-1)
    print("median consistency  AF", round(c[d.af].median(), 2), " sinus", round(c[~d.af].median(), 2))
    for q in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pres = (c >= q) & d.p_present
        print(f"thr {q}: P 'present' in AF {pres[d.af].mean():.2f} | in sinus {pres[~d.af].mean():.3f}")


if __name__ == "__main__":
    main()
