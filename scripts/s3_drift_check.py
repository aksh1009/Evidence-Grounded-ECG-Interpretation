"""Step 3 checks: (1) drift correction on annotated-drift records, (2) drift detector vs dataset
annotations, (3) inter-algorithm ceiling (12SL vs Uni-G). Training folds only."""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from claims.config import load_config
from claims.data import CANONICAL_LEADS, load_metadata, split_ids
from claims.validate_measure import compare, flatten, load_12sl

import s3_measure as S

ORDER = [l.upper() for l in CANONICAL_LEADS]


def parse_drift(s) -> set:
    """Dataset notes like ', V5,V6', ', II-AVF', ', alles' -> set of canonical leads."""
    if not isinstance(s, str):
        return set()
    s = s.upper().replace(" ", "")
    if "ALLES" in s or "ALL" in s:
        return set(CANONICAL_LEADS)
    out = set()
    for part in filter(None, re.split(r"[,;]", s)):
        if "-" in part:
            a, b = part.split("-", 1)
            if a in ORDER and b in ORDER:
                out |= {CANONICAL_LEADS[i] for i in range(ORDER.index(a), ORDER.index(b) + 1)}
        elif part in ORDER:
            out.add(CANONICAL_LEADS[ORDER.index(part)])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    meta = load_metadata(cfg)
    r12, rug = (load_12sl(cfg.path("ptbxl_plus") / f"{n}_features.csv") for n in ("12sl", "unig"))
    pos = pd.Series(np.arange(len(meta)), index=meta.index)
    train = split_ids(meta, cfg)["train"]
    drift = meta.loc[train, "baseline_drift"].map(parse_drift)
    ids_d = drift[drift.map(len) > 0].index.to_numpy()
    rng = np.random.default_rng(0)
    ids_d = rng.choice(ids_d, min(400, len(ids_d)), replace=False)
    ids_c = rng.choice(drift[drift.map(len) == 0].index.to_numpy(), 400, replace=False)
    ids = np.concatenate([ids_d, ids_c])

    res = {}
    for sb in (False, True):
        out = S.run(pos.loc[ids].values, ids, meta.loc[ids, "sex"].values, {"spline_baseline": sb})
        res[sb] = pd.DataFrame([flatten(i, out[i]) for i in ids]).set_index("ecg_id")

    print("(1) ST at J point vs 12SL, ONLY on the leads the dataset marks as drifting")
    for sb in (False, True):
        e = []
        for i in ids_d:
            for l in drift[i]:
                a, b = res[sb].loc[i, f"st_j_{l}"], r12.loc[i, f"ST_Amp_{l}"] if i in r12.index else np.nan
                if pd.notna(a) and pd.notna(b):
                    e.append(abs(a - b))
        e = np.asarray(e)
        print(f"   spline={sb!s:5}: n={len(e)} leads  MAE {e.mean():.4f} mV  median {np.median(e):.4f}  "
              f"errors > 0.05 mV: {np.mean(e > 0.05):.1%}")

    print("\n(2) Our per-lead wander metric vs the dataset's drift annotation (lead level)")
    y, s = [], []
    for i in ids:
        for l in CANONICAL_LEADS:
            y.append(l in drift.get(i, set()))
            s.append(res[True].loc[i, f"baseline_wander_mv_{l}"])
    y, s = np.asarray(y), np.asarray(s)
    print(f"   AUROC {roc_auc_score(y, s):.3f}  (annotated leads: {y.sum()} / {len(y)})")
    for thr in (0.2, 0.3, 0.5):
        f = s > thr
        print(f"   flag if wander > {thr} mV: sensitivity {f[y].mean():.2f}  specificity {(~f[~y]).mean():.2f}")

    print("\n(3) Inter-algorithm ceiling on the same records: 12SL vs Uni-G")
    both = r12.index.intersection(rug.index).intersection(ids)
    rows = []
    for name, c12, cug in [("hr_bpm", "HR__Global", "HR__Global"), ("pr_ms", "PR_Int_Global", "PR_Int_Global"),
                           ("qrs_ms", "QRS_Dur_Global", "QRS_Dur_Global"), ("qt_ms", "QT_Int_Global", "QT_Int_Global")]:
        d = (r12.loc[both, c12] - rug.loc[both, cug]).dropna()
        rows.append({"measure": name, "12SL vs UniG median_ae": d.abs().median(), "mae": d.abs().mean()})
    a = pd.concat([r12.loc[both, f"ST_Amp_{l}"] - rug.loc[both, f"ST_Amp_{l}"] for l in CANONICAL_LEADS]).dropna()
    rows.append({"measure": "st_j (all leads)", "12SL vs UniG median_ae": a.abs().median(), "mae": a.abs().mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\n   ours (spline) vs 12SL on the same records:")
    print(compare(res[True].reset_index()[lambda d: d.ok], r12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
