"""Compare our measurements with GE 12SL (PTB-XL+) reference values."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CANONICAL_LEADS

GLOBAL_MAP = {  # ours -> 12SL column
    "hr_bpm": "HR__Global",
    "pr_ms": "PR_Int_Global",
    "qrs_ms": "QRS_Dur_Global",
    "qt_ms": "QT_Int_Global",
    "qtc_bazett_ms": "QT_IntBazett_Global",
    "qrs_axis_deg": "R_AxisFrontal_Global",
}
LEAD_MAP = {"st_j": "ST_Amp", "t_amp": "T_Amp", "r_amp": "R_Amp", "s_amp": "S_Amp"}


def load_12sl(path) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("ecg_id")
    ren = {}
    for c in df.columns:  # unify lead suffix spelling (AVR/aVR)
        for lead in CANONICAL_LEADS:
            if c.upper().endswith("_" + lead.upper()):
                ren[c] = c[: -len(lead)] + lead
    return df.rename(columns=ren)


def flatten(ecg_id: int, m: dict) -> dict:
    row = {"ecg_id": ecg_id, "ok": m["ok"]}
    if m["ok"]:
        row.update(m["globals"])
        for lead, d in m["leads"].items():
            for k, v in d.items():
                row[f"{k}_{lead}"] = v
    return row


def _angle_diff(a, b):
    return (a - b + 180) % 360 - 180


def compare(ours: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    ours = ours.set_index("ecg_id")
    ids = ours.index.intersection(ref.index)
    o, r = ours.loc[ids], ref.loc[ids]
    rows = []

    def add(name, a, b, angle=False):
        m = a.notna() & b.notna()
        if m.sum() == 0:
            return
        d = _angle_diff(a[m], b[m]) if angle else a[m] - b[m]
        rows.append({"measure": name, "n": int(m.sum()), "bias": d.mean(), "mae": d.abs().mean(),
                     "median_ae": d.abs().median(), "corr": np.corrcoef(a[m], b[m])[0, 1] if not angle else np.nan,
                     "within_tol": None})

    for k, c in GLOBAL_MAP.items():
        if c not in r and k == "qrs_axis_deg":
            c = "QRS_AxisFront_Global"            # Uni-G naming
        if c not in r:
            continue
        add(k, o[k].astype(float), r[c].astype(float), angle=(k == "qrs_axis_deg"))
    for k, c in LEAD_MAP.items():
        if f"{c}_I" not in r:
            continue
        a = pd.concat([o[f"{k}_{l}"] for l in CANONICAL_LEADS]).astype(float).reset_index(drop=True)
        b = pd.concat([r[f"{c}_{l}"] for l in CANONICAL_LEADS]).astype(float).reset_index(drop=True)
        if k == "s_amp":
            b = -b.abs()
        add(f"{k} (all leads)", a, b)
    # P-wave presence agreement
    pf = r["P_Found_Global"] if "P_Found_Global" in r else r["PR_Int_Global"].notna()
    agree = (o["p_present"].astype(bool) == (pf.astype(float) > 0)).mean()
    rows.append({"measure": "p_present agreement", "n": len(ids), "bias": np.nan, "mae": np.nan,
                 "median_ae": np.nan, "corr": agree, "within_tol": None})
    return pd.DataFrame(rows).drop(columns="within_tol")
