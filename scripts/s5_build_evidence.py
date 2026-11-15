"""Step 5: evidence JSON for every ECG.

* probabilities: out-of-fold CNN predictions for training folds (honest), final model for val/test
* attributions (Integrated Gradients) for val/test, where citations are shown and evaluated
* faithfulness probe on a test sample
* rule-claim sanity check against cardiologist labels (training folds only)
Outputs: artifacts/evidence/evidence_{train,val,test}.jsonl + step5_report.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from claims.claims_builder import build_evidence, measurement_claims
from claims.config import load_config
from claims.data import CANONICAL_LEADS, load_metadata, split_ids
from claims.explain import deletion_check, integrated_gradients, summarize_attribution
from claims.signal import Standardizer
from claims.train_cnn import DEVICE, load_checkpoint
from claims.utils import keep_awake

# rule finding -> cardiologist labels it should indicate (for the sanity check)
RULE_VS_LABEL = {
    "WIDE_QRS": ["CLBBB", "CRBBB", "IVCD", "PACE", "WPW"],
    "LONG_PR": ["_AVB"],
    "LVH_V": ["LVH"],
    "TACHY": ["STACH", "SVTAC", "PSVT"],
    "BRADY": ["SBRAD"],
    "LAD": ["LAFB/LPFB"],
    "LOWV": ["LVOLT"],
    "LONG_QT": ["LNGQT"],
    "QWAVE": ["IMI", "AMI", "LMI", "QWAVE"],
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    keep_awake()
    cfg = load_config()
    meta = load_metadata(cfg)
    labels = pd.read_pickle(cfg.path("cache") / "labels.pkl")
    out = cfg.path("artifacts") / "evidence"
    out.mkdir(parents=True, exist_ok=True)
    cdir = cfg.path("artifacts") / "cnn"
    t0 = time.time()
    meas = json.load(open(cfg.path("artifacts") / "measurements" / "measurements_full.json"))
    print(f"loaded measurements ({time.time()-t0:.0f}s)")

    model, ck = load_checkpoint(cdir / "cnn_final.pt")
    classes, th = ck["classes"], np.asarray(ck["thresholds"])
    std = Standardizer.from_state(ck["standardizer"])
    X = np.load(cfg.path("cache") / "cnn100.npy", mmap_mode="r")
    pos = pd.Series(np.arange(len(meta)), index=meta.index)

    probs = {}
    for split, pfile, ifile in [("train", "pred_oof_train.npy", "ids_oof_train.npy"),
                                ("val", "pred_val.npy", "ids_val.npy"), ("test", "pred_test.npy", "ids_test.npy")]:
        p, ids = np.load(cdir / pfile), np.load(cdir / ifile)
        assert not np.isnan(p).any(), f"{split}: missing predictions"
        probs[split] = dict(zip(ids.tolist(), p))

    report = {"faithfulness": None, "rule_vs_label": {}, "evidence_stats": {}}
    splits = split_ids(meta, cfg)

    # ---------------------------------------------------------------- build evidence
    faith = []
    rng = np.random.default_rng(0)
    for split in ("test", "val", "train"):
        ids = splits[split]
        with_attr = split in ("val", "test")
        n_claims, n_dx, n_sup, n_con, n_flag = 0, 0, 0, 0, 0
        t = time.time()
        with open(out / f"evidence_{split}.jsonl", "w", encoding="utf-8") as f:
            for k, eid in enumerate(ids):
                m = meas[str(eid)]
                p = probs[split][int(eid)]
                row = meta.loc[eid]
                attr = None
                if with_attr:
                    pos_idx = [j for j in range(len(classes)) if p[j] >= th[j]]
                    if pos_idx:
                        x = torch.as_tensor(std(np.asarray(X[pos.loc[eid]])[None])[0], device=DEVICE)
                        A = integrated_gradients(model, x, pos_idx, steps=24)
                        beats = m.get("beats") if m.get("ok") else None
                        attr = {classes[j]: summarize_attribution(A[i], 100, beats) for i, j in enumerate(pos_idx)}
                        if split == "test" and len(faith) < 400 and rng.random() < 0.3:
                            j = pos_idx[int(np.argmax(p[pos_idx]))]
                            order = [CANONICAL_LEADS.index(l) for l in attr[classes[j]]["top_leads"]]
                            faith.append(deletion_check(model, x, j, order))
                ev = build_evidence(int(eid), {"sex": row.sex, "age": row.age}, m, p, classes, th, attr)
                ev["split"] = split
                f.write(json.dumps(ev) + "\n")
                n_claims += len(ev["findings"])
                n_dx += len(ev["diagnoses"])
                n_sup += sum(bool(c["supported_by"]) for c in ev["diagnoses"])
                n_con += sum(bool(c["contradicted_by"]) for c in ev["diagnoses"])
                n_flag += bool(ev["flags"])
                if k % 2000 == 0:
                    print(f"  {split}: {k}/{len(ids)} ({time.time()-t:.0f}s)", flush=True)
        st = {"records": len(ids), "findings_per_ecg": n_claims / len(ids), "diagnoses_per_ecg": n_dx / len(ids),
              "dx_with_measurement_support": n_sup / max(1, n_dx), "dx_contradicted": n_con / max(1, n_dx),
              "ecgs_with_flags": n_flag / len(ids)}
        report["evidence_stats"][split] = st
        print(split, {k: round(v, 3) for k, v in st.items()})

    fd = pd.DataFrame(faith)
    report["faithfulness"] = {"n": len(fd), "mean_drop_top_lead": fd.drop_top.mean(),
                              "mean_drop_random_lead": fd.drop_random.mean(),
                              "top_hurts_more_fraction": (fd.drop_top > fd.drop_random).mean()}
    print("faithfulness:", {k: round(float(v), 3) for k, v in report["faithfulness"].items()})

    # ---------------------------------------------------------------- rule findings vs cardiologist labels (train)
    tr = [e for e in splits["train"] if meas[str(e)].get("ok")]
    rule_c = {e: {c["concept"] for c in measurement_claims(meas[str(e)], meta.at[e, "sex"], meta.at[e, "age"])}
              for e in tr}
    lab_c = {e: set(labels.at[e, "subclass"]) | set(labels.at[e, "rhythm"]) | set(labels.at[e, "form"])
             | set(labels.at[e, "diagnostic"]) for e in tr}
    rows = []
    for concept, labs in RULE_VS_LABEL.items():
        has_rule = [concept in rule_c[e] for e in tr]
        has_lab = [bool(lab_c[e] & set(labs)) for e in tr]
        r, l = np.asarray(has_rule), np.asarray(has_lab)
        rows.append({"rule": concept, "labels": "/".join(labs), "n_label": int(l.sum()),
                     "sensitivity": r[l].mean() if l.any() else np.nan, "specificity": (~r[~l]).mean(),
                     "ppv": l[r].mean() if r.any() else np.nan})
        print(rows[-1], flush=True)
    report["rule_vs_label"] = rows
    json.dump(report, open(out / "step5_report.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
