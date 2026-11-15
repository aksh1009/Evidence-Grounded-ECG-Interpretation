"""Steps 7-8: grounding + final evaluation on the held-out test fold (fold 10). Run once.

Outputs artifacts/eval/: metrics.json, summary.md, reports_test_grounded.jsonl
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from claims.config import load_config
from claims.data import build_labels, cnn_targets, label_vocab, load_metadata, load_scp_statements, \
    superclass_of_subclass
from claims.grounding import extract_concepts, ground_report

SYSTEMS = ["reference", "template", "zero_shot", "finetuned"]


def rouge_l(a: str, b: str) -> float:
    x, y = a.split(), b.split()
    if not x or not y:
        return 0.0
    dp = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        prev = 0
        for j in range(1, len(y) + 1):
            cur = dp[j]
            dp[j] = prev + 1 if x[i - 1] == y[j - 1] else max(dp[j], dp[j - 1])
            prev = cur
    l = dp[-1]
    p, r = l / len(x), l / len(y)
    return 0.0 if l == 0 else 2 * p * r / (p + r)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    meta = load_metadata(cfg)
    scp = load_scp_statements(cfg)
    labels = build_labels(meta, scp, cfg.data.diag_min_likelihood)
    y_all, classes = cnn_targets(labels, label_vocab(scp))
    out = cfg.path("artifacts") / "eval"
    out.mkdir(exist_ok=True, parents=True)
    M = {}

    # ------------------------------------------------------------- CNN on test
    cdir = cfg.path("artifacts") / "cnn"
    ids = np.load(cdir / "ids_test.npy")
    p = np.load(cdir / "pred_test.npy")
    y = y_all[meta.index.get_indexer(ids)]
    th = np.asarray(json.load(open(cdir / "thresholds.json"))["thresholds"])
    per = {classes[j]: roc_auc_score(y[:, j], p[:, j]) for j in range(len(classes)) if 0 < y[:, j].sum() < len(y)}
    sub2sup = superclass_of_subclass(scp)
    sup_auc = {}
    for s in ["NORM", "MI", "STTC", "CD", "HYP"]:
        idx = [j for j, c in enumerate(classes) if sub2sup.get(c) == s]
        ys, ps = y[:, idx].max(1), p[:, idx].max(1)
        sup_auc[s] = roc_auc_score(ys, ps)
    pred = p >= th
    M["cnn"] = {
        "macro_auc_35": float(np.mean(list(per.values()))),
        "macro_auc_classes_with_20plus_test_examples": float(np.mean([a for c, a in per.items()
                                                                      if y[:, classes.index(c)].sum() >= 20])),
        "superclass_auc": sup_auc, "superclass_macro_auc": float(np.mean(list(sup_auc.values()))),
        "macro_f1_val_thresholds": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1_val_thresholds": float(f1_score(y, pred, average="micro", zero_division=0)),
        "per_class_auc": per, "n_positive": {c: int(y[:, j].sum()) for j, c in enumerate(classes)},
    }
    print("CNN test macro-AUC", round(M["cnn"]["macro_auc_35"], 4), "superclass", round(M["cnn"]["superclass_macro_auc"], 4))

    # ------------------------------------------------------------- measurements (computed in Step 3)
    mdir = cfg.path("artifacts") / "measurements"
    M["measurements_vs_references"] = pd.read_csv(mdir / "agreement_summary_test.csv").to_dict("records")

    # ------------------------------------------------------------- reports
    evs = {}
    for line in open(cfg.path("artifacts") / "evidence" / "evidence_test.jsonl", encoding="utf-8"):
        e = json.loads(line)
        evs[e["ecg_id"]] = e
    reps = [json.loads(l) for l in open(cfg.path("artifacts") / "llm" / "reports_test.jsonl", encoding="utf-8")]
    reps = [r for r in reps if r.get("reference")]                    # records with a cardiologist interpretation
    eval_classes = [c for j, c in enumerate(classes) if y_all[meta.index.get_indexer([r["ecg_id"] for r in reps])][:, j].sum() >= 10]
    Y = y_all[meta.index.get_indexer([r["ecg_id"] for r in reps])]
    Y = Y[:, [classes.index(c) for c in eval_classes]]
    M["reports"] = {"n": len(reps), "eval_classes": eval_classes}
    grounded = []
    for sysname in SYSTEMS:
        P = np.zeros_like(Y)
        n_con, n_uns, rep_with_uns, rl, lens, verified_kept = 0, 0, 0, [], [], []
        n_weak, n_unhedged = 0, 0
        for i, r in enumerate(reps):
            txt = r[sysname] or ""
            cs = extract_concepts(txt)
            for k, c in enumerate(eval_classes):
                P[i, k] = c in cs
            g = ground_report(txt, evs[r["ecg_id"]])
            n_con += g["n_concepts"]
            n_uns += g["n_unsupported"]
            n_weak += g["n_low_confidence"]
            n_unhedged += g["n_low_confidence_unhedged"]
            rep_with_uns += g["n_unsupported"] > 0
            if sysname != "reference":
                rl.append(rouge_l(txt.lower(), r["reference"].lower()))
            lens.append(len(txt.split()))
            verified_kept.append(sum(1 for s in g["sentences"] if not s["unsupported"]) / max(1, len(g["sentences"])))
            r.setdefault("grounding", {})[sysname] = {"cited": g["cited_report"], "verified": g["verified_report"],
                                                     "unsupported": [c for s in g["sentences"] for c in s["unsupported"]]}
        M["reports"][sysname] = {
            "clinical_macro_f1": float(f1_score(Y, P, average="macro", zero_division=0)),
            "clinical_micro_f1": float(f1_score(Y, P, average="micro", zero_division=0)),
            "per_class_f1": {c: float(f1_score(Y[:, k], P[:, k], zero_division=0)) for k, c in enumerate(eval_classes)},
            "unsupported_concept_rate": n_uns / max(1, n_con),
            "reports_with_unsupported": rep_with_uns / len(reps),
            "sentences_kept_after_verification": float(np.mean(verified_kept)),
            "rouge_l_vs_reference": float(np.mean(rl)) if rl else None,
            "mean_words": float(np.mean(lens)),
            "low_confidence_mentions": n_weak,
            "low_confidence_stated_without_hedge": n_unhedged / max(1, n_weak),
        }
        s = M["reports"][sysname]
        print(f"{sysname:10s} clinical micro-F1 {s['clinical_micro_f1']:.3f} macro-F1 {s['clinical_macro_f1']:.3f} | "
              f"unsupported concepts {s['unsupported_concept_rate']:.1%} | reports w/ unsupported {s['reports_with_unsupported']:.1%}"
              f" | ROUGE-L {s['rouge_l_vs_reference'] if s['rouge_l_vs_reference'] is None else round(s['rouge_l_vs_reference'], 3)}")
    # after verification: clinical accuracy of the verified fine-tuned report
    P = np.zeros_like(Y)
    for i, r in enumerate(reps):
        cs = extract_concepts(r["grounding"]["finetuned"]["verified"])
        for k, c in enumerate(eval_classes):
            P[i, k] = c in cs
    n_weak = n_unhedged = 0
    for r in reps:
        g = ground_report(re.sub(r"\s*\[[MC]\d+(, [MC]\d+)*\]", "", r["grounding"]["finetuned"]["verified"]),
                          evs[r["ecg_id"]])
        n_weak += g["n_low_confidence"]
        n_unhedged += g["n_low_confidence_unhedged"]
    M["reports"]["finetuned_verified"] = {"low_confidence_stated_without_hedge": n_unhedged / max(1, n_weak),"clinical_micro_f1": float(f1_score(Y, P, average="micro", zero_division=0)),
                                          "clinical_macro_f1": float(f1_score(Y, P, average="macro", zero_division=0)),
                                          "unsupported_concept_rate": 0.0}
    print("finetuned after verification: micro-F1", round(M["reports"]["finetuned_verified"]["clinical_micro_f1"], 3),
          "| low-confidence findings stated without hedge:",
          f'{M["reports"]["finetuned"]["low_confidence_stated_without_hedge"]:.1%} ->',
          f'{M["reports"]["finetuned_verified"]["low_confidence_stated_without_hedge"]:.1%}')

    with open(out / "reports_test_grounded.jsonl", "w", encoding="utf-8") as f:
        for r in reps:
            f.write(json.dumps(r) + "\n")
    json.dump(M, open(out / "metrics.json", "w"), indent=1, default=float)
    print("saved", out)


if __name__ == "__main__":
    main()
