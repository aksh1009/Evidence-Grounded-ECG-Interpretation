"""Step 9: build the demo page artifacts/demo/index.html (self-contained, opens offline in any browser).

For a curated set of test-fold ECGs it shows: the 12-lead signal, every evidence item [M#]/[C#] with the
time window and leads it refers to (click to highlight), our measurements next to GE 12SL, the CNN
probabilities, and four reports (cardiologist, template, fine-tuned LLM raw, fine-tuned LLM verified).
Everything comes from the artifacts of steps 3-8; no GPU needed.
"""
import json
import re
import sys

import numpy as np
import pandas as pd

from claims.config import load_config
from claims.data import CANONICAL_LEADS, load_metadata, load_signals, to_mv
from claims.grounding import ground_report
from claims.signal import clean_for_measurement
from claims.validate_measure import GLOBAL_MAP, load_12sl

CURATED = {  # ecg_id -> (title, category); categories are shown as filters in the case selector
    9: ("Normal ECG", "Normal"),
    318: ("Atrial fibrillation with ST depression", "Rhythm"),
    180: ("Left bundle branch block (secondary ST/Q)", "Conduction"),
    172: ("Right bundle branch block + left anterior fascicular block", "Conduction"),
    514: ("Inferior infarction", "Infarction"),
    63: ("Anterior infarction", "Infarction"),
    299: ("Left ventricular hypertrophy", "Hypertrophy"),
    382: ("Pacemaker", "Rhythm"),
    218: ("First-degree AV block", "Conduction"),
    611: ("Sinus bradycardia", "Rhythm"),
    892: ("Sinus tachycardia", "Rhythm"),
    167: ("Anterior ischaemia", "Ischaemia"),
    4825: ("WPW pre-excitation", "Conduction"),
}
SUPER_TO_CATEGORY = {"MI": "Infarction", "STTC": "Ischaemia", "CD": "Conduction", "HYP": "Hypertrophy", "NORM": "Normal"}
FS_OUT = 250
CITE = re.compile(r"\s*\[([MC]\d+(?:, [MC]\d+)*)\]")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    meta = load_metadata(cfg)
    sig = load_signals(cfg, meta)
    s = cfg.signal
    evs, reps = {}, {}
    for line in open(cfg.path("artifacts") / "evidence" / "evidence_test.jsonl", encoding="utf-8"):
        e = json.loads(line)
        evs[e["ecg_id"]] = e
    for line in open(cfg.path("artifacts") / "eval" / "reports_test_grounded.jsonl", encoding="utf-8"):
        r = json.loads(line)
        reps[r["ecg_id"]] = r
    meas = json.load(open(cfg.path("artifacts") / "measurements" / "measurements_full.json"))
    ref = load_12sl(cfg.path("ptbxl_plus") / "12sl_features.csv")
    probs = dict(zip(np.load(cfg.path("artifacts") / "cnn" / "ids_test.npy").tolist(),
                     np.load(cfg.path("artifacts") / "cnn" / "pred_test.npy")))
    import torch
    ck = torch.load(cfg.path("artifacts") / "cnn" / "cnn_final.pt", map_location="cpu", weights_only=False)
    classes, th = ck["classes"], ck["thresholds"]
    from claims.claims_builder import CLASS_NAMES

    # add two automatically chosen teaching cases: one where hedging fired, one where a sentence was removed
    cur = dict(CURATED)
    for eid in sorted(reps):
        g = ground_report(reps[eid]["finetuned"], evs[eid])
        if "hedge" not in cur.values() and g["n_low_confidence_unhedged"] and eid not in cur:
            cur[eid] = "hedge"
        if "removed" not in cur.values() and g["n_unsupported"] and eid not in cur:
            cur[eid] = "removed"
    labels = pd.read_pickle(cfg.path("cache") / "labels.pkl")
    reports_en = pd.read_parquet(cfg.path("artifacts") / "reports_en.parquet")

    def category_from_labels(eid):
        sup = labels.at[eid, "superclass"]
        for k in ("MI", "STTC", "CD", "HYP", "NORM"):
            if k in sup:
                return SUPER_TO_CATEGORY[k]
        return "Rhythm"

    teaching = {"hedge": "Low-confidence finding hedged by the verifier",
                "removed": "Unsupported phrase removed by the verifier"}
    cur = {k: (teaching[v], category_from_labels(k), True) if isinstance(v, str) else (v[0], v[1], False)
           for k, v in cur.items()}

    cases = []
    for eid, (why, category, verifier_example) in cur.items():
        row = meta.index.get_loc(eid)
        x = clean_for_measurement(to_mv(sig[row]), 500, s.highpass_hz, s.lowpass_hz, s.notch_hz)
        x = x[:: 500 // FS_OUT]
        r, ev, m = reps[eid], evs[eid], meas[str(eid)]
        g = ground_report(r["finetuned"], ev)
        ours = m.get("globals", {}) if m.get("ok") else {}
        cmp_rows = []
        for k, col in GLOBAL_MAP.items():
            a = ours.get(k)
            b = ref.at[eid, col] if eid in ref.index and col in ref.columns else None
            cmp_rows.append({"name": k, "ours": None if a is None else round(float(a), 1),
                             "ref": None if b is None or pd.isna(b) else round(float(b), 1)})
        p = probs[eid]
        top = sorted(range(len(classes)), key=lambda j: -p[j])[:8]
        cases.append({
            "id": int(eid), "why": why, "category": category, "verifier_example": verifier_example,
            "labels": sorted(set(labels.at[eid, "subclass"]) | set(labels.at[eid, "rhythm"])),
            "age": ev["patient"]["age"], "sex": ev["patient"]["sex"], "quality": ev["quality"],
            "signal": [np.round(x[:, i] * 1000).astype(int).tolist() for i in range(12)],   # microvolts
            "findings": ev["findings"], "diagnoses": ev["diagnoses"], "flags": ev["flags"],
            "measure_vs_12sl": cmp_rows,
            "probs": [{"code": classes[j], "name": CLASS_NAMES.get(classes[j], classes[j]),
                       "p": round(float(p[j]), 3), "th": round(float(th[j]), 3)} for j in top],
            "reports": {
                "cardiologist": r["reference"],
                "cardiologist_lang": reports_en.at[eid, "lang"] if eid in reports_en.index else None,
                "cardiologist_original": reports_en.at[eid, "report_raw"] if eid in reports_en.index else None, "template": r["template"], "zero_shot": r["zero_shot"],
                "llm_raw": r["finetuned"],
                "llm_sentences": [{"text": s_["verified_text"], "original": s_["text"], "cites": s_["verified_citations"],
                                   "removed": s_["removed"], "hedged": s_["hedged"]} for s_ in g["sentences"]],
            },
        })
        print("added", eid, why)

    allm = json.load(open(cfg.path("artifacts") / "eval" / "metrics.json"))
    data = {"fs": FS_OUT, "leads": CANONICAL_LEADS, "cases": cases, "metrics": allm["reports"],
            "cnn_metrics": {k: allm["cnn"][k] for k in ("macro_auc_35", "superclass_auc", "superclass_macro_auc")},
            "measure_summary": allm["measurements_vs_references"]}
    for k in ("reference", "template", "zero_shot", "finetuned"):
        data["metrics"][k].pop("per_class_f1", None)
    html = open(cfg.path("artifacts").parent / "scripts" / "demo_template.html", encoding="utf-8").read()
    out = cfg.path("artifacts") / "demo"
    out.mkdir(exist_ok=True, parents=True)
    (out / "index.html").write_text(html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))),
                                    encoding="utf-8")
    print("wrote", out / "index.html", f"({(out / 'index.html').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
