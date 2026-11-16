"""Showcase: pick the best test-fold case per diagnosis and render a one-page output figure for each.

Selection (all automatic, test fold only, from existing outputs; nothing is re-run or edited):
  * labels validated by a cardiologist, and a cardiologist report exists
  * the target diagnosis is in the cardiologist's labels AND the CNN reports it (p >= its threshold)
    AND the verified CLAIMS report states it
  * score = agreement of the verified report with the labels (F1 over the 35 classes)
          + bonus if the CNN finding is supported by a measurement
          + bonus for clean signal (no baseline-wander flags, most beats dominant)
          - penalty for statements that are not in the labels
Two teaching cases are added: one where the verifier hedged a finding, one where it removed a phrase.

Outputs: showcase/ecg_<id>_<label>.png, showcase/overview.png, showcase/cases.json, showcase/README.md
"""
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from claims.claims_builder import CLASS_NAMES
from claims.config import load_config
from claims.data import CANONICAL_LEADS, load_metadata, load_signals, to_mv
from claims.grounding import extract_concepts, ground_report
from claims.signal import clean_for_measurement
from claims.validate_measure import GLOBAL_MAP, load_12sl

TARGETS = [  # (class code, category, title)
    ("NORM", "Normal", "Normal ECG"),
    ("AFIB", "Rhythm", "Atrial fibrillation"),
    ("SBRAD", "Rhythm", "Sinus bradycardia"),
    ("STACH", "Rhythm", "Sinus tachycardia"),
    ("PACE", "Rhythm", "Paced rhythm"),
    ("CLBBB", "Conduction", "Complete left bundle branch block"),
    ("CRBBB", "Conduction", "Complete right bundle branch block"),
    ("LAFB/LPFB", "Conduction", "Left anterior fascicular block"),
    ("_AVB", "Conduction", "AV block"),
    ("IMI", "Infarction", "Inferior myocardial infarction"),
    ("AMI", "Infarction", "Anterior myocardial infarction"),
    ("LVH", "Hypertrophy", "Left ventricular hypertrophy"),
    ("ISCA", "Ischaemia", "Anterior ischaemia"),
    ("NST_", "Ischaemia", "Non-specific ST/T changes"),
]
CITE = re.compile(r"\s*\[[MC]\d+(?:, [MC]\d+)*\]")
INK, INK2, MUTED, LINE = "#0f1b2d", "#3d4b5f", "#6b7a8f", "#e3e8ef"
ACCENT, MEAS, CNN = "#0f766e", "#c2410c", "#1d4ed8"
GRID_MINOR, GRID_MAJOR = "#f6dcdd", "#eab8ba"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})


def load(cfg):
    evs = {}
    for line in open(cfg.path("artifacts") / "evidence" / "evidence_test.jsonl", encoding="utf-8"):
        e = json.loads(line)
        evs[e["ecg_id"]] = e
    reps = {}
    for line in open(cfg.path("artifacts") / "eval" / "reports_test_grounded.jsonl", encoding="utf-8"):
        r = json.loads(line)
        reps[r["ecg_id"]] = r
    return evs, reps


def score_cases(evs, reps, labels, meta, classes):
    rows = []
    cls = set(classes)
    for eid, r in reps.items():
        if not meta.at[eid, "validated_by_human"] or not r.get("reference"):
            continue
        ev = evs[eid]
        g = ground_report(r["finetuned"], ev)
        verified = CITE.sub("", g["verified_report"])
        said = extract_concepts(verified) & cls
        truth = (set(labels.at[eid, "subclass"]) | set(labels.at[eid, "rhythm"])) & cls
        tp = len(said & truth)
        f1 = 2 * tp / max(1, len(said) + len(truth))
        dx = {d["concept"]: d for d in ev["diagnoses"]}
        lowq = any(f.get("low_quality_leads") for f in ev["findings"])
        q = ev["quality"]
        dom = (q.get("dominant_beats") or 0) / max(1, q.get("beats") or 1)
        rows.append({"ecg_id": eid, "said": said, "truth": truth, "f1": f1, "extra": len(said - truth),
                     "dx": dx, "clean": (not lowq) and not ev["flags"] and dom >= .8, "g": g,
                     "hedged": any(s["hedged"] for s in g["sentences"]),
                     "removed": any(s["removed"] for s in g["sentences"]),
                     "n_words": len(verified.split())})
    return rows


def pick(rows):
    chosen, used = [], set()
    for code, cat, title in TARGETS:
        cands = []
        for r in rows:
            d = r["dx"].get(code)
            if r["ecg_id"] in used or code not in r["truth"] or code not in r["said"] or d is None:
                continue
            s = 2 * r["f1"] + (1.0 if d["supported_by"] else 0) + (0.7 if r["clean"] else 0) - 0.5 * r["extra"] \
                + 0.3 * d["probability"] - (0.3 if r["n_words"] > 40 else 0)
            cands.append((s, r))
        if not cands:
            print(f"  no qualifying case for {code}")
            continue
        s, r = max(cands, key=lambda x: x[0])
        used.add(r["ecg_id"])
        chosen.append({"ecg_id": r["ecg_id"], "code": code, "category": cat, "title": title, "score": round(s, 3),
                       "f1": round(r["f1"], 3), "why": "best case for " + title})
        print(f"  {code:10s} ECG {r['ecg_id']:>5}  score {s:.2f}  F1 {r['f1']:.2f}  supported {bool(r['dx'][code]['supported_by'])}")
    for kind, title in (("hedged", "Verifier hedges a low-confidence finding"), ("removed", "Verifier removes an unsupported phrase")):
        cands = [r for r in rows if r[kind] and r["ecg_id"] not in used and r["clean"]]
        if cands:
            r = max(cands, key=lambda r: r["f1"] - 0.3 * r["extra"])
            used.add(r["ecg_id"])
            chosen.append({"ecg_id": r["ecg_id"], "code": None, "category": "Verifier", "title": title,
                           "score": None, "f1": round(r["f1"], 3), "why": title})
            print(f"  {kind:10s} ECG {r['ecg_id']:>5}  F1 {r['f1']:.2f}")
    return chosen


# ----------------------------------------------------------------------------- figure
def wrap(s, w):
    return "\n".join(textwrap.wrap(s, w)) if s else ""


def draw_case(case, ev, rep, g, x_mv, ref_row, out_png):
    eid = case["ecg_id"]
    fig = plt.figure(figsize=(17, 11), dpi=150, facecolor="white")
    # ---------------- header
    p = ev["patient"]
    fig.text(.03, .955, f"ECG {eid}  ·  {case['title']}", fontsize=20, fontweight="semibold", color=INK)
    sub = f"{p['age']} y · {p['sex'] or 'sex not recorded'} · PTB-XL held-out test fold · labels validated by a cardiologist"
    fig.text(.03, .925, sub, fontsize=11, color=MUTED)
    fig.text(.97, .955, case["category"].upper(), fontsize=10, color=ACCENT, fontweight="bold", ha="right")

    # ---------------- ECG panel (12 x 10 s, standard grid)
    ax = fig.add_axes([.03, .05, .60, .855])
    p2p = np.percentile(x_mv, 99.5, axis=0) - np.percentile(x_mv, .5, axis=0)
    gain = 5 if p2p.max() > 3.2 else 10
    k = gain / 10                                             # display scaling of mV
    spacing = 2.0
    for t in np.arange(0, 10.0001, .04):
        major = abs(t / .2 - round(t / .2)) < 1e-6
        ax.axvline(t, color=GRID_MAJOR if major else GRID_MINOR, lw=.7 if major else .35, zorder=0)
    lo, hi = -spacing * 11 - 1.3, 1.3
    for v in np.arange(np.floor(lo * 10) / 10, hi + .001, .1):
        major = abs(v * 2 - round(v * 2)) < 1e-6
        ax.axhline(v, color=GRID_MAJOR if major else GRID_MINOR, lw=.7 if major else .35, zorder=0)
    t = np.arange(x_mv.shape[0]) / 500
    cited = []
    for s in g["sentences"]:
        cited += s.get("verified_citations", [])
    items = {c["id"]: c for c in ev["findings"] + ev["diagnoses"]}
    show = list(dict.fromkeys(cited + [m for c in cited if c.startswith("C") for m in items[c]["supported_by"]]))
    # evidence regions under the trace
    tag_y = {}
    for iid in show:
        it = items[iid]
        leads = [l for l in it.get("leads", []) if l not in ("all", "global")]
        a, b = it["window_s"]
        if not leads or b - a > 3:
            continue
        isM = iid.startswith("M")
        for j, l in enumerate(sorted(leads, key=CANONICAL_LEADS.index)):
            y0 = -spacing * CANONICAL_LEADS.index(l)
            ax.add_patch(Rectangle((a, y0 - .95), max(b - a, .03), 1.9, facecolor=(MEAS if isM else CNN), alpha=.13,
                                   edgecolor=MEAS if isM else CNN, lw=1.2, ls="-" if isM else (0, (4, 2)), zorder=1))
            if j == 0:
                key = (round(a, 1), l)
                off = tag_y.get(key, 0)
                tag_y[key] = off + 1
                ax.text(a + .01, y0 + .95 - .32 * off, iid, fontsize=8, color="white", fontweight="bold", va="top",
                        family="monospace", zorder=5,
                        bbox=dict(boxstyle="square,pad=.18", fc=MEAS if isM else CNN, ec="none"))
    for i, l in enumerate(CANONICAL_LEADS):
        y = x_mv[:, i] - np.median(x_mv[:, i])
        y = np.clip(y * k, -1.25, 1.25) - spacing * i
        ax.plot(t, y, color="#111827", lw=.8, zorder=3, solid_joinstyle="round")
        ax.text(-.08, -spacing * i, l, ha="right", va="center", fontsize=11, color=INK2, fontweight="bold")
    # whole-ECG interval measurements cited by the report: bracket under the last lead
    yb = -spacing * 11 - 1.0
    for iid in show:
        it = items[iid]
        if iid.startswith("M") and it.get("leads") in (["global"], ["all"]) and it["window_s"][1] - it["window_s"][0] < 1:
            a, b = it["window_s"]
            ax.plot([a, a, b, b], [yb + .15, yb, yb, yb + .15], color=MEAS, lw=1.6, zorder=4)
            ax.text(b + .06, yb + .02, f"{iid}  {it['finding']} {it['value']:.0f} {it['unit']}", ha="left", va="center",
                    fontsize=8.5, color=INK2, zorder=5,
                    bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.9))
    ax.set_xlim(-.02, 10.02)
    ax.set_ylim(lo - .35, hi)
    ax.set_aspect(.4)
    ax.set_xticks(range(11))
    ax.set_xticklabels([f"{s} s" for s in range(11)], fontsize=9, color=MUTED)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"25 mm/s · {gain} mm/mV · small box 0.04 s / 0.1 mV", fontsize=9, color=MUTED, loc="right", pad=4)

    # ---------------- right panel
    R = fig.add_axes([.655, .05, .32, .855])
    R.axis("off")
    R.set_xlim(0, 1)
    R.set_ylim(0, 1)
    LH = 1.45 / (.855 * 11 * 72)          # axes-fraction height of one text line, per point of font size
    y = 1.0

    def heading(txt, color=MUTED):
        nonlocal y
        R.text(0, y, txt.upper(), fontsize=9, fontweight="bold", color=color, va="top")
        y -= .03

    def para(txt, size=11.5, color=INK, width=52, gap=.012, weight="normal"):
        nonlocal y
        w = wrap(txt, width)
        R.text(0, y, w, fontsize=size, color=color, va="top", linespacing=1.45, fontweight=weight)
        y -= (w.count("\n") + 1) * size * LH + gap

    # verified report box
    verified = [(s["verified_text"], s.get("verified_citations", []), s["hedged"]) for s in g["sentences"] if s["verified_text"]]
    vtxt = " ".join(f"{s[0]} [{', '.join(s[1])}]" if s[1] else s[0] for s in verified)
    top = y
    heading("CLAIMS output · fine-tuned LLM, verified", ACCENT)
    para(vtxt[0].upper() + vtxt[1:] if vtxt else "(nothing survived verification)", size=13.5, width=40, weight="semibold")
    notes = []
    if any(s[2] for s in verified):
        notes.append("“possible” added by the verifier: CNN probability below 0.5, no supporting measurement")
    removed = [r for s in g["sentences"] for r in s["removed"]]
    if removed:
        notes.append("removed by the verifier (not in the evidence): " + "; ".join(removed))
    for n in notes:
        para(n, size=9.5, color=MEAS if "removed" in n else "#a16207", width=62, gap=.004)
    R.add_patch(FancyBboxPatch((-.02, y + .002), 1.04, top - y + .012, boxstyle="round,pad=0.008,rounding_size=0.012",
                               fc="#e6f4f2", ec=ACCENT, lw=1.2, zorder=-1, transform=R.transAxes))
    y -= .03

    heading("Evidence behind the report")
    for iid in show:
        it = items[iid]
        if iid.startswith("C"):
            line = f"{CLASS_NAMES.get(it['concept'], it['finding'])} — p {it['probability']:.2f} (threshold {it['threshold']:.2f})"
            if it["supported_by"]:
                line += f"; supported by {', '.join(it['supported_by'])}"
            leads = [l for l in it.get("leads", []) if l not in ("all", "global")]
            if leads:
                line += f"; CNN looked at {', '.join(leads)}"
        else:
            val = "" if it["value"] is None else (f" {it['value']:.2f} {it['unit']}" if it["unit"] == "mV" else f" {it['value']:.0f} {it['unit']}")
            leads = [l for l in it.get("leads", []) if l not in ("all", "global")]
            line = f"{it['finding']}{val}" + (f" in {', '.join(leads)}" if leads else "") + f" (criterion: {it['criterion']})"
        R.text(0, y, iid, fontsize=8.5, color="white", fontweight="bold", family="monospace", va="top",
               bbox=dict(boxstyle="round,pad=.25", fc=MEAS if iid.startswith("M") else CNN, ec="none"))
        w = wrap(line.replace(">=", "≥").replace("<=", "≤"), 56)
        R.text(.075, y + .002, w, fontsize=9.8, color=INK2, va="top", linespacing=1.4)
        y -= (w.count("\n") + 1) * 9.8 * LH + .016
    y -= .015

    heading("Cardiologist report (PTB-XL)")
    para(rep["reference"], size=11, color=INK, width=56, gap=.004)
    lang = {"de": "German", "sv": "Swedish"}.get(rep.get("_lang"))
    para(f"machine-translated from {lang} for this project" if lang else "original English report", size=9, color=MUTED, gap=.02)

    heading("Measurements: CLAIMS vs GE 12SL")
    ours = rep.get("_globals", {})
    names = {"hr_bpm": ("Heart rate", "bpm"), "pr_ms": ("PR", "ms"), "qrs_ms": ("QRS", "ms"), "qt_ms": ("QT", "ms"),
             "qrs_axis_deg": ("Axis", "°")}
    cols = [0, .42, .66]
    R.text(cols[1], y, "CLAIMS", fontsize=9, color=MUTED, va="top", fontweight="bold")
    R.text(cols[2], y, "GE 12SL", fontsize=9, color=MUTED, va="top", fontweight="bold")
    y -= .026
    for key, (nm, u) in names.items():
        a = ours.get(key)
        b = ref_row.get(GLOBAL_MAP[key]) if ref_row is not None else None
        if key == "qrs_axis_deg" and (b is None or pd.isna(b)) and ref_row is not None:
            b = ref_row.get("QRS_AxisFront_Global")
        fa = "not available" if a is None or pd.isna(a) else f"{a:.0f} {u}".replace(" °", "°")
        fb = "not available" if b is None or pd.isna(b) else f"{b:.0f} {u}".replace(" °", "°")
        R.text(cols[0], y, nm, fontsize=10, color=INK2, va="top")
        R.text(cols[1], y, fa, fontsize=10, color=INK, va="top")
        R.text(cols[2], y, fb, fontsize=10, color=INK, va="top")
        y -= .024

    # footer / legend
    for x0, col, ls, label in ((.03, MEAS, "-", "measurement window [M#]"),
                               (.16, CNN, (0, (3, 2)), "CNN attribution region [C#] (Integrated Gradients)")):
        fig.patches.append(Rectangle((x0, .019), .012, .014, transform=fig.transFigure, fc=col, alpha=.18, ec=col, lw=1.1, ls=ls))
        fig.text(x0 + .017, .021, label, color=INK2, fontsize=9)
    fig.text(.97, .022, "CLAIMS research prototype · not a medical device · data: PTB-XL / PTB-XL+ (CC BY 4.0)",
             color=MUTED, fontsize=8.5, ha="right")
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)


def overview(cases, out_dir):
    from PIL import Image
    ims = [Image.open(out_dir / c["file"]) for c in cases]
    w = 900
    ims = [im.resize((w, int(im.height * w / im.width)), Image.LANCZOS) for im in ims]
    cols = 2
    rows = (len(ims) + cols - 1) // cols
    h = ims[0].height
    sheet = Image.new("RGB", (cols * w + (cols + 1) * 20, rows * h + (rows + 1) * 20), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, (20 + (i % cols) * (w + 20), 20 + (i // cols) * (h + 20)))
    sheet.save(out_dir / "overview.png", optimize=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    meta = load_metadata(cfg)
    labels = pd.read_pickle(cfg.path("cache") / "labels.pkl")
    evs, reps = load(cfg)
    import torch
    classes = torch.load(cfg.path("artifacts") / "cnn" / "cnn_final.pt", map_location="cpu", weights_only=False)["classes"]
    print("scoring", len(reps), "test reports")
    rows = score_cases(evs, reps, labels, meta, classes)
    print(len(rows), "candidates (validated, with cardiologist report)")
    cases = pick(rows)

    out = cfg.path("artifacts").parent / "showcase"
    out.mkdir(exist_ok=True)
    sig = load_signals(cfg, meta)
    meas = json.load(open(cfg.path("artifacts") / "measurements" / "measurements_full.json"))
    ref = load_12sl(cfg.path("ptbxl_plus") / "12sl_features.csv")
    rep_en = pd.read_parquet(cfg.path("artifacts") / "reports_en.parquet")
    s = cfg.signal
    by_id = {r["ecg_id"]: r for r in rows}
    cdir = cfg.path("artifacts") / "cnn"
    probs = dict(zip(np.load(cdir / "ids_test.npy").tolist(), np.load(cdir / "pred_test.npy")))
    thresholds = np.asarray(json.load(open(cdir / "thresholds.json"))["thresholds"])
    sys.path.insert(0, str(Path(__file__).parent))
    from showcase_figure import draw_card
    for c in cases:
        eid = c["ecg_id"]
        ev, rep, g = evs[eid], dict(reps[eid]), by_id[eid]["g"]
        rep["_lang"] = rep_en.at[eid, "lang"] if eid in rep_en.index else None
        m = meas[str(eid)]
        rep["_globals"] = m.get("globals", {}) if m.get("ok") else {}
        x = clean_for_measurement(to_mv(sig[meta.index.get_loc(eid)]), 500, s.highpass_hz, s.lowpass_hz, s.notch_hz)
        slug = re.sub(r"[^a-z0-9]+", "_", c["title"].lower()).strip("_")
        c["file"] = f"ecg_{eid}_{slug}.png"
        c["_reference"] = rep["reference"]
        draw_card(c, ev, g, x, rep["_globals"], ref.loc[eid] if eid in ref.index else None,
                  probs[eid], classes, thresholds, rep["_lang"], out / c["file"])
        c.pop("_reference")
        c.update({"age": ev["patient"]["age"], "sex": ev["patient"]["sex"],
                  "cardiologist_labels": sorted(by_id[eid]["truth"]),
                  "claims_verified_report": g["verified_report"],
                  "fine_tuned_raw": rep["finetuned"], "template": rep["template"], "cardiologist_report": rep["reference"],
                  "cardiologist_report_language": rep["_lang"],
                  "evidence": {"findings": ev["findings"], "diagnoses": ev["diagnoses"]}})
        print("  wrote", c["file"])
    overview(cases, out)
    json.dump(cases, open(out / "cases.json", "w", encoding="utf-8"), indent=1, default=float)

    lines = ["# CLAIMS showcase", "",
             "The best case per diagnosis from the **held-out PTB-XL test fold**, chosen automatically by "
             "`scripts/s10_showcase.py` (labels validated by a cardiologist; the CNN reports the diagnosis; the verified "
             "CLAIMS report states it and agrees with the cardiologist's labels; measurement support and clean signal "
             "preferred). Two extra cases show the verifier at work.", "",
             "| ECG | Diagnosis | CLAIMS verified report | Cardiologist report |", "|---|---|---|---|"]
    for c in cases:
        v = CITE.sub("", c["claims_verified_report"]).replace("|", "/")
        lines.append(f"| [{c['ecg_id']}]({c['file']}) | {c['title']} | {v} | {c['cardiologist_report'].replace('|', '/')} |")
    lines += ["", "![overview](overview.png)", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
