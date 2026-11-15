"""Step 5 - merge measurements + CNN predictions + attributions into one evidence record per ECG.

Every claim carries: finding, value/unit (measurements) or probability (CNN), leads, a time window
in seconds of the original 10 s record, its source, and - for CNN claims - which measurement
claims support or contradict it.
"""
from __future__ import annotations

import math

import numpy as np

CLASS_NAMES = {
    "NORM": "normal ECG", "IMI": "inferior myocardial infarction", "AMI": "anterior myocardial infarction",
    "LMI": "lateral myocardial infarction", "PMI": "posterior myocardial infarction",
    "STTC": "ST/T changes", "NST_": "non-specific ST changes", "ISCA": "ischemic changes in anterior leads",
    "ISCI": "ischemic changes in inferior leads", "ISC_": "non-specific ischemic changes",
    "LAFB/LPFB": "left anterior/posterior fascicular block", "IRBBB": "incomplete right bundle branch block",
    "ILBBB": "incomplete left bundle branch block", "CLBBB": "complete left bundle branch block",
    "CRBBB": "complete right bundle branch block", "_AVB": "AV block",
    "IVCD": "non-specific intraventricular conduction delay", "WPW": "Wolff-Parkinson-White pattern",
    "LVH": "left ventricular hypertrophy", "RVH": "right ventricular hypertrophy",
    "LAO/LAE": "left atrial enlargement", "RAO/RAE": "right atrial enlargement", "SEHYP": "septal hypertrophy",
    "SR": "sinus rhythm", "AFIB": "atrial fibrillation", "STACH": "sinus tachycardia",
    "SARRH": "sinus arrhythmia", "SBRAD": "sinus bradycardia", "PACE": "pacemaker rhythm",
    "SVARR": "supraventricular arrhythmia", "BIGU": "bigeminal pattern", "AFLT": "atrial flutter",
    "SVTAC": "supraventricular tachycardia", "PSVT": "paroxysmal supraventricular tachycardia",
    "TRIGU": "trigeminal pattern",
}

TERRITORY = {
    "inferior": ["II", "III", "aVF"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "septal": ["V1", "V2"],
    "anterior": ["V3", "V4"],
}


SECONDARY_IF_WIDE = {"STE", "STD", "TINV", "QWAVE"}
ISCHEMIA_MI = {"IMI", "AMI", "LMI", "PMI", "ISCA", "ISCI", "ISC_", "STTC", "NST_"}
DRIFT_FLAG_MV = 0.3   # baseline wander above this (sens 0.82 / spec 0.75 vs PTB-XL drift annotations)


def _r(v, n=3):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), n)


def _ref_beat(m: dict) -> dict | None:
    """Representative beat for citations: dominant beat closest to the middle of the record."""
    beats = [b for b in m["beats"] if b.get("dominant")]
    if not beats:
        return None
    return min(beats, key=lambda b: abs(b["r_s"] - 5.0))


def _territories(leads: list[str]) -> list[str]:
    return [t for t, ls in TERRITORY.items() if sum(l in leads for l in ls) >= 2]


# --------------------------------------------------------------------------- measurement claims
def measurement_claims(m: dict, sex: int | None, age: float | None) -> list[dict]:
    """Rule-based findings with standard clinical criteria. sex: PTB-XL 0 = male, 1 = female."""
    if not m.get("ok"):
        return []
    g, L = m["globals"], m["leads"]
    rb = _ref_beat(m)
    span = [round(m["r_peaks_s"][0], 2), round(m["r_peaks_s"][-1], 2)]
    out = []

    noisy = {l for l, d in L.items() if d.get("baseline_wander_mv", 0) > DRIFT_FLAG_MV}
    # With QRS >= 120 ms (bundle branch block, pacing, IVCD) repolarisation and QRS shape are altered by
    # the conduction defect itself: ST/T shifts and QS complexes are expected and NOT diagnostic of
    # ischemia/infarction (standard ECG teaching; Sgarbossa criteria exist for exactly this reason).
    wide = g["qrs_ms"] >= 120

    def add(concept, finding, value, unit, criterion, leads, window, extra=None):
        c = {"id": f"M{len(out)+1}", "source": "measurement", "concept": concept, "finding": finding,
             "value": _r(value), "unit": unit, "criterion": criterion, "leads": leads,
             "window_s": window, **(extra or {})}
        if concept in ("STE", "STD", "TINV") and noisy & set(leads):
            c["low_quality_leads"] = sorted(noisy & set(leads))   # drift-corrected, but treat with caution
        if concept in SECONDARY_IF_WIDE and wide:
            c["secondary_to_wide_qrs"] = True
            c["finding"] += " (secondary to wide QRS)"
        out.append(c)

    qrs_w = rb["qrs_s"] if rb else span
    hr = g["hr_bpm"]
    add("HR", "heart rate", hr, "bpm", "mean RR", ["all"], span)
    if hr > 100:
        add("TACHY", "tachycardia", hr, "bpm", "> 100 bpm", ["all"], span)
    elif hr < 60:
        add("BRADY", "bradycardia", hr, "bpm", "< 60 bpm", ["all"], span)

    if g["n_beats"] >= 5 and g["rr_cv"] > 0.10 and g["rr_pnn_irregular"] > 0.3:
        add("IRREG", "irregular RR intervals", g["rr_cv"], "CV", "RR CV > 0.10 with > 30% successive changes > 80 ms",
            ["all"], span)
    if not g["p_present"] and g["n_beats"] >= 4:
        add("NO_P", "no consistent P waves", g["p_amp_mv"], "mV", "median-beat P magnitude < 0.045 mV",
            ["I", "II", "aVF", "V1", "V2"], span)

    pr = g["pr_ms"]
    if pr is not None and not math.isnan(pr):
        w = [round(qrs_w[0] - pr / 1000, 2), qrs_w[0]] if rb else span
        add("PR", "PR interval", pr, "ms", "P onset to QRS onset", ["global"], w)
        if pr > 200:
            add("LONG_PR", "prolonged PR interval", pr, "ms", "> 200 ms", ["global"], w)
        elif pr < 120:
            add("SHORT_PR", "short PR interval", pr, "ms", "< 120 ms", ["global"], w)

    qrs = g["qrs_ms"]
    add("QRS", "QRS duration", qrs, "ms", "QRS onset to offset", ["global"], qrs_w)
    if qrs >= 120:
        add("WIDE_QRS", "wide QRS complex", qrs, "ms", ">= 120 ms", ["global"], qrs_w)
    elif qrs >= 110:
        add("BORDER_QRS", "mildly prolonged QRS", qrs, "ms", "110-119 ms", ["global"], qrs_w)

    # Fridericia (QT / RR^(1/3)): guideline-preferred over Bazett, which over-corrects at high heart rates
    # (Bazett flagged 3051 training ECGs vs 1805 for Fridericia at similar sensitivity; see Step 5 report)
    qtc = g["qtc_fridericia_ms"]
    if qtc is not None and not math.isnan(qtc):
        lim = 460 if sex == 1 else 450
        qt_w = [qrs_w[0], rb["t_s"][1]] if rb and rb["t_s"][1] else span
        add("QTC", "QTc (Fridericia)", qtc, "ms", "QT / RR^(1/3)", ["global"], qt_w)
        if qtc > lim:
            add("LONG_QT", "prolonged QTc", qtc, "ms", f"> {lim} ms ({'female' if sex == 1 else 'male'})",
                ["global"], qt_w)

    ax = g["qrs_axis_deg"]
    add("AXIS", "frontal QRS axis", ax, "deg", "net QRS area in I and aVF", ["I", "aVF"], qrs_w)
    if -90 <= ax < -30:
        add("LAD", "left axis deviation", ax, "deg", "< -30 deg", ["I", "aVF"], qrs_w)
    elif 90 < ax <= 180:
        add("RAD", "right axis deviation", ax, "deg", "> +90 deg", ["I", "aVF"], qrs_w)
    elif -180 <= ax < -90:
        add("EXT_AX", "extreme axis deviation", ax, "deg", "-90 to -180 deg", ["I", "aVF"], qrs_w)

    # ---- ST elevation (4th Universal Definition of MI, J point)
    young_male = sex == 0 and (age is not None and age < 40)
    st_w = rb["st_s"] if rb else span
    elev = []
    for lead, d in L.items():
        thr = (0.25 if young_male else 0.2 if sex == 0 else 0.15) if lead in ("V2", "V3") else 0.1
        if d["st_j"] >= thr:
            elev.append((lead, d["st_j"], thr))
    if elev:
        leads = [e[0] for e in elev]
        terr = _territories(leads)
        add("STE", "ST elevation", max(e[1] for e in elev), "mV",
            ">= 0.1 mV (V2-V3: sex/age-specific) at J point", leads, st_w,
            {"per_lead": {e[0]: _r(e[1]) for e in elev}, "contiguous_territories": terr})
    dep = [(lead, d["st_j"]) for lead, d in L.items() if lead != "aVR" and d["st_j"] <= -0.05]
    if dep:
        add("STD", "ST depression", min(v for _, v in dep), "mV", "<= -0.05 mV at J point",
            [l for l, _ in dep], st_w, {"per_lead": {l: _r(v) for l, v in dep},
                                        "contiguous_territories": _territories([l for l, _ in dep])})

    t_w = rb["t_s"] if rb and rb["t_s"][1] else span
    inv = [(lead, L[lead]["t_amp"]) for lead in ("I", "II", "aVF", "V3", "V4", "V5", "V6")
           if L[lead]["t_amp"] <= -0.1]
    if inv:
        add("TINV", "T-wave inversion", min(v for _, v in inv), "mV", "<= -0.1 mV where T is normally upright",
            [l for l, _ in inv], t_w, {"per_lead": {l: _r(v) for l, v in inv},
                                        "contiguous_territories": _territories([l for l, _ in inv])})

    q = [(lead, d["q_dur_ms"], d["q_amp"]) for lead, d in L.items()
         if lead not in ("aVR", "III", "V1") and d["q_dur_ms"] >= 40 and d["q_amp"] <= -0.1]
    if q:
        add("QWAVE", "pathological Q waves", max(x[1] for x in q), "ms", "Q >= 40 ms and >= 0.1 mV deep",
            [x[0] for x in q], qrs_w, {"per_lead": {x[0]: _r(x[1], 0) for x in q},
                                        "contiguous_territories": _territories([x[0] for x in q])})

    cornell_lim = 2.0 if sex == 1 else 2.8
    if g["sokolow_lyon_mv"] >= 3.5 or g["cornell_mv"] >= cornell_lim:
        add("LVH_V", "LVH voltage criteria", max(g["sokolow_lyon_mv"], g["cornell_mv"]), "mV",
            f"Sokolow-Lyon {g['sokolow_lyon_mv']:.2f} (>=3.5) / Cornell {g['cornell_mv']:.2f} (>={cornell_lim})",
            ["V1", "V5", "V6", "aVL", "V3"], qrs_w)
    if g["limb_qrs_p2p_max_mv"] < 0.5:
        add("LOWV", "low QRS voltage (limb leads)", g["limb_qrs_p2p_max_mv"], "mV", "all limb leads < 0.5 mV",
            ["I", "II", "III", "aVR", "aVL", "aVF"], qrs_w)

    if g["n_non_dominant"] >= 1:
        nd = [b for b in m["beats"] if b.get("dominant") is False]
        add("ECTOPY", "beats with atypical morphology (possible ectopic beats)", g["n_non_dominant"], "beats",
            "template correlation < 0.90", ["all"],
            [nd[0]["qrs_s"][0], nd[0]["qrs_s"][1]] if nd else span,
            {"beat_times_s": [b["r_s"] for b in nd]})
    return out


# --------------------------------------------------------------------------- CNN claims
SUPPORT = {  # CNN class -> (supporting measurement concepts, contradicting concepts)
    "AFIB": ({"IRREG", "NO_P"}, set()), "AFLT": ({"NO_P"}, set()),
    "SR": (set(), {"NO_P"}), "STACH": ({"TACHY"}, {"BRADY"}), "SBRAD": ({"BRADY"}, {"TACHY"}),
    "SARRH": ({"IRREG"}, {"NO_P"}),
    "CLBBB": ({"WIDE_QRS"}, set()), "CRBBB": ({"WIDE_QRS"}, set()), "IVCD": ({"WIDE_QRS", "BORDER_QRS"}, set()),
    "IRBBB": ({"BORDER_QRS"}, set()), "ILBBB": ({"BORDER_QRS"}, set()),
    "_AVB": ({"LONG_PR"}, {"SHORT_PR"}), "WPW": ({"SHORT_PR", "WIDE_QRS", "BORDER_QRS"}, set()),
    "LVH": ({"LVH_V"}, {"LOWV"}), "LAFB/LPFB": ({"LAD", "RAD"}, set()), "PACE": ({"WIDE_QRS"}, set()),
    "IMI": ({"QWAVE", "STE", "TINV"}, set()), "AMI": ({"QWAVE", "STE", "TINV"}, set()),
    "LMI": ({"QWAVE", "STE", "TINV"}, set()), "PMI": ({"STD"}, set()),
    "ISCA": ({"STD", "TINV", "STE"}, set()), "ISCI": ({"STD", "TINV", "STE"}, set()),
    "ISC_": ({"STD", "TINV"}, set()), "STTC": ({"STD", "TINV"}, set()), "NST_": ({"STD", "TINV"}, set()),
    "NORM": (set(), {"WIDE_QRS", "STE", "QWAVE", "LONG_PR", "LONG_QT", "IRREG", "NO_P", "LVH_V", "TINV"}),
}
TERRITORY_OF = {"IMI": "inferior", "ISCI": "inferior", "AMI": "anterior", "ISCA": "anterior", "LMI": "lateral"}
NON_DIAGNOSTIC = {"HR", "PR", "QRS", "QTC", "AXIS"}


def cnn_claims(prob: np.ndarray, classes: list[str], thresholds: np.ndarray, mclaims: list[dict],
               attributions: dict | None = None) -> list[dict]:
    by_concept = {}
    for c in mclaims:
        by_concept.setdefault(c["concept"], []).append(c)
    out = []
    order = np.argsort(-prob)
    for j in order:
        if prob[j] < thresholds[j]:
            continue
        cls = classes[j]
        sup, con = SUPPORT.get(cls, (set(), set()))
        s_ids, c_ids = [], []
        for concept in sup:
            for mc in by_concept.get(concept, []):
                terr = TERRITORY_OF.get(cls)
                if terr and mc.get("contiguous_territories") is not None and terr not in mc["contiguous_territories"]:
                    continue  # e.g. inferior MI is only supported by inferior-territory changes
                if cls in ISCHEMIA_MI and mc.get("secondary_to_wide_qrs"):
                    continue  # conduction-related ST/Q changes cannot support ischemia or infarction
                s_ids.append(mc["id"])
        for concept in con:
            c_ids += [mc["id"] for mc in by_concept.get(concept, [])]
        claim = {"id": f"C{len(out)+1}", "source": "cnn", "concept": cls, "finding": CLASS_NAMES.get(cls, cls),
                 "probability": round(float(prob[j]), 3), "threshold": round(float(thresholds[j]), 3),
                 "supported_by": sorted(set(s_ids)), "contradicted_by": sorted(set(c_ids))}
        if attributions and cls in attributions:
            claim.update({"leads": attributions[cls]["top_leads"], "window_s": attributions[cls]["window_s"],
                          "lead_share": attributions[cls]["lead_share"]})
        out.append(claim)
    return out


def build_evidence(ecg_id: int, meta_row: dict, m: dict, prob: np.ndarray | None, classes: list[str] | None,
                   thresholds: np.ndarray | None, attributions: dict | None = None) -> dict:
    sex = meta_row.get("sex")
    age = meta_row.get("age")
    mc = measurement_claims(m, sex, age)
    cc = cnn_claims(prob, classes, thresholds, mc, attributions) if prob is not None else []
    flags = []
    if not m.get("ok"):
        flags.append(f"measurement failed: {m.get('reason')}")
    for c in cc:
        if c["contradicted_by"]:
            flags.append(f"{c['id']} ({c['concept']}) conflicts with {', '.join(c['contradicted_by'])}")
    if prob is not None and not cc:
        flags.append("no diagnostic class above threshold")
    return {
        "ecg_id": int(ecg_id),
        "patient": {"age": None if age is None or (isinstance(age, float) and math.isnan(age)) else int(age),
                    "sex": {0: "male", 1: "female"}.get(sex)},
        "quality": {"beats": m.get("globals", {}).get("n_beats"),
                    "dominant_beats": m.get("globals", {}).get("n_dominant")},
        "findings": mc,
        "diagnoses": cc,
        "flags": flags,
    }


def evidence_to_text(ev: dict) -> str:
    """Compact, deterministic serialisation used as the LLM prompt input."""
    p = ev["patient"]
    lines = [f"Patient: {p['age'] if p['age'] is not None else '?'} y, {p['sex'] or 'unknown sex'}"]
    lines.append("Measurements:")
    for c in ev["findings"]:
        v = "" if c["value"] is None else f" {c['value']} {c['unit']}"
        lead = "" if c["leads"] in (["all"], ["global"]) else f" leads {','.join(c['leads'])}"
        terr = f" territories {','.join(c['contiguous_territories'])}" if c.get("contiguous_territories") else ""
        lines.append(f"[{c['id']}] {c['finding']}{v}{lead}{terr} @ {c['window_s'][0]}-{c['window_s'][1]}s")
    lines.append("Classifier:")
    for c in ev["diagnoses"]:
        sup = f" supported_by {','.join(c['supported_by'])}" if c["supported_by"] else ""
        con = f" contradicted_by {','.join(c['contradicted_by'])}" if c["contradicted_by"] else ""
        loc = f" leads {','.join(c['leads'])} @ {c['window_s'][0]}-{c['window_s'][1]}s" if c.get("leads") else ""
        lines.append(f"[{c['id']}] {c['finding']} p={c['probability']}{loc}{sup}{con}")
    if ev["flags"]:
        lines.append("Flags: " + "; ".join(ev["flags"]))
    return "\n".join(lines)
