"""Step 7 - concept extraction from report text, grounding check and citation attachment.

* ``extract_concepts(text)``: rule-based clinical concept extractor with negation handling.
  Concept codes = the 35 CNN classes + measurement concepts (TACHY, LAD, LONG_QT, ...).
* ``ground_report(text, evidence)``: per sentence, which concepts are mentioned, which evidence
  items support them, and which are unsupported (hallucinations). Returns the report with
  citations appended to each sentence and unsupported sentences flagged.
* Hedging: a concept whose only evidence is a CNN diagnosis with probability < 0.5 and no
  measurement support is "low confidence". If the text states it without a hedge word, the
  verified report inserts "possible" in front of the mention.
"""
from __future__ import annotations

import re

_LOC = {
    "inferior": "I", "inferolateral": "IL", "inferoposterior": "I", "anterior": "A", "anteroseptal": "A", "anteroseptic": "A",
    "septal": "A", "anterolateral": "AL", "lateral": "L", "high lateral": "L", "posterior": "P", "front": "A",
    "front wall": "A", "rear wall": "I", "posterior wall": "P",
}

PATTERNS: list[tuple[str, str]] = [
    # rhythm
    ("STACH", r"sinus tachycardi"), ("SBRAD", r"sinus bradycardi"), ("SARRH", r"sinus arrhythmi"),
    ("SR", r"sinus rhythm"), ("AFIB", r"atrial fibrillation|fibrillation/flutter"), ("AFLT", r"atrial flutter"),
    ("PACE", r"pacemaker|paced (beat|rhythm)|pacing"), ("BIGU", r"bigemin"), ("TRIGU", r"trigemin"),
    ("SVTAC", r"supraventricular tachycardi"), ("SVARR", r"supraventricular (extrasystole|premature|ectop|arrhythmi)"),
    ("ECTOPY", r"extrasystole|premature (ventricular|atrial|supraventricular)|ectopic|extra beats|premature beat"),
    ("TACHY", r"(?<!sinus )(?<!supraventricular )tachycardi"), ("BRADY", r"(?<!sinus )bradycardi"),
    # normal
    ("NORM", r"normal ecg|normal ekg|no definite pathology|within normal limits|normal electrocardiogram"),
    # conduction
    ("ILBBB", r"incomplete left bun(dle|ch)"), ("IRBBB", r"incomplete right bun(dle|ch)"),
    ("CLBBB", r"(?<!incomplete )left bun(dle|ch)[- ]?(branch )?block|\blbbb"),
    ("CRBBB", r"(?<!incomplete )right bun(dle|ch)[- ]?(branch )?block|\brbbb"),
    ("LAFB/LPFB", r"(left )?(anterior|posterior) (fascicular|hemi)[- ]?block|hemiblock|fascicular block"),
    ("_AVB", r"\ba-?v-?\s?block|atrioventricular block|first[- ]degree|prolonged pr|pr[- ]?(interval )?prolong"),
    ("IVCD", r"intraventricular (conduction|block)|conduction (disturbance|delay)"),
    ("WPW", r"\bwpw|wolff|pre-?excitation"),
    # hypertrophy / atrial
    ("LVH", r"left ventricular hypertroph|\blvh\b|left hypertroph|lv hypertroph"),
    ("RVH", r"right ventricular hypertroph|\brvh\b"),
    ("LAO/LAE", r"left atri\w* (enlargement|abnormality|overload|hypertroph)|p[- ]mitrale|sinistrocardial"),
    ("RAO/RAE", r"right atri\w* (enlargement|abnormality|overload|hypertroph)|p[- ]pulmonale|dextrocardial"),
    ("SEHYP", r"septal hypertroph"),
    # axis / intervals / voltage
    ("LAD", r"left axis deviation|extreme left axis"), ("RAD", r"right axis deviation"),
    ("LONG_QT", r"qt[- ]?(prolong|extension|lengthening)|prolonged qt|long qt"),
    ("LONG_PR", r"pr[- ]?(prolong)|prolonged pr|long pr"), ("SHORT_PR", r"short pr"),
    ("LOWV", r"low (qrs )?voltage|low amplitude"), ("WIDE_QRS", r"wide qrs|broad qrs|qrs widening"),
    ("QWAVE", r"\bq waves?|qs complex|\bqs\b"),
    # ST/T
    ("STE", r"st[- ]?(segment )?elevat|st[- ]?raising|st-hebung"),
    ("STD", r"st[- ]?(segment )?(depress|lower)|st segments are depressed"),
    ("TINV", r"t[- ]?waves? (are )?invert|inverted t|negative t|t[- ]?inversion"),
    ("NST_", r"non-?specific (st|t|repolari)|unspecific (st|t)|nonspecific (st|t)|st-t (wave )?(change|abnormal)|"
             r"t (wave )?(abnormal|change|flatten)|flat t|abnormal t|st change|repolari[sz]ation (disturbance|abnormal)|"
             r"ventricular strain|lv strain|st ?& ?t abnormal"),
]
_COMPILED = [(c, re.compile(p)) for c, p in PATTERNS]
_INFARCT = re.compile(r"infarct|myocardial infarction|\bmi\b")
_ISCH = re.compile(r"isch(a)?emi|myocardial (injury|damage)|subendocardial")
_NEG = re.compile(r"\b(no|not|without|absence of|negative for|rather than)\b[^.;,]{0,25}$")


_LOC_RX = re.compile(r"(?<![a-z])(%s)(?![a-z])" % "|".join(sorted(map(re.escape, _LOC), key=len, reverse=True)))


def _location_match(prefix: str):
    """Last whole-word location phrase in the 45 characters before a mention (re.Match or None)."""
    off = max(0, len(prefix) - 45)
    ms = list(_LOC_RX.finditer(prefix, off))
    return ms[-1] if ms else None


def _location(prefix: str) -> str | None:
    m = _location_match(prefix)
    return _LOC[m.group(1)] if m else None


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_concepts(text: str) -> set[str]:
    t = " " + text.lower() + " "
    found = set()
    spans = []
    for code, rx in _COMPILED:
        for m in rx.finditer(t):
            if any(a <= m.start() < b for a, b in spans) and code in ("TACHY", "BRADY", "NST_"):
                continue
            if _NEG.search(t[max(0, m.start() - 40):m.start()]):
                continue
            found.add(code)
            spans.append((m.start(), m.end()))
    for m in _INFARCT.finditer(t):
        if _NEG.search(t[max(0, m.start() - 40):m.start()]):
            continue
        loc = _location(t[:m.start()])
        found |= {"I": {"IMI"}, "IL": {"IMI", "LMI"}, "A": {"AMI"}, "AL": {"AMI", "LMI"}, "L": {"LMI"},
                  "P": {"PMI"}}.get(loc, {"MI_UNSPEC"})
    for m in _ISCH.finditer(t):
        if _NEG.search(t[max(0, m.start() - 40):m.start()]):
            continue
        loc = _location(t[:m.start()]) or _location(t[m.end():m.end() + 30][::-1][::-1])
        found |= {"I": {"ISCI"}, "IL": {"ISCI"}, "A": {"ISCA"}, "AL": {"ISCA"}, "L": {"ISCA"}}.get(loc, {"ISC_"})
    if "SR" in found and found & {"STACH", "SBRAD", "SARRH"}:
        pass
    return found


# report concept -> evidence concepts that can support it (CNN classes and/or measurement concepts)
SUPPORTED_BY: dict[str, set[str]] = {
    "SR": {"SR", "SARRH", "STACH", "SBRAD"}, "STACH": {"STACH", "TACHY"}, "SBRAD": {"SBRAD", "BRADY"},
    "SARRH": {"SARRH"}, "AFIB": {"AFIB"}, "AFLT": {"AFLT", "AFIB"}, "PACE": {"PACE"}, "BIGU": {"BIGU"},
    "TRIGU": {"TRIGU"}, "SVTAC": {"SVTAC", "PSVT"}, "SVARR": {"SVARR", "ECTOPY"},
    "ECTOPY": {"ECTOPY", "SVARR", "BIGU", "TRIGU"}, "TACHY": {"TACHY", "STACH", "SVTAC", "PSVT"},
    "BRADY": {"BRADY", "SBRAD"}, "NORM": {"NORM"},
    "ILBBB": {"ILBBB"}, "IRBBB": {"IRBBB"}, "CLBBB": {"CLBBB"}, "CRBBB": {"CRBBB"},
    "LAFB/LPFB": {"LAFB/LPFB"}, "_AVB": {"_AVB", "LONG_PR"}, "IVCD": {"IVCD", "WIDE_QRS", "BORDER_QRS"},
    "WPW": {"WPW"}, "LVH": {"LVH", "LVH_V"}, "RVH": {"RVH"}, "LAO/LAE": {"LAO/LAE"}, "RAO/RAE": {"RAO/RAE"},
    "SEHYP": {"SEHYP"}, "LAD": {"LAD", "EXT_AX"}, "RAD": {"RAD"}, "LONG_QT": {"LONG_QT"},
    "LONG_PR": {"LONG_PR", "_AVB"}, "SHORT_PR": {"SHORT_PR", "WPW"}, "LOWV": {"LOWV"},
    "WIDE_QRS": {"WIDE_QRS", "CLBBB", "CRBBB", "IVCD", "PACE"}, "QWAVE": {"QWAVE", "IMI", "AMI", "LMI"},
    "STE": {"STE"}, "STD": {"STD"}, "TINV": {"TINV"},
    "NST_": {"NST_", "STTC", "ISC_", "ISCA", "ISCI", "STD", "TINV"},
    "IMI": {"IMI"}, "AMI": {"AMI"}, "LMI": {"LMI"}, "PMI": {"PMI"}, "MI_UNSPEC": {"IMI", "AMI", "LMI", "PMI"},
    "ISCA": {"ISCA", "ISC_", "STTC"}, "ISCI": {"ISCI", "ISC_", "STTC"},
    "ISC_": {"ISC_", "ISCA", "ISCI", "STTC", "NST_", "STD", "TINV"},
}


def evidence_index(ev: dict) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for c in ev["findings"] + ev["diagnoses"]:
        idx.setdefault(c["concept"], []).append(c["id"])
    return idx


LOW_CONF_P = 0.5
_HEDGE = re.compile(r"\b(possibl[ey]|probabl[ey]|may|might|suspect\w*|consider\w*|suggest\w*|cannot be excluded|"
                    r"borderline|likely|questionable|not excluded)\b|\?")
_LOC_TO_CODES = {"I": {"IMI", "ISCI"}, "IL": {"IMI", "LMI", "ISCI"}, "A": {"AMI", "ISCA"},
                 "AL": {"AMI", "LMI", "ISCA"}, "L": {"LMI", "ISCA"}, "P": {"PMI"}}


def _mention_spans(s: str, code: str) -> list[tuple[int, int]]:
    """(start, end) spans in s of every non-negated mention of `code`."""
    t = s.lower()
    out = []
    rx = dict(_COMPILED).get(code)
    if rx is not None:
        return [(m.start(), m.end()) for m in rx.finditer(t) if not _NEG.search(t[max(0, m.start() - 40):m.start()])]
    big = _ISCH if code in {"ISCA", "ISCI", "ISC_"} else _INFARCT
    for m in big.finditer(t):
        prefix = t[:m.start()]
        if _NEG.search(prefix[-40:]):
            continue
        lm = _location_match(prefix)
        loc = _LOC[lm.group(1)] if lm else None
        if code in {"MI_UNSPEC", "ISC_"} or (loc and code in _LOC_TO_CODES.get(loc, set())):
            out.append((lm.start() if lm else m.start(), m.end()))   # "inferior infarction", from the location word
    return out


def _mention_span(s: str, code: str) -> tuple[int, int] | None:
    sp = _mention_spans(s, code)
    return sp[0] if sp else None


def _mention_start(s: str, code: str) -> int | None:
    sp = _mention_span(s, code)
    return sp[0] if sp else None


_TAIL = re.compile(r"\w*(?:\s+(?:wall|segments?|leads?|of indeterminate age|age undetermined|age undeterined|"
                   r"possibly recent|probably old|old|recent|acute))*")
_LEAD_IN = re.compile(r"(?:(?:probabl[ey]|possibl[ey]|consistent with|suggestive of|compatible with|signs of)\s+)+$")


def trim_unsupported(s: str, codes: list[str]) -> tuple[str, list[str]]:
    """Cut only the phrases that state unsupported concepts; keep the rest of the sentence."""
    spans = []
    for c in codes:
        found = _mention_spans(s, c)
        if not found:
            return "", [s]                          # cannot locate the phrase: drop the whole sentence
        spans += found
    spans, raw = [], spans
    for a, b in raw:
        b += _TAIL.match(s.lower(), b).end() - b
        mod = _MODIFIER.search(s[:a].lower())
        a = mod.start() if mod else a
        lead = _LEAD_IN.search(s[:a].lower())
        a = lead.start() if lead else a
        spans.append((a, b))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    removed = [s[a:b].strip() for a, b in merged]
    for a, b in reversed(merged):
        s = s[:a] + " " + s[b:]
    joiner = r"(?:,|&|\bor\b|\band\b|\bwith\b)"
    prev = None
    while prev != s:                                    # remove joiners left dangling by the cuts
        prev = s
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"%s(\s*%s)+" % (joiner, joiner), lambda m: m.group(0).split()[-1] if m.group(0).split() else "", s)
        s = re.sub(r"%s\s*(?=[.;]|$)" % joiner, "", s)
        s = re.sub(r"^\s*(?:%s|[.])\s*" % joiner, "", s)
        s = re.sub(r"\s+([.,;])", r"\1", s).strip()
    if not extract_concepts(s) or not re.search(r"[a-z]", s.lower()):
        return "", removed
    if s and s[-1] not in ".;":
        s += "."
    return s, removed


def low_confidence_concepts(concepts: set[str], ev: dict) -> set[str]:
    """Concepts backed only by CNN diagnoses with p < LOW_CONF_P and no measurement support."""
    items = {c["id"]: c for c in ev["findings"] + ev["diagnoses"]}
    idx = evidence_index(ev)
    weak = set()
    for c in concepts:
        ids = [i for e in SUPPORTED_BY.get(c, {c}) for i in idx.get(e, [])]
        if ids and all(i.startswith("C") and items[i].get("probability", 1.0) < LOW_CONF_P and not items[i].get("supported_by")
                       for i in ids):
            weak.add(c)
    return weak


_MODIFIER = re.compile(r"(?<![a-z-])(?:(?:non-?specific|unspecific|nonspecific|minor|slight|mild|marked|old|recent|acute|"
                       r"complete|incomplete|first[- ]degree|left|right|lv|rv)[ ,]+)+$")


def hedge_sentence(s: str, weak: set[str]) -> tuple[str, int, int]:
    """Insert 'possible' before each unhedged low-confidence mention. Returns (text, n_weak, n_inserted)."""
    starts = sorted({st for c in weak if (st := _mention_start(s, c)) is not None}, reverse=True)
    n_ins = 0
    for st in starts:
        if _HEDGE.search(s[max(0, st - 30):st].lower()):
            continue
        mod = _MODIFIER.search(s[:st].lower())       # "non-specific abnormal t" -> before "non-specific"
        if mod:
            st = mod.start()
        s = s[:st] + "possible " + s[st:]
        n_ins += 1
    return s, len(weak), n_ins


def ground_report(text: str, ev: dict) -> dict:
    """Per sentence: concepts, citations, unsupported concepts; the verified report cuts unsupported
    phrases (or the sentence, if nothing supported is left) and hedges low-confidence mentions."""
    idx = evidence_index(ev)
    sents = []
    unsupported_total, concepts_total, weak_total, hedged_total = 0, 0, 0, 0
    for s in split_sentences(text):
        cs = extract_concepts(s)
        cites, unsup = set(), []
        for c in sorted(cs):
            ids = [i for e in SUPPORTED_BY.get(c, {c}) for i in idx.get(e, [])]
            if ids:
                cites.update(ids)
            else:
                unsup.append(c)
        kept, removed = trim_unsupported(s, unsup) if unsup else (s, [])
        kept_cs = extract_concepts(kept) if kept else set()
        if kept and kept_cs - cs:        # trimming must never create a new claim
            kept, removed = "", [s]
            kept_cs = set()
        kept_cites = {i for c in kept_cs for e in SUPPORTED_BY.get(c, {c}) for i in idx.get(e, [])}
        weak = low_confidence_concepts(kept_cs, ev)
        hedged, n_weak, n_ins = hedge_sentence(kept, weak) if kept else ("", 0, 0)
        concepts_total += len(cs)
        unsupported_total += len(unsup)
        weak_total += n_weak
        hedged_total += n_ins
        sents.append({"text": s, "verified_text": hedged, "hedged_text": hedged, "hedged": n_ins > 0,
                      "concepts": sorted(cs), "unsupported": unsup, "removed": removed,
                      "citations": sorted(cites, key=lambda x: (x[0], int(x[1:]))),
                      "verified_citations": sorted(kept_cites, key=lambda x: (x[0], int(x[1:]))),
                      "low_confidence": sorted(weak)})
    cited = " ".join(s["text"] + (f" [{', '.join(s['citations'])}]" if s["citations"] else "") +
                     (" [UNSUPPORTED]" if s["unsupported"] else "") for s in sents)
    verified = " ".join(s["verified_text"] + (f" [{', '.join(s['verified_citations'])}]" if s["verified_citations"] else "")
                        for s in sents if s["verified_text"])
    return {"sentences": sents, "cited_report": cited, "verified_report": verified,
            "n_concepts": concepts_total, "n_unsupported": unsupported_total,
            "n_low_confidence": weak_total, "n_low_confidence_unhedged": hedged_total}
