"""Step 6a - turn PTB-XL free-text reports (German / English / Swedish) into English targets.

Strategy: normalise -> detect language -> domain glossary (ECG terminology, longest match first)
-> Marian MT for German residue. Swedish is glossary-only (opus-mt-sv-en mistranslates ECG jargon).
"""
from __future__ import annotations

import re

SV_MARK = re.compile(r"sinusrytm|förmaks|vänster|höger|kammar|avvikande|el-axel|normalt ekg|skänkelblock|ischemi\b")
DE_MARK = re.compile(r"sinusrhythmus|normales ekg|typ\b|infarkt|vorhof|unbest|wahrscheinlich|kammer|sinustachy|"
                     r"sinusbrady|schenkelblock|ableitung|extrasystole\(n\)|lagetyp|hypertrophie|abnorm\b|bericht|"
                     r"linksbelastung|niederspannung|ischämie|verdacht|möglich|leitungsst")

GLOSSARY_DE = {
    "unbestätigter bericht": "",
    "überdrehter linkstyp": "left axis deviation", "ueberdrehter linkstyp": "left axis deviation",
    "überdrehter rechtstyp": "right axis deviation", "ueberdrehter rechtstyp": "right axis deviation",
    "lagetyp normal": "normal axis", "linkstyp": "left axis", "rechtstyp": "right axis",
    "steiltyp": "vertical axis", "indifferenztyp": "normal axis", "sagittaltyp": "sagittal axis",
    "p-sinistrocardiale": "left atrial abnormality", "p-dextrocardiale": "right atrial abnormality",
    "p-verbreiterung": "widened p wave",
    "linksbelastung": "left ventricular strain", "rechtsbelastung": "right ventricular strain",
    "linkshypertrophie": "left ventricular hypertrophy", "rechtshypertrophie": "right ventricular hypertrophy",
    "periphere niederspannung": "low voltage in limb leads",
    "unvollständiger rechtsschenkelblock": "incomplete right bundle branch block",
    "unvollständiger linksschenkelblock": "incomplete left bundle branch block",
    "rechtsschenkelblock": "right bundle branch block", "linksschenkelblock": "left bundle branch block",
    "linksanteriorer hemiblock": "left anterior fascicular block",
    "linksposteriorer hemiblock": "left posterior fascicular block",
    "vorhofflimmern/-flattern": "atrial fibrillation/flutter", "vorhofflimmern": "atrial fibrillation",
    "vorhofflattern": "atrial flutter",
    "sinusrhythmus": "sinus rhythm", "sinustachykardie": "sinus tachycardia", "sinusbradykardie": "sinus bradycardia",
    "sinusarrhythmie": "sinus arrhythmia", "sinus arrhythmie": "sinus arrhythmia",
    "ventrikuläre extrasystole(n)": "ventricular extrasystole(s)",
    "supraventrikuläre extrasystole(n)": "supraventricular extrasystole(s)",
    "schrittmacher": "pacemaker", "qt-verlängerung": "qt prolongation",
    "st-senkung": "st depression", "st-hebung": "st elevation", "myokardschaden": "myocardial injury",
    "intraventrikuläre leitungsstörung": "intraventricular conduction disturbance",
    "bifaszikulärer block": "bifascicular block", "alter unbest.": "age undetermined",
    "sonst normales ekg": "otherwise normal ecg", "normales ekg": "normal ecg",
    "nicht auszuschliessen": "cannot be excluded", "verdacht auf": "suspected", "grenzbefund": "borderline finding",
}

GLOSSARY_SV = {
    "extrem vänster el-axel": "extreme left axis deviation", "vänster el-axel": "left axis deviation",
    "höger el-axel": "right axis deviation", "horisontalställd el-axel": "horizontal axis",
    "avvikande qrs(t) förlopp": "abnormal qrs(t) morphology", "normalt ekg": "normal ecg",
    "sinusrytm": "sinus rhythm", "sinusbradykardi": "sinus bradycardia", "sinustachykardi": "sinus tachycardia",
    "sinusarytmi": "sinus arrhythmia", "sinus arytmi": "sinus arrhythmia",
    "förmaksflimmer/-fladder": "atrial fibrillation/flutter", "förmaksflimmer": "atrial fibrillation",
    "vänster anteriort hemiblock": "left anterior fascicular block",
    "vänster posteriort hemiblock": "left posterior fascicular block",
    "inkomplett högersidigt skänkelblock": "incomplete right bundle branch block",
    "inkomplett vänstersidigt skänkelblock": "incomplete left bundle branch block",
    "högersidigt skänkelblock": "right bundle branch block", "vänstersidigt skänkelblock": "left bundle branch block",
    "vänster kammarhypertrofi": "left ventricular hypertrophy", "kammarkhypertrofi": "ventricular hypertrophy",
    "kammarhypertrofi": "ventricular hypertrophy", "vänster kammarbelastning": "left ventricular strain",
    "kammarbelastning": "ventricular strain", "vänster belastning": "left ventricular strain",
    "låga qrs amplituder i extremitetsavledningarna": "low qrs voltage in limb leads",
    "intraventrikulärt retledningshinder": "intraventricular conduction delay",
    "bifascikulärt block": "bifascicular block", "st-t sänkning": "st-t depression", "st-sänkning": "st depression",
    "t-förändring": "t-wave change", "st-förändring": "st change", "myokardaffektion": "myocardial involvement",
    "myokardskada": "myocardial injury", "subendokardiell": "subendocardial", "flacka t-vågor": "flat t waves",
    "digitalispåverkan": "digitalis effect", "ålder ej bestämmbar": "age undetermined",
    "ålder svårbestämmbar": "age undetermined", "kan ej uteslutas": "cannot be excluded",
    "bör övervägas": "should be considered", "överväg": "consider", "inger misstanke om": "raises suspicion of",
    "misstänkt": "suspected", "supraventrikulär(a)": "supraventricular", "supraventrikulära": "supraventricular",
    "ventrikulär(a)": "ventricular", "ventrikulära": "ventricular", "extrasystoli(er)": "extrasystole(s)",
    "extraslag": "extra beats", "kammarfrekvens": "ventricular rate", "regelbunden": "regular",
    "p-våg": "p wave", "q-våg": "q wave", "t-vågor": "t waves", "amplitudkriterier": "voltage criteria",
    "bröstavledningarnas r-s transition förskjuten åt vänster": "r/s transition in chest leads shifted to the left",
    "förlängd": "prolonged", "ospecifikt": "nonspecific", "ospecifik": "nonspecific", "inkomplett": "incomplete",
    "patologiskt": "pathological", "patologisk": "pathological", "sannolikt": "probably", "möjligen": "possibly",
    "äldre": "older", "gammal": "old", "färsk": "recent", "eljest": "otherwise", "infarkt": "infarction",
    "ischemi": "ischemia", "belastning": "strain", "sänkning": "depression", "hypertrofi": "hypertrophy",
    "hemiblock": "fascicular block", "skänkelblock": "bundle branch block", "främre": "anterior",
    "anteriort": "anterior", "anterolateralt": "anterolateral", "inferolateralt": "inferolateral",
    "lateralt": "lateral", "hög lateral": "high lateral", "måttliga": "moderate", "lätta": "mild", "lätt": "mild",
    "såsom vid": "as in", "orsakad av": "caused by", "eller": "or", "och": "and", "med": "with", "ej": "not",
    "inga": "no", "inget": "no", "funna": "found", "säkert": "definite", "förändringar": "changes",
    "förändring": "change", "vid": "at", "på": "on", "för": "for", "av": "of", "kan": "may", "saknas": "absent",
}


POST_FIXES = [  # residual MT / device artefacts in the English text
    (r"\b\d\.\d\d\b", ""),                                   # device software version, e.g. "4.46"
    (r"\bekg\b", "ecg"), (r"\bderivations?\b", "leads"), (r"\bdischarges?\b", "leads"),
    (r"left position type", "left axis"), (r"right position type", "right axis"),
    (r"steep position type|steep type", "vertical axis"), (r"\bdesz\.", ""), (r"\bneg\.", "negative"),
    (r"\s+([.,;])", r"\1"), (r"([.,;]){2,}", r"\1"),
]
NO_INTERPRETATION = re.compile(r"^(trace only( requested)?\.?|.{0,3})$")
COMPARISON = re.compile(r"compared (with|to)|previous (tracing|ecg)|tracing of|since the last|unchanged from")


def clean_target(s: str) -> str | None:
    """Final English training/evaluation target, or None if the report carries no interpretation."""
    s = normalise(s)
    for pat, rep in POST_FIXES:
        s = re.sub(pat, rep, s)
    # drop sentences that refer to earlier recordings: the system cannot see them, so cannot ground them
    parts = [p for p in re.split(r"(?<=\.)\s+", s) if not COMPARISON.search(p)]
    s = re.sub(r"\s+", " ", " ".join(parts)).strip(" ,;")
    if NO_INTERPRETATION.match(s):
        return None
    return s if s.endswith(".") else s + "."


def normalise(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_language(s: str) -> str:
    if SV_MARK.search(s):
        return "sv"
    if DE_MARK.search(s):
        return "de"
    return "en"


def apply_glossary(s: str, glossary: dict[str, str]) -> str:
    for k in sorted(glossary, key=len, reverse=True):
        s = re.sub(r"(?<![a-zäöüåé])" + re.escape(k) + r"(?![a-zäöüåé])", glossary[k], s)
    return re.sub(r"\s+", " ", s).strip(" .,")


def translate_reports(texts: list[str], langs: list[str], device: str = "cuda", bs: int = 64) -> list[str]:
    from transformers import MarianMTModel, MarianTokenizer
    out = list(texts)
    pre = [apply_glossary(t, GLOSSARY_DE) if l == "de" else apply_glossary(t, GLOSSARY_SV) if l == "sv" else t
           for t, l in zip(texts, langs)]
    idx = [i for i, l in enumerate(langs) if l == "de"]
    name = "Helsinki-NLP/opus-mt-de-en"
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name).to(device).eval()
    import torch
    for i in range(0, len(idx), bs):
        chunk = [pre[j] for j in idx[i:i + bs]]
        with torch.no_grad():
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            gen = model.generate(**enc, max_new_tokens=200, num_beams=4)
        for j, t in zip(idx[i:i + bs], tok.batch_decode(gen, skip_special_tokens=True)):
            out[j] = t
    for i, l in enumerate(langs):
        if l == "sv":
            out[i] = pre[i]
    return [normalise(t) for t in out]
