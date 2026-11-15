"""Unit tests for the deterministic parts of the pipeline (no dataset or GPU needed)."""
import numpy as np
import pandas as pd
import pytest

from claims.claims_builder import build_evidence, cnn_claims, evidence_to_text, measurement_claims
from claims.data import CANONICAL_LEADS, _file_lead_names, build_labels, cnn_targets, label_vocab
from claims.grounding import extract_concepts, ground_report
from claims.measure import measure
from claims.signal import clean_for_measurement

FS = 500


def synth_ecg(hr=60, qrs_ms=90, st_mv=0.0, st_leads=("V2", "V3"), pr_ms=160, t_amp=0.3, seed=0):
    """Piecewise synthetic 12-lead ECG with known fiducials."""
    rng = np.random.default_rng(seed)
    T = 10 * FS
    t = np.arange(T) / FS
    x = np.zeros((T, 12))
    rr = 60 / hr
    for r in np.arange(0.6, 9.6, rr):
        for li, lead in enumerate(CANONICAL_LEADS):
            sign = -1 if lead == "aVR" else 1
            amp = 1.0 if lead not in ("V1", "V2") else 0.6
            # P wave (gaussian), QRS (triangle-ish gaussian of the right width), T wave
            x[:, li] += sign * 0.12 * np.exp(-0.5 * ((t - (r - pr_ms / 1000 + 0.05)) / 0.02) ** 2)
            sd = qrs_ms / 1000 / 6
            x[:, li] += sign * amp * np.exp(-0.5 * ((t - r) / sd) ** 2)
            if lead in st_leads:  # ST plateau from J point to T
                j = r + qrs_ms / 2000
                x[:, li] += st_mv * ((t > j) & (t < j + 0.20)) * 1.0
            x[:, li] += sign * t_amp * np.exp(-0.5 * ((t - (r + 0.30)) / 0.05) ** 2)
    return x + rng.normal(0, 0.005, x.shape)


def _m(x):
    return measure(clean_for_measurement(x, FS), FS)


def test_lead_name_mapping_handles_uppercase_augmented_leads():
    names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    assert _file_lead_names(names)[3:6] == ["AVR", "AVL", "AVF"]


def test_labels_keep_rhythm_codes_with_zero_likelihood():
    scp = pd.DataFrame({"diagnostic": [1, 0, 1], "form": [0, 0, 0], "rhythm": [0, 1, 0],
                        "diagnostic_class": ["NORM", None, "MI"], "diagnostic_subclass": ["NORM", None, "IMI"]},
                       index=["NORM", "SR", "IMI"])
    meta = pd.DataFrame({"scp_codes": [{"NORM": 100.0, "SR": 0.0}, {"IMI": 15.0, "SR": 0.0}]}, index=[1, 2])
    lab = build_labels(meta, scp, 0)
    assert lab.loc[1, "rhythm"] == ["SR"] and lab.loc[1, "subclass"] == ["NORM"]
    assert lab.loc[2, "superclass"] == ["MI"]
    assert build_labels(meta, scp, 50).loc[2, "subclass"] == []
    y, classes = cnn_targets(lab, label_vocab(scp))
    assert classes == ["IMI", "NORM", "SR"] and y.tolist() == [[0, 1, 1], [1, 0, 1]]


@pytest.mark.parametrize("hr", [50, 75, 110])
def test_heart_rate(hr):
    m = _m(synth_ecg(hr=hr))
    assert m.ok and abs(m.globals["hr_bpm"] - hr) < 2


def test_qrs_width_ordering_and_wide_qrs_claim():
    narrow = _m(synth_ecg(qrs_ms=80)).globals["qrs_ms"]
    wide = _m(synth_ecg(qrs_ms=160)).globals["qrs_ms"]
    assert wide > narrow + 30


def test_st_elevation_detected_only_in_affected_leads():
    m = _m(synth_ecg(st_mv=0.3, st_leads=("V2", "V3", "V4")))
    st = {k: v["st_j"] for k, v in m.leads.items()}
    assert st["V3"] > 0.15 and abs(st["V6"]) < 0.08
    claims = measurement_claims(m.to_dict(), sex=0, age=60)
    ste = [c for c in claims if c["concept"] == "STE"]
    assert ste and "V3" in ste[0]["leads"] and "V6" not in ste[0]["leads"]
    assert "anterior" in ste[0]["contiguous_territories"]
    # every claim must carry a valid time window inside the record
    for c in claims:
        assert 0 <= c["window_s"][0] <= c["window_s"][1] <= 10


def test_cnn_claim_support_and_contradiction():
    m = _m(synth_ecg(qrs_ms=170)).to_dict()
    mc = measurement_claims(m, 0, 60)
    classes = ["CLBBB", "NORM"]
    cc = cnn_claims(np.array([0.9, 0.8]), classes, np.array([0.5, 0.5]), mc)
    by = {c["concept"]: c for c in cc}
    assert by["CLBBB"]["supported_by"], "wide QRS should support LBBB"
    assert by["NORM"]["contradicted_by"], "wide QRS should contradict NORM"
    ev = build_evidence(1, {"sex": 0, "age": 60}, m, np.array([0.9, 0.8]), classes, np.array([0.5, 0.5]))
    assert any("conflicts" in f for f in ev["flags"])
    assert "[C1]" in evidence_to_text(ev)


def test_st_changes_with_wide_qrs_are_secondary_and_do_not_support_mi():
    m = _m(synth_ecg(qrs_ms=170, st_mv=0.3, st_leads=("II", "III", "aVF"))).to_dict()
    mc = measurement_claims(m, 0, 60)
    ste = [c for c in mc if c["concept"] == "STE"]
    assert ste and ste[0].get("secondary_to_wide_qrs")
    cc = cnn_claims(np.array([0.9]), ["IMI"], np.array([0.5]), mc)
    assert cc[0]["supported_by"] == [], "conduction-related ST elevation must not support infarction"


@pytest.mark.parametrize("text,expected,absent", [
    ("sinus rhythm. left bundle branch block.", {"SR", "CLBBB"}, {"ILBBB"}),
    ("incomplete right bundle branch block", {"IRBBB"}, {"CRBBB"}),
    ("sinus rhythm left axis t abnormal, probably inferior myocardial injury", {"SR", "NST_", "ISCI"}, set()),
    ("atrial fibrillation. q waves in ii, iii, avf consistent with old inferior myocardial infarction.",
     {"AFIB", "QWAVE", "IMI"}, set()),
    ("no st elevation. normal ecg.", {"NORM"}, {"STE"}),
    ("sinus tachycardia. left ventricular hypertrophy.", {"STACH", "LVH"}, {"SR", "TACHY"}),
])
def test_concept_extractor(text, expected, absent):
    c = extract_concepts(text)
    assert expected <= c, c
    assert not (absent & c), c


def test_grounding_flags_unsupported_sentences():
    ev = {"findings": [{"id": "M1", "concept": "HR"}],
          "diagnoses": [{"id": "C1", "concept": "SR"}, {"id": "C2", "concept": "CRBBB"}]}
    g = ground_report("Sinus rhythm. Right bundle branch block. Atrial fibrillation.", ev)
    assert g["sentences"][0]["citations"] == ["C1"]
    assert g["sentences"][2]["unsupported"] == ["AFIB"]
    assert "Atrial fibrillation" not in g["verified_report"]
    assert g["n_unsupported"] == 1


def test_llm_example_masks_prompt_tokens():
    pytest.importorskip("transformers")
    from claims.llm import encode_example

    class Tok:
        eos_token = "</s>"

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            return "PROMPT " + msgs[-1]["content"]

        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [ord(ch) for ch in s]}

    e = encode_example(Tok(), "ev", "report", 1000)
    n_prompt = len("PROMPT Evidence:\nev\n\nWrite the ECG report.")
    assert e["labels"][:n_prompt] == [-100] * n_prompt
    assert e["labels"][n_prompt:] == e["input_ids"][n_prompt:]


def test_low_probability_diagnosis_is_hedged_in_verified_report():
    ev = {"findings": [{"id": "M1", "concept": "HR"}],
          "diagnoses": [{"id": "C1", "concept": "SR", "probability": 0.99, "supported_by": []},
                        {"id": "C2", "concept": "LAO/LAE", "probability": 0.15, "supported_by": []},
                        {"id": "C3", "concept": "IMI", "probability": 0.30, "supported_by": []}]}
    g = ground_report("sinus rhythm. left atrial abnormality. inferior infarction.", ev)
    v = g["verified_report"]
    assert "possible left atrial abnormality" in v and "possible inferior infarction" in v
    assert "possible sinus" not in v
    assert g["n_low_confidence"] == 2 and g["n_low_confidence_unhedged"] == 2
    # already hedged text is left alone
    g2 = ground_report("probably inferior infarction.", ev)
    assert g2["n_low_confidence_unhedged"] == 0 and "possible" not in g2["verified_report"]


def test_verifier_cuts_only_the_unsupported_phrase():
    ev = {"findings": [{"id": "M1", "concept": "HR"}],
          "diagnoses": [{"id": "C1", "concept": "SR", "probability": 0.96, "supported_by": []},
                        {"id": "C2", "concept": "IRBBB", "probability": 0.6, "supported_by": []}]}
    g = ground_report("sinus rhythm, anterior infarction, incomplete right bundle branch block.", ev)
    assert g["verified_report"] == "sinus rhythm, incomplete right bundle branch block. [C1, C2]"
    assert g["sentences"][0]["removed"] == ["anterior infarction"]
    # a sentence with nothing supported left is dropped entirely
    assert ground_report("atrial fibrillation.", ev)["verified_report"] == ""
    # location words are matched as whole words ("anterolateral" is not "lateral")
    assert extract_concepts("anterolateral infarction") == {"AMI", "LMI"}
