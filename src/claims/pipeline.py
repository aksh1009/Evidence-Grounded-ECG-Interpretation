"""End-to-end analysis of ONE 12-lead ECG: raw signal -> measurements -> CNN -> evidence -> report -> grounding.

This is the same chain the batch scripts (s3-s8) run, packaged for a single record. It is used by the
CLI (``claims analyze``) and the demo page builder.
"""
from __future__ import annotations

import numpy as np
import torch

from .claims_builder import build_evidence, evidence_to_text
from .explain import integrated_gradients, summarize_attribution
from .grounding import ground_report
from .llm import template_report
from .measure import measure
from .signal import Standardizer, clean_for_measurement, prepare_batch_for_cnn
from .train_cnn import DEVICE, load_checkpoint, predict


class Analyzer:
    """Loads the trained CNN once (and the LLM lazily) and analyses records one at a time."""

    def __init__(self, cfg, use_llm: bool = True):
        self.cfg = cfg
        self.model, ck = load_checkpoint(cfg.path("artifacts") / "cnn" / "cnn_final.pt")
        self.classes, self.th = ck["classes"], np.asarray(ck["thresholds"])
        self.std = Standardizer.from_state(ck["standardizer"])
        self.use_llm = use_llm
        self._llm = None

    def llm(self):
        if self._llm is None:
            from .llm import load_llm
            adapter = self.cfg.path("artifacts") / "llm" / "lora"
            self._llm = load_llm(self.cfg.llm.base_model, str(adapter),
                                 device="cuda" if torch.cuda.is_available() else "cpu")
        return self._llm

    def analyze(self, x_uv: np.ndarray, ecg_id: int = 0, age: float | None = None, sex: int | None = None,
                attributions: bool = True) -> dict:
        """x_uv: (5000, 12) int16 microvolts at 500 Hz, leads in CANONICAL_LEADS order."""
        s = self.cfg.signal
        # 1) measurements on the 500 Hz signal
        x_mv = clean_for_measurement(np.asarray(x_uv, dtype=np.float32) / 1000.0, 500,
                                     s.highpass_hz, s.lowpass_hz, s.notch_hz)
        try:
            m = measure(x_mv, 500, sex=sex).to_dict()
        except Exception as e:  # keep going: the CNN can still run
            m = {"ok": False, "reason": f"error: {e}"}
        # 2) CNN on the 100 Hz copy
        xin = torch.as_tensor(self.std(prepare_batch_for_cnn(np.asarray(x_uv)[None])), device=DEVICE)
        p = predict(self.model, xin)[0]
        # 3) where the CNN looked (Integrated Gradients) for every class above threshold
        attr = None
        pos = [j for j in range(len(self.classes)) if p[j] >= self.th[j]]
        if attributions and pos:
            A = integrated_gradients(self.model, xin[0], pos, steps=24)
            beats = m.get("beats") if m.get("ok") else None
            attr = {self.classes[j]: summarize_attribution(A[i], 100, beats) for i, j in enumerate(pos)}
        # 4) evidence sheet
        ev = build_evidence(ecg_id, {"sex": sex, "age": age}, m, p, self.classes, self.th, attr)
        ev_text = evidence_to_text(ev)
        # 5) report + 6) grounding
        out = {"ecg_id": ecg_id, "measurements": m, "probabilities": dict(zip(self.classes, p.round(4).tolist())),
               "evidence": ev, "evidence_text": ev_text, "template_report": template_report(ev)}
        if self.use_llm:
            from .llm import generate
            tok, model = self.llm()
            out["llm_report"] = generate(tok, model, [ev_text], bs=1)[0]
            out["grounding"] = ground_report(out["llm_report"], ev)
        return out
