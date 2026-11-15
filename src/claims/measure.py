"""Step 3 - deterministic measurement engine (the "evidence").

Pipeline per 12-lead record (500 Hz, mV, cleaned by ``signal.clean_for_measurement``):

1. R-peak detection on a multi-lead QRS-energy signal (robust to a single flat/noisy lead).
2. Beat alignment + dominant-beat selection (beats that correlate with the median template);
   non-dominant beats are counted as ectopic candidates.
3. 12-lead median beat.
4. Global fiducials on the median beat using the spatial velocity (QRS on/off), a tangent method
   (T end) and a spatial-magnitude search (P on/off).
5. Global intervals (HR, PR, QRS, QT, QTc), frontal QRS axis, RR irregularity, and per-lead
   amplitudes (ST at J and J+60 ms vs. PR-segment baseline, T, R, S, Q, QRS peak-to-peak).
6. Beat-level time windows so every measurement can be cited at seconds in the original record.

Thresholds in ``PARAMS`` were calibrated against the GE 12SL algorithm (PTB-XL+) on training-fold
records only; see scripts/s3_measure.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as ss

from .data import CANONICAL_LEADS

PARAMS = {
    "pre_s": 0.40,          # median-beat window before the R alignment point
    "post_s": 0.65,         # ... and after
    "qrs_on_frac": 0.03,    # spatial-velocity threshold (fraction of QRS peak) for QRS onset
    "qrs_off_frac": 0.06,   # ... for QRS offset
    "gap_ms": 16,           # tolerated dips inside the QRS when growing the QRS region
    "tend_leads_min_amp": 0.10,   # leads used for global T end need |T| >= this (mV)
    "tend_percentile": 75,        # global T end = this percentile of per-lead tangent T ends
    "p_min_amp": 0.045,     # spatial P magnitude (mV) required to call P waves present
    "p_edge_frac": 0.40,    # P on/off where P magnitude falls to this fraction of its peak
    "p_min_consistency": 0.6,   # beat-to-beat P correlation; AF false-P 36%->7%, sinus P kept 98% (train folds)
    "dominant_corr": 0.90,  # beats with lower template correlation are flagged non-dominant
}

FS = 500


@dataclass
class Measurements:
    ok: bool
    reason: str = ""
    fs: int = FS
    r_peaks_s: list = field(default_factory=list)       # all detected beats (s)
    dominant: list = field(default_factory=list)        # bool per beat
    globals: dict = field(default_factory=dict)         # HR, PR, QRS, QT, QTc, axis, RR stats ...
    leads: dict = field(default_factory=dict)           # lead -> {st_j, st_60, t_amp, r_amp, ...}
    fiducials_ms: dict = field(default_factory=dict)    # relative to R alignment point
    beats: list = field(default_factory=list)           # per-beat absolute windows (s)
    per_beat_st: dict = field(default_factory=dict)     # lead -> {st_j: [...], st_60: [...]} per dominant beat
    dominant_r_s: list = field(default_factory=list)    # R times (s) of the beats in per_beat_st

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------- helpers
def _movavg(x: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, x)


def detect_r_peaks(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Multi-lead QRS detector. x: (T, 12) mV."""
    sos = ss.butter(2, [8, 25], btype="bandpass", fs=fs, output="sos")
    b = ss.sosfiltfilt(sos, x, axis=0)
    scale = np.percentile(np.abs(b), 99, axis=0) + 1e-6          # equalise lead contributions
    e = (np.diff(b / scale, axis=0, prepend=b[:1] / scale) ** 2).sum(axis=1)
    e = _movavg(e, int(0.08 * fs))
    thr = 0.25 * np.percentile(e, 99)
    peaks, _ = ss.find_peaks(e, height=thr, distance=int(0.25 * fs))
    # refine: point of max absolute deflection (summed over leads) within +-50 ms
    w = int(0.05 * fs)
    mag = np.abs(x - np.median(x, axis=0)).sum(axis=1)
    out = []
    for p in peaks:
        lo, hi = max(0, p - w), min(len(x), p + w)
        out.append(lo + int(np.argmax(mag[lo:hi])))
    return np.unique(np.asarray(out, dtype=int))


def _align_beats(x, r, pre, post):
    keep = [p for p in r if p - pre >= 0 and p + post <= len(x)]
    if not keep:
        return np.zeros((0, pre + post, x.shape[1])), np.asarray([], dtype=int)
    return np.stack([x[p - pre:p + post] for p in keep]), np.asarray(keep, dtype=int)


def _dominant_mask(beats: np.ndarray, pre: int, fs: int, thr: float) -> np.ndarray:
    """Correlate each beat's QRS-T region with the median template (all leads concatenated)."""
    if len(beats) < 3:
        return np.ones(len(beats), dtype=bool)
    a, b = pre - int(0.08 * fs), pre + int(0.30 * fs)
    seg = beats[:, a:b, :]
    tmpl = np.median(seg, axis=0).ravel()
    tmpl = (tmpl - tmpl.mean()) / (tmpl.std() + 1e-9)
    corr = []
    for s in seg:
        v = s.ravel()
        v = (v - v.mean()) / (v.std() + 1e-9)
        corr.append(float(np.mean(v * tmpl)))
    m = np.asarray(corr) >= thr
    return m if m.sum() >= 2 else np.ones(len(beats), dtype=bool)


def _grow(sv: np.ndarray, p: int, thr: float, gap: int, lo: int, hi: int, step: int) -> int:
    """Grow a region from p in direction ``step`` while sv stays above thr (allowing short dips)."""
    i = p
    while lo < i < hi:
        nxt = sv[i + step:i + step * gap:step] if step > 0 else sv[max(lo, i - gap):i][::-1]
        if len(nxt) == 0 or nxt.max() <= thr:
            break
        i += step
    return i


def _tangent_tend(y: np.ndarray, start: int, stop: int, fs: int) -> tuple[float, float, int]:
    """Tangent-method T end for one baseline-corrected lead. Returns (t_end_idx, t_amp, t_peak_idx)."""
    stop = max(stop, start + 2)
    seg = y[start:stop]
    tp = start + int(np.argmax(np.abs(seg)))
    amp = float(y[tp])
    d = np.gradient(y)
    lim = min(len(y) - 1, tp + int(0.20 * fs))
    if lim <= tp + 2:
        return float("nan"), amp, tp
    desc = d[tp:lim] * (-np.sign(amp))          # slope heading back toward baseline is positive
    k = tp + int(np.argmax(desc))
    slope = d[k]
    if abs(slope) < 1e-6:
        return float("nan"), amp, tp
    te = k - y[k] / slope
    if not (tp <= te <= len(y) - 1):
        return float("nan"), amp, tp
    return float(te), amp, tp


def _qrs_bounds(M: np.ndarray, pre: int, fs: int, P: dict) -> tuple[int, int, np.ndarray]:
    """QRS onset/offset (sample indices in the median beat) from the 12-lead spatial velocity."""
    ms = 1000.0 / fs
    Ms = _movavg(M, max(1, int(0.008 * fs)), axis=0)
    sv = np.sqrt((np.gradient(Ms, axis=0) ** 2).sum(axis=1))
    sv = _movavg(sv, max(1, int(0.010 * fs)))
    w = int(0.06 * fs)
    pk = pre - w + int(np.argmax(sv[pre - w:pre + w]))
    floor = np.percentile(sv, 10)
    span = sv[pk] - floor
    gap = max(2, int(P["gap_ms"] / ms))
    on = _grow(sv, pk, floor + P["qrs_on_frac"] * span, gap, pre - int(0.20 * fs), pk + 1, -1)
    off = _grow(sv, pk, floor + P["qrs_off_frac"] * span, gap, pk - 1, pre + int(0.25 * fs), +1)
    return on, off, sv


def remove_baseline_spline(x: np.ndarray, r: np.ndarray, on_offset: int, fs: int = FS
                           ) -> tuple[np.ndarray, dict]:
    """Classic PR-segment knot method: the PR segment is electrically silent, so its level in each
    beat samples the baseline. A cubic spline through those knots estimates the wander, which is
    subtracted. Returns (corrected signal, {lead: wander range in mV}).
    on_offset: QRS onset relative to the R alignment point (samples, negative)."""
    from scipy.interpolate import CubicSpline

    a, b = on_offset - int(0.030 * fs), on_offset - int(0.010 * fs)
    knots_t, knots_v = [], []
    for p in r:
        lo, hi = p + a, p + b
        if lo >= 0 and hi <= len(x):
            knots_t.append((lo + hi) / 2)
            knots_v.append(x[lo:hi].mean(axis=0))
    if len(knots_t) < 3:
        return x, {}
    kt, kv = np.asarray(knots_t), np.asarray(knots_v)                   # (K,), (K, 12)
    t = np.clip(np.arange(len(x)), kt[0], kt[-1])                       # hold first/last knot at the edges
    base = CubicSpline(kt, kv, axis=0, bc_type="natural")(t)
    wander = {lead: float(np.ptp(kv[:, i])) for i, lead in enumerate(CANONICAL_LEADS)}
    return x - base, wander


# --------------------------------------------------------------------------- main
def measure(x: np.ndarray, fs: int = FS, sex: int | None = None, params: dict | None = None) -> Measurements:
    """x: (T, 12) cleaned mV signal in canonical lead order."""
    P = {**PARAMS, **(params or {})}
    r = detect_r_peaks(x, fs)
    if len(r) < 3:
        return Measurements(ok=False, reason=f"only {len(r)} beats detected", r_peaks_s=(r / fs).tolist())

    rr = np.diff(r) / fs
    pre, post = int(P["pre_s"] * fs), int(P["post_s"] * fs)
    ms = 1000.0 / fs
    beats, kept = _align_beats(x, r, pre, post)
    if len(beats) < 2:
        return Measurements(ok=False, reason="too few complete beats", r_peaks_s=(r / fs).tolist())
    dom = _dominant_mask(beats, pre, fs, P["dominant_corr"])

    # ---- baseline wander removal: spline through each beat's PR segment, then re-segment
    wander = {}
    if P.get("spline_baseline", True):
        on0, _, _ = _qrs_bounds(np.median(beats[dom], axis=0), pre, fs, P)
        x, wander = remove_baseline_spline(x, r, on0 - pre, fs)
        beats, kept = _align_beats(x, r, pre, post)

    M = np.median(beats[dom], axis=0)                                  # (L, 12) median beat
    L = M.shape[0]
    on, off, sv = _qrs_bounds(M, pre, fs, P)

    # ---- isoelectric baseline: PR segment just before QRS onset
    b0, b1 = max(0, on - int(0.030 * fs)), max(1, on - int(0.010 * fs))
    iso = M[b0:b1].mean(axis=0)
    Mb = M - iso                                                         # baseline-corrected median
    Mt = _movavg(Mb, max(1, int(0.020 * fs)), axis=0)                    # smoothed for T/P

    # ---- T end (tangent method per lead, robust global aggregate)
    rr_med = float(np.median(rr))
    t_lo = off + int(0.06 * fs)
    t_hi = min(L - 2, pre + int(min(0.62, max(0.30, 0.66 * rr_med)) * fs))
    tends, tamps, tpeaks = [], [], []
    for li in range(12):
        te, ta, tp = _tangent_tend(Mt[:, li], t_lo, t_hi, fs)
        tamps.append(ta)
        tpeaks.append(tp)
        if abs(ta) >= P["tend_leads_min_amp"] and not math.isnan(te):
            tends.append(te)
    tend = float(np.percentile(tends, P["tend_percentile"])) if tends else float("nan")

    # ---- P wave (spatial magnitude before QRS onset, after previous T)
    p_lo = max(0, on - int(0.30 * fs))
    if not math.isnan(tend):
        p_lo = max(p_lo, int(tend - rr_med * fs) + int(0.02 * fs))
    p_hi = on - int(0.025 * fs)
    p_on = p_off = float("nan")
    p_amp = 0.0
    p_consistency = float("nan")
    if p_hi - p_lo > int(0.04 * fs):
        sel = [CANONICAL_LEADS.index(c) for c in ("I", "II", "aVF", "V1", "V2")]
        seg = Mt[p_lo:p_hi][:, sel]
        seg = seg - seg[-1]                                  # relative to the PR segment
        pm = np.sqrt((seg ** 2).sum(axis=1))
        k = int(np.argmax(pm))
        p_amp = float(pm[k])
        interior = int(0.01 * fs) < k < len(pm) - int(0.005 * fs)
        if p_amp >= P["p_min_amp"] and interior:
            # walk outward until the magnitude falls below thr or starts rising again
            # (a local minimum = start of the TP/PR baseline, whose level may differ from PR)
            thr = P["p_edge_frac"] * p_amp
            a = k
            while a > 0 and pm[a] > thr and pm[a - 1] <= pm[a] + 1e-4:
                a -= 1
            b = k
            while b < len(pm) - 1 and pm[b] > thr and pm[b + 1] <= pm[b] + 1e-4:
                b += 1
            p_on, p_off = float(p_lo + a), float(p_lo + b)
            # beat-to-beat consistency: a true P wave repeats before every QRS, fibrillatory
            # waves do not. Correlate each dominant beat's P window with the median P template.
            w0, w1 = max(0, int(p_on) - int(0.02 * fs)), min(L, int(p_off) + int(0.02 * fs))
            segs = beats[dom][:, w0:w1][:, :, sel]
            segs = segs - segs.mean(axis=1, keepdims=True)
            tmpl = np.median(segs, axis=0).ravel()
            cs = [float(np.corrcoef(s.ravel(), tmpl)[0, 1]) for s in segs if s.std() > 1e-6]
            p_consistency = float(np.median(cs)) if cs else 0.0
            if p_consistency < P["p_min_consistency"]:
                p_on = p_off = float("nan")                  # inconsistent "P" = fibrillatory/noise

    # ---- per-lead amplitudes
    j = off
    j60 = min(L - 1, off + int(0.060 * fs))
    leads = {}
    for li, name in enumerate(CANONICAL_LEADS):
        q = Mb[on:off + 1, li]
        rpos = int(np.argmax(q))
        r_amp = float(max(q.max(), 0.0))
        before = q[:rpos] if rpos > 0 else q[:1]
        after = q[rpos:]
        q_amp = float(min(before.min(), 0.0))
        s_amp = float(min(after.min(), 0.0))
        q_dur = float(np.sum(before < 0) * ms) if q_amp < -0.05 else 0.0
        leads[name] = {
            "st_j": float(Mb[j, li]),
            "st_60": float(Mb[j60, li]),
            "t_amp": float(tamps[li]),
            "r_amp": r_amp,
            "s_amp": s_amp,
            "q_amp": q_amp,
            "q_dur_ms": q_dur,
            "qrs_p2p": float(q.max() - q.min()),
            "qrs_area": float(q.sum() * ms),          # mV*ms
            "baseline_wander_mv": round(wander.get(name, 0.0), 4),   # before correction (quality metric)
        }

    # ---- globals
    qrs_ms = (off - on) * ms
    qt_ms = (tend - on) * ms if not math.isnan(tend) else float("nan")
    rr_mean = float(rr.mean())
    axis = math.degrees(math.atan2(leads["aVF"]["qrs_area"], leads["I"]["qrs_area"]))
    sdiff = np.abs(np.diff(rr)) if len(rr) > 1 else np.array([0.0])
    g = {
        "hr_bpm": 60.0 / rr_mean,
        "rr_mean_ms": rr_mean * 1000,
        "rr_sd_ms": float(rr.std() * 1000),
        "rr_cv": float(rr.std() / rr_mean),
        "rr_rmssd_ms": float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000) if len(rr) > 1 else 0.0,
        "rr_pnn_irregular": float(np.mean(sdiff > 0.08)) if len(rr) > 1 else 0.0,
        "p_present": not math.isnan(p_on),
        "p_amp_mv": p_amp,
        "p_consistency": p_consistency,
        "pr_ms": (on - p_on) * ms if not math.isnan(p_on) else float("nan"),
        "p_dur_ms": (p_off - p_on) * ms if not math.isnan(p_on) else float("nan"),
        "qrs_ms": qrs_ms,
        "qt_ms": qt_ms,
        "qtc_bazett_ms": qt_ms / math.sqrt(rr_mean) if not math.isnan(qt_ms) else float("nan"),
        "qtc_fridericia_ms": qt_ms / rr_mean ** (1 / 3) if not math.isnan(qt_ms) else float("nan"),
        "qrs_axis_deg": axis,
        "n_beats": int(len(r)),
        "n_dominant": int(dom.sum()),
        "n_non_dominant": int((~dom).sum()),
        "sokolow_lyon_mv": abs(leads["V1"]["s_amp"]) + max(leads["V5"]["r_amp"], leads["V6"]["r_amp"]),
        "cornell_mv": leads["aVL"]["r_amp"] + abs(leads["V3"]["s_amp"]),
        "limb_qrs_p2p_max_mv": max(leads[c]["qrs_p2p"] for c in ("I", "II", "III", "aVR", "aVL", "aVF")),
        "precordial_qrs_p2p_max_mv": max(leads[c]["qrs_p2p"] for c in ("V1", "V2", "V3", "V4", "V5", "V6")),
    }

    # ---- beat-level citation windows (absolute seconds in the record)
    fid = {"qrs_on": (on - pre) * ms, "qrs_off": (off - pre) * ms, "j60": (j60 - pre) * ms,
           "t_end": (tend - pre) * ms if not math.isnan(tend) else float("nan"),
           "p_on": (p_on - pre) * ms if not math.isnan(p_on) else float("nan"),
           "p_off": (p_off - pre) * ms if not math.isnan(p_off) else float("nan")}
    dom_full = np.zeros(len(r), dtype=bool)
    idx = {int(p): i for i, p in enumerate(r)}
    for p, d in zip(kept, dom):
        dom_full[idx[int(p)]] = d
    beat_list = []
    for i, p in enumerate(r):
        t = p / fs
        beat_list.append({
            "r_s": round(t, 3),
            "dominant": bool(dom_full[i]) if p in set(kept.tolist()) else None,
            "qrs_s": [round(t + fid["qrs_on"] / 1000, 3), round(t + fid["qrs_off"] / 1000, 3)],
            "st_s": [round(t + fid["qrs_off"] / 1000, 3), round(t + fid["j60"] / 1000, 3)],
            "t_s": [round(t + fid["qrs_off"] / 1000 + 0.06, 3),
                    round(t + fid["t_end"] / 1000, 3) if not math.isnan(fid["t_end"]) else None],
        })
    per_beat = {}
    for li, name in enumerate(CANONICAL_LEADS):
        base = beats[dom][:, b0:b1, li].mean(axis=1)
        per_beat[name] = {"st_j": [round(float(v), 4) for v in beats[dom][:, j, li] - base],
                          "st_60": [round(float(v), 4) for v in beats[dom][:, j60, li] - base]}
    dom_r_s = [round(float(p) / fs, 3) for p, d in zip(kept, dom) if d]

    return Measurements(ok=True, r_peaks_s=(r / fs).round(3).tolist(), dominant=dom_full.tolist(),
                        globals=g, leads=leads, fiducials_ms=fid, beats=beat_list, per_beat_st=per_beat,
                        dominant_r_s=dom_r_s)
