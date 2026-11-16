"""Clean ECG images for slides: a 12-lead (3 x 4 + rhythm) and single-lead evidence close-ups.

Real signal only (measurement-path filtered, 500 Hz). Output: showcase/slide_assets/*.png
  python scripts/make_slide_assets.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from showcase_figure import CNN, GROUPS, MEAS, TRACE, lead_pill, paper_grid  # noqa: E402

from claims.config import load_config
from claims.data import CANONICAL_LEADS, load_metadata, load_signals, to_mv
from claims.signal import clean_for_measurement

FS = 500
# (ecg_id, evidence id) close-ups used on the slides
ZOOMS = [(20120, "C1"), (20120, "M4"), (8622, "C2"), (2569, "C2")]
TWELVE = [20120]


def signal(cfg, meta, sig, eid):
    s = cfg.signal
    return clean_for_measurement(to_mv(sig[meta.index.get_loc(eid)]), 500, s.highpass_hz, s.lowpass_hz, s.notch_hz)


def twelve_lead(x, out):
    gain = 10 if (np.percentile(x, 99.5, 0) - np.percentile(x, .5, 0)).max() <= 3.0 else 5
    k = gain / 10
    fig = plt.figure(figsize=(10.4, 5.0), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    row, top = 3.0, 1.5
    lo = -row * 3 - 1.5
    paper_grid(ax, 0, 10, lo, top)
    t = np.arange(x.shape[0]) / FS
    base = np.median(x, 0)

    def tr(lead, t0, t1, yoff):
        i = CANONICAL_LEADS.index(lead)
        a, b = int(t0 * FS), min(x.shape[0], int(t1 * FS) + 1)
        ax.plot(t[a:b], np.clip((x[a:b, i] - base[i]) * k, -1.45, 1.45) + yoff, color=TRACE, lw=1.1, zorder=4)

    for c, g in enumerate(GROUPS):
        for r, lead in enumerate(g):
            tr(lead, 2.5 * c, 2.5 * (c + 1), -row * r)
            lead_pill(ax, 2.5 * c + .05, -row * r + 1.38, lead, size=10)
            if c:
                ax.plot([2.5 * c] * 2, [-row * r - .45, -row * r + .45], color="#3d4b5f", lw=1.5, zorder=5)
    tr("II", 0, 10, -row * 3)
    lead_pill(ax, .05, -row * 3 + 1.38, "II  rhythm", size=10)
    ax.text(9.95, lo + .12, f"25 mm/s · {gain} mm/mV", fontsize=9, color="#6b7a8f", ha="right", va="bottom", zorder=6,
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec="none", alpha=.85))
    ax.set_xlim(0, 10)
    ax.set_ylim(lo, top)
    ax.axis("off")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return gain


def close_up(x, it, out, span=1.6):
    isM = it["id"].startswith("M")
    col = MEAS if isM else CNN
    lead = ([l for l in it.get("leads", []) if l not in ("all", "global")] or ["II"])[0]
    a, b = it["window_s"]
    c0 = (a + b) / 2
    t0 = min(max(0, c0 - span / 2), 10 - span)
    t1 = t0 + span
    i = CANONICAL_LEADS.index(lead)
    t = np.arange(x.shape[0]) / FS
    sl = slice(int(t0 * FS), int(t1 * FS) + 1)
    seg = x[sl, i] - np.median(x[:, i])
    zk = 1.0 if np.abs(seg).max() <= 1.38 else .5
    fig = plt.figure(figsize=(span / 1.12 * 2.2, 2.2), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    paper_grid(ax, t0, t1, -1.4, 1.4)
    ax.add_patch(Rectangle((a, -1.4), max(b - a, .02), 2.8, fc=col, alpha=.15, ec="none", zorder=2))
    for v in (a, b):
        ax.plot([v, v], [-1.4, 1.4], color=col, lw=1.8, zorder=3, ls="-" if isM else (0, (3, 2)))
    ax.plot(t[sl], np.clip(seg * zk, -1.38, 1.38), color=TRACE, lw=1.7, zorder=4)
    lead_pill(ax, t0 + .03, 1.33, lead, size=10)
    ax.text(t1 - .03, -1.34, f"{int(zk * 10)} mm/mV", fontsize=8, color="#6b7a8f", ha="right", va="bottom", zorder=6)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return lead


def main():
    cfg = load_config()
    meta = load_metadata(cfg)
    sig = load_signals(cfg, meta)
    evs = {}
    for line in open(cfg.path("artifacts") / "evidence" / "evidence_test.jsonl", encoding="utf-8"):
        e = json.loads(line)
        evs[e["ecg_id"]] = e
    out = cfg.path("artifacts").parent / "showcase" / "slide_assets"
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for eid in TWELVE:
        info[f"ecg{eid}_12lead"] = {"gain": twelve_lead(signal(cfg, meta, sig, eid), out / f"ecg{eid}_12lead.png")}
    for eid, iid in ZOOMS:
        it = next(c for c in evs[eid]["findings"] + evs[eid]["diagnoses"] if c["id"] == iid)
        lead = close_up(signal(cfg, meta, sig, eid), it, out / f"ecg{eid}_{iid}.png")
        info[f"ecg{eid}_{iid}"] = {"lead": lead, "window_s": it["window_s"], "finding": it["finding"]}
    json.dump(info, open(out / "assets.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
