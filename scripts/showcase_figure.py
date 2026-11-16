"""Presentation-quality one-page figure for a single CLAIMS case (used by s10_showcase.py).

Layout (16 x 10 in):
  header band    ECG id, diagnosis, patient, provenance chips
  left           standard 3 x 4 + lead II rhythm strip on ECG paper (25 mm/s, 10 mm/mV, square grid)
                 evidence zoom panels: the real signal around each cited time window, window highlighted
  right          measurement tiles (CLAIMS vs GE 12SL), verified CLAIMS report with inline citation chips,
                 cardiologist report, CNN probabilities with per-class thresholds
Only real signal samples and stored outputs are drawn; nothing is synthesised.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from claims.claims_builder import CLASS_NAMES
from claims.data import CANONICAL_LEADS

NAVY, INK, INK2, MUTED, FAINT, LINE = "#0f1b2d", "#0f1b2d", "#3d4b5f", "#6b7a8f", "#94a3b8", "#e3e8ef"
BG, CARD = "#f4f6f9", "#ffffff"
ACCENT, ACCENT_SOFT, TEAL_LIGHT = "#0f766e", "#e6f4f2", "#5eead4"
MEAS, MEAS_SOFT, CNN, CNN_SOFT = "#c2410c", "#fdeee4", "#1d4ed8", "#e8eefc"
AMBER, RED = "#a16207", "#b42318"
PAPER, G_MINOR, G_MAJOR, TRACE = "#fffdfc", "#f8e3e3", "#efbcbe", "#18202e"
FS = 500
GROUPS = [["I", "II", "III"], ["aVR", "aVL", "aVF"], ["V1", "V2", "V3"], ["V4", "V5", "V6"]]
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.unicode_minus": False})


# ----------------------------------------------------------------------------- helpers
def card(fig, rect, fc=CARD, ec=LINE, lw=1.0, r=.006):
    x, y, w, h = rect
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=fig.transFigure,
                                      fc=fc, ec=ec, lw=lw, zorder=-5))


def paper_grid(ax, x0, x1, y0, y1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=PAPER, ec="none", zorder=0))
    for k in range(int(np.floor(x0 / .04)), int(np.ceil(x1 / .04)) + 1):
        t = k * .04
        if x0 - 1e-9 <= t <= x1 + 1e-9:
            ax.plot([t, t], [y0, y1], color=G_MAJOR if k % 5 == 0 else G_MINOR, lw=.75 if k % 5 == 0 else .35, zorder=1)
    for k in range(int(np.ceil(y0 / .1)), int(np.floor(y1 / .1)) + 1):
        v = k * .1
        ax.plot([x0, x1], [v, v], color=G_MAJOR if k % 5 == 0 else G_MINOR, lw=.75 if k % 5 == 0 else .35, zorder=1)


def lead_pill(ax, x, y, txt, size=9.5):
    ax.text(x, y, txt, fontsize=size, fontweight="bold", color=INK2, va="top", ha="left", zorder=6,
            bbox=dict(boxstyle="round,pad=.25,rounding_size=.35", fc="white", ec=LINE, lw=.6, alpha=.95))


def rich_text(fig, ax, x0, y0, maxw, tokens, size=12.5, lh=None, weight="normal"):
    """Draw word tokens and citation chips with wrapping. tokens: list of (kind, text) with kind in
    {'w' word, 'M'/'C' chip, 'h' hedged word, 's' strike}. Coordinates are axes fractions of `ax`."""
    r = fig.canvas.get_renderer()
    axw, axh = ax.bbox.width, ax.bbox.height
    lh = lh or size * 1.75 / 72 * fig.dpi / axh
    x, y = x0, y0
    for kind, txt in tokens:
        if kind in ("M", "C"):
            t = ax.text(x + .004, y, txt, fontsize=size * .72, color="white", fontweight="bold", family="monospace",
                        va="baseline", transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=.28,rounding_size=.3", fc=MEAS if kind == "M" else CNN, ec="none"))
            w = t.get_window_extent(r).width / axw + .022
        else:
            col = {"w": INK, "h": AMBER, "s": RED}[kind]
            t = ax.text(x, y, txt, fontsize=size, color=col, va="baseline", transform=ax.transAxes, fontweight=weight)
            w = t.get_window_extent(r).width / axw
        if x + w > x0 + maxw and x > x0:
            t.remove()
            x, y = x0, y - lh
            if kind in ("M", "C"):
                t = ax.text(x + .004, y, txt, fontsize=size * .72, color="white", fontweight="bold", family="monospace",
                            va="baseline", transform=ax.transAxes,
                            bbox=dict(boxstyle="round,pad=.28,rounding_size=.3", fc=MEAS if kind == "M" else CNN, ec="none"))
            else:
                t = ax.text(x, y, txt, fontsize=size, color={"w": INK, "h": AMBER, "s": RED}[kind], va="baseline",
                            transform=ax.transAxes, fontweight=weight)
        if kind == "h":
            ax.plot([x, x + w - .012], [y - lh * .18] * 2, color=AMBER, lw=1.3, transform=ax.transAxes)
        x += w + (.012 if kind not in ("M", "C") else .006)
    return y - lh


# ----------------------------------------------------------------------------- main figure
def draw_card(case, ev, g, x_mv, ours, ref_row, probs, classes, thresholds, rep_lang, out_png):
    eid = case["ecg_id"]
    fig = plt.figure(figsize=(16, 10), dpi=160, facecolor=BG)
    items = {c["id"]: c for c in ev["findings"] + ev["diagnoses"]}

    # ---------------- header band
    fig.patches.append(Rectangle((0, .905), 1, .095, transform=fig.transFigure, fc=NAVY, ec="none", zorder=-6))
    fig.text(.025, .968, "CLAIMS  ·  EVIDENCE-GROUNDED ECG INTERPRETATION", fontsize=9, color=TEAL_LIGHT,
             fontweight="bold", va="center")
    tid = fig.text(.025, .932, f"ECG {eid}", fontsize=22, color="white", fontweight="bold", va="center", family="monospace")
    fig.canvas.draw()
    x_title = .025 + tid.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width + .012
    fig.text(x_title, .932, case["title"], fontsize=22, color="white", va="center")
    p = ev["patient"]
    chips = [case["category"], f"{p['age']} y · {p['sex'] or 'sex n/a'}", "PTB-XL held-out test fold", "cardiologist-validated labels"]
    xr = .975
    for txt in reversed(chips):
        t = fig.text(xr, .932, txt, fontsize=9.5, color="white", ha="right", va="center",
                     bbox=dict(boxstyle="round,pad=.45,rounding_size=.8", fc="#1f2f47", ec="#36506f", lw=.8))
        fig.canvas.draw()
        bb = t.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
        xr -= bb.width / fig.bbox.width + .007

    # ---------------- 12-lead ECG, 3 x 4 + rhythm strip
    gain = 10 if (np.percentile(x_mv, 99.5, 0) - np.percentile(x_mv, .5, 0)).max() <= 3.0 else 5
    k = gain / 10
    card(fig, (.018, .355, .655, .535))
    fig.text(.03, .872, "12-lead ECG", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(.663, .872, f"25 mm/s  ·  {gain} mm/mV  ·  0.04 s × 0.1 mV small squares", fontsize=8.5, color=MUTED, ha="right")
    ax = fig.add_axes([.028, .368, .635, .495])
    row_mv, top = 3.0, 1.5
    y_lo = -row_mv * 3 - 1.5
    paper_grid(ax, 0, 10, y_lo, top)
    t = np.arange(x_mv.shape[0]) / FS
    base = np.median(x_mv, axis=0)

    def trace(lead, t0, t1, yoff, lw=1.05):
        i = CANONICAL_LEADS.index(lead)
        a, b = int(t0 * FS), min(x_mv.shape[0], int(t1 * FS) + 1)
        y = np.clip((x_mv[a:b, i] - base[i]) * k, -1.45, 1.45) + yoff
        ax.plot(t[a:b], y, color=TRACE, lw=lw, zorder=4, solid_joinstyle="round", solid_capstyle="round")

    for c, group in enumerate(GROUPS):
        for r, lead in enumerate(group):
            yoff = -row_mv * r
            trace(lead, 2.5 * c, 2.5 * (c + 1), yoff)
            lead_pill(ax, 2.5 * c + .05, yoff + 1.38, lead)
            if c > 0:
                ax.plot([2.5 * c] * 2, [yoff - .45, yoff + .45], color=INK2, lw=1.4, zorder=5)
    trace("II", 0, 10, -row_mv * 3)
    lead_pill(ax, .05, -row_mv * 3 + 1.38, "II  rhythm strip")
    ax.set_xlim(0, 10)
    ax.set_ylim(y_lo, top)
    ax.set_aspect(.4, adjustable="box")
    ax.axis("off")

    # ---------------- evidence zoom panels
    cited = []
    for s in g["sentences"]:
        cited += s.get("verified_citations", [])
    support = [m for c in cited if c.startswith("C") for m in items[c]["supported_by"]]
    order = list(dict.fromkeys([c for c in cited if c.startswith("C")] + support + [c for c in cited if c.startswith("M")]))
    zoom, seen = [], set()
    for i in order:                     # one panel per distinct (lead, window); skip recording-wide items
        it = items[i]
        a_, b_ = it["window_s"]
        lead0 = ([l for l in it.get("leads", []) if l not in ("all", "global")] or ["II"])[0]
        key = (lead0, round(a_, 2), round(b_, 2))
        if b_ - a_ > 1.5 or key in seen:
            continue
        seen.add(key)
        zoom.append(i)
    zoom = zoom[:4]
    card(fig, (.018, .03, .655, .305))
    fig.text(.03, .312, "Evidence on the signal", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(.663, .312, "each panel: the real trace around one cited item; shaded = the exact window it refers to",
             fontsize=8.5, color=MUTED, ha="right")
    n = max(1, len(zoom))
    pw, gap = (.635 - (n - 1) * .014) / n, .014
    for j, iid in enumerate(zoom):
        it = items[iid]
        isM = iid.startswith("M")
        leads = [l for l in it.get("leads", []) if l not in ("all", "global")] or ["II"]
        lead = leads[0]
        a, b = it["window_s"]
        zx = .028 + j * (pw + gap)
        zax = fig.add_axes([zx, .045, pw, .2])
        box_w_in, box_h_in = pw * 16, .2 * 10
        span = box_w_in / box_h_in * (2.8 * .4)
        c0 = (a + b) / 2
        t0, t1 = max(0, c0 - span / 2), min(10, c0 + span / 2)
        if t1 - t0 < span:
            t0, t1 = (0, span) if t0 == 0 else (10 - span, 10)
        paper_grid(zax, t0, t1, -1.4, 1.4)
        i = CANONICAL_LEADS.index(lead)
        sl = slice(int(t0 * FS), int(t1 * FS) + 1)
        seg = x_mv[sl, i] - base[i]
        zk = 1.0 if np.ptp(seg) <= 2.6 and np.abs(seg).max() <= 1.38 else .5
        zax.plot(t[sl], np.clip(seg * zk, -1.38, 1.38), color=TRACE, lw=1.4, zorder=4)
        zax.text(t1 - .02, -1.33, f"{int(zk * 10)} mm/mV", fontsize=7.5, color=MUTED, ha="right", va="bottom", zorder=6)
        col, soft = (MEAS, MEAS_SOFT) if isM else (CNN, CNN_SOFT)
        zax.add_patch(Rectangle((a, -1.4), max(b - a, .02), 2.8, fc=col, alpha=.14, ec="none", zorder=2))
        zax.plot([a, a], [-1.4, 1.4], color=col, lw=1.4, zorder=3, ls="-" if isM else (0, (3, 2)))
        zax.plot([b, b], [-1.4, 1.4], color=col, lw=1.4, zorder=3, ls="-" if isM else (0, (3, 2)))
        lead_pill(zax, t0 + .02, 1.33, lead, size=8.5)
        zax.set_xlim(t0, t1)
        zax.set_ylim(-1.4, 1.4)
        zax.set_aspect(.4, adjustable="box")
        zax.axis("off")
        # caption above panel
        if isM:
            val = "" if it["value"] is None else (f"{it['value']:.2f} mV" if it["unit"] == "mV" else f"{it['value']:.0f} {it['unit']}".replace(" deg", "°"))
            cap = f"{it['finding']}  {val}"
            sub = f"{fmt_leads(leads)} · {a:.2f}–{b:.2f} s"
        else:
            cap = f"{CLASS_NAMES.get(it['concept'], it['finding'])}"
            sub = f"p {it['probability']:.2f} (thr {it['threshold']:.2f}) · CNN focus {fmt_leads(leads)}"
        fig.text(zx, .282, iid, fontsize=8.5, color="white", fontweight="bold", family="monospace", va="center",
                 bbox=dict(boxstyle="round,pad=.28,rounding_size=.3", fc=col, ec="none"))
        fig.text(zx + .028, .282, shorten(cap, int(pw * 16 * 11)), fontsize=9.5, color=INK, va="center", fontweight="semibold")
        fig.text(zx, .259, shorten(sub, int(pw * 16 * 14)), fontsize=8.3, color=MUTED, va="center")
    if not zoom:
        fig.text(.345, .15, "No time-localised evidence item is cited by this report.", ha="center", color=MUTED)

    # ---------------- right column
    R = fig.add_axes([.688, .03, .294, .86])
    R.set_xlim(0, 1)
    R.set_ylim(0, 1)
    R.axis("off")

    # measurement tiles
    tiles = [("Heart rate", ours.get("hr_bpm"), "bpm", "HR__Global"), ("PR", ours.get("pr_ms"), "ms", "PR_Int_Global"),
             ("QRS", ours.get("qrs_ms"), "ms", "QRS_Dur_Global"), ("QTc", ours.get("qtc_fridericia_ms"), "ms", None),
             ("Axis", ours.get("qrs_axis_deg"), "°", "R_AxisFrontal_Global")]
    tw, th, tg = (1 - 4 * .02) / 5, .112, .02
    for j, (nm, v, u, col12) in enumerate(tiles):
        x0 = j * (tw + tg)
        R.add_patch(FancyBboxPatch((x0, .875), tw, th, boxstyle="round,pad=0,rounding_size=.012", fc=CARD, ec=LINE, lw=1,
                                   transform=R.transAxes))
        R.text(x0 + .018, .965, nm.upper(), fontsize=7.5, color=MUTED, fontweight="bold", va="center")
        vs = "—" if v is None or pd.isna(v) else f"{v:.0f}"
        tv = R.text(x0 + .018, .928, vs, fontsize=17, color=INK, fontweight="bold", va="center")
        vw = tv.get_window_extent(fig.canvas.get_renderer()).width / R.bbox.width
        R.text(x0 + .018 + vw + .006, .924, u, fontsize=8.5, color=MUTED, va="center")
        if col12:
            b12 = ref_row.get(col12) if ref_row is not None else None
            if col12 == "R_AxisFrontal_Global" and (b12 is None or pd.isna(b12)) and ref_row is not None:
                b12 = ref_row.get("QRS_AxisFront_Global")
            s12 = "n/a" if b12 is None or pd.isna(b12) else f"{b12:.0f}"
            R.text(x0 + .018, .89, f"12SL {s12}", fontsize=7.8, color=MUTED, va="center")
        else:
            R.text(x0 + .018, .89, "Fridericia", fontsize=7.8, color=MUTED, va="center")

    # verified report card
    y_top = .852
    tokens = []
    for s in g["sentences"]:
        if not s["verified_text"]:
            continue
        words = s["verified_text"].split()
        if words:
            words[0] = words[0][0].upper() + words[0][1:]
        for w in words:
            tokens.append(("h" if s["hedged"] and w.lower().strip(".,") == "possible" else "w", w))
        for c in s.get("verified_citations", []):
            tokens.append((c[0], c))
    removed = [r for s in g["sentences"] for r in s["removed"]]
    hedged = any(s["hedged"] for s in g["sentences"])
    # measure height first on a scratch pass
    y_end = rich_text(fig, R, .045, y_top - .075, .91, tokens, size=13.5, weight="semibold")
    extra = (.042 if hedged else 0) + (.032 * (1 + len(" ; ".join(removed)) // 60) if removed else 0)
    lh_r = 13.5 * 1.75 / 72 * fig.dpi / R.bbox.height
    y_end += lh_r * .55
    box_bottom = y_end - extra - .012
    R.add_patch(FancyBboxPatch((0, box_bottom), 1, y_top - box_bottom, boxstyle="round,pad=0,rounding_size=.015",
                               fc=ACCENT_SOFT, ec=ACCENT, lw=1.3, transform=R.transAxes, zorder=-1))
    R.add_patch(Rectangle((0, y_top - .006), 1, .006, fc=ACCENT, ec="none", transform=R.transAxes, zorder=0))
    R.text(.045, y_top - .032, "CLAIMS REPORT  ·  fine-tuned LLM, verified against the evidence", fontsize=8.3,
           color=ACCENT, fontweight="bold", va="center")
    yy = y_end - .012
    if hedged:
        R.text(.045, yy, "●", fontsize=8, color=AMBER, va="center")
        R.text(.075, yy, "“possible”: added by the verifier — CNN p < 0.5 and no supporting measurement",
               fontsize=8.3, color=AMBER, va="center")
        yy -= .032
    if removed:
        R.text(.045, yy, "●", fontsize=8, color=RED, va="center")
        R.text(.075, yy, wrap_to("removed (not in the evidence): " + "; ".join(removed), 66), fontsize=8.3, color=RED,
               va="top" if len(removed) > 1 else "center")
    y = box_bottom - .03

    # cardiologist card
    ref = case_ref_text(case)
    txt = wrap_to(ref[0].upper() + ref[1:], 58)
    nlines = txt.count("\n") + 1
    h = .05 + nlines * .03 + .03
    R.add_patch(FancyBboxPatch((0, y - h), 1, h, boxstyle="round,pad=0,rounding_size=.015", fc=CARD, ec=LINE, lw=1,
                               transform=R.transAxes, zorder=-1))
    R.text(.045, y - .028, "CARDIOLOGIST REPORT  ·  PTB-XL", fontsize=8.3, color=MUTED, fontweight="bold", va="center")
    R.text(.045, y - .055, txt, fontsize=11, color=INK, va="top", linespacing=1.5)
    lang = {"de": "German", "sv": "Swedish"}.get(rep_lang)
    R.text(.045, y - h + .022, f"machine-translated from {lang}" if lang else "original English report", fontsize=8,
           color=MUTED, va="center", style="italic")
    y = y - h - .03

    # CNN probabilities
    top = sorted(range(len(classes)), key=lambda j: -probs[j])[:5]
    hh = .055 + len(top) * .052 + .02
    R.add_patch(FancyBboxPatch((0, y - hh), 1, hh, boxstyle="round,pad=0,rounding_size=.015", fc=CARD, ec=LINE, lw=1,
                               transform=R.transAxes, zorder=-1))
    R.text(.045, y - .028, "CNN PROBABILITIES  ·  top 5", fontsize=8.3, color=MUTED, fontweight="bold", va="center")
    R.text(.955, y - .028, "▏ = class threshold", fontsize=8, color=MUTED, ha="right", va="center")
    yb = y - .07
    for j in top:
        pj, tj = float(probs[j]), float(thresholds[j])
        above = pj >= tj
        R.text(.045, yb + .012, shorten(CLASS_NAMES.get(classes[j], classes[j]), 40), fontsize=9.2,
               color=INK if above else MUTED, va="center", fontweight="semibold" if above else "normal")
        R.text(.955, yb + .012, f"{pj:.2f}", fontsize=9.2, color=INK if above else MUTED, ha="right", va="center")
        R.add_patch(FancyBboxPatch((.045, yb - .016), .91, .011, boxstyle="round,pad=0,rounding_size=.005", fc="#eef1f5",
                                   ec="none", transform=R.transAxes))
        if pj > .004:
            R.add_patch(FancyBboxPatch((.045, yb - .016), .91 * pj, .011, boxstyle="round,pad=0,rounding_size=.005",
                                       fc=CNN if above else "#a8b4c4", ec="none", transform=R.transAxes))
        R.plot([.045 + .91 * tj] * 2, [yb - .022, yb + .001], color=INK, lw=1.6, transform=R.transAxes, zorder=5)
        yb -= .052

    # footer
    for x0, col, ls, label in ((.025, MEAS, "-", "measurement evidence [M#]"),
                               (.15, CNN, (0, (3, 2)), "CNN attribution, Integrated Gradients [C#]")):
        fig.patches.append(Rectangle((x0, .009), .011, .012, transform=fig.transFigure, fc=col, alpha=.2, ec=col, lw=1, ls=ls))
        fig.text(x0 + .016, .0148, label, color=INK2, fontsize=8.5, va="center")
    fig.text(.975, .0148, "Research prototype · not a medical device · ECG data: PTB-XL / PTB-XL+ (PhysioNet, CC BY 4.0)",
             color=MUTED, fontsize=8, ha="right", va="center")
    fig.savefig(out_png, dpi=160, facecolor=BG)
    plt.close(fig)


def fmt_leads(leads):
    return ", ".join(leads[:4]) + (" …" if len(leads) > 4 else "")


def shorten(s, n):
    return s if len(s) <= n else s[: max(1, n - 1)].rstrip() + "…"


def wrap_to(s, w):
    import textwrap
    return "\n".join(textwrap.wrap(s, w))


def case_ref_text(case):
    return case.get("_reference") or ""
