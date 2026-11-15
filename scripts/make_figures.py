"""Figures for Steps 1-5 -> artifacts/figures/*.png"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from claims.config import load_config
from claims.data import CANONICAL_LEADS, load_metadata, load_signals, split_ids, to_mv
from claims.explain import integrated_gradients
from claims.signal import Standardizer, clean_for_measurement
from claims.train_cnn import DEVICE, load_checkpoint, macro_auc
from claims.validate_measure import load_12sl

# reference palette (dataviz skill, light mode)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def evidence_panel(cfg, meta, sig, ev, model, ck, out):
    eid = ev["ecg_id"]
    row = meta.index.get_loc(eid)
    x = clean_for_measurement(to_mv(sig[row]))
    t = np.arange(len(x)) / 500
    classes, th = ck["classes"], np.asarray(ck["thresholds"])
    std = Standardizer.from_state(ck["standardizer"])
    X = np.load(cfg.path("cache") / "cnn100.npy", mmap_mode="r")
    xin = torch.as_tensor(std(np.asarray(X[row])[None])[0], device=DEVICE)
    dx = [c for c in ev["diagnoses"] if c["concept"] not in ("SR", "NORM")] or ev["diagnoses"]
    main = dx[0]
    A = integrated_gradients(model, xin, [classes.index(main["concept"])], steps=32)[0]
    attr = np.abs(A)
    attr = attr / (attr.max() + 1e-9)

    leads = CANONICAL_LEADS
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(12, 2, width_ratios=[3.2, 1.25], wspace=0.04, hspace=0.12)
    meas_by_lead = {}
    for c in ev["findings"]:
        if c["concept"] in ("STE", "STD", "TINV", "QWAVE"):
            for l in c["leads"]:
                meas_by_lead.setdefault(l, []).append(c)
    for i, l in enumerate(leads):
        ax = fig.add_subplot(gs[i, 0])
        li = CANONICAL_LEADS.index(l)
        a = np.repeat(attr[li], 5)[: len(t)]
        a = np.convolve(a, np.ones(100) / 100, mode="same")
        lo, hi = x[:, li].min(), x[:, li].max()
        pad = 0.1 * (hi - lo + 1e-3)
        ax.imshow(a[None], aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("s", [SURF] + SEQ),
                  extent=[0, 10, lo - pad, hi + pad], vmin=0, vmax=max(0.05, a.max()), alpha=0.85, zorder=0)
        ax.plot(t, x[:, li], color=INK, lw=0.8, zorder=2)
        labs = []
        for c in meas_by_lead.get(l, []):
            ax.axvspan(*c["window_s"], color=ORANGE, alpha=0.35, lw=0, zorder=1)
            labs.append(f'[{c["id"]}] {c["finding"]} {c.get("per_lead", {}).get(l)}')
        if labs:   # one stacked label per lead, so findings on the same beat never overlap
            ax.text(meas_by_lead[l][0]["window_s"][1] + 0.04, hi, "\n".join(labs), fontsize=7.5, color=INK2,
                    va="top", zorder=3, bbox={"facecolor": SURF, "edgecolor": "none", "alpha": 0.8, "pad": 1})
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, 10)
        ax.set_ylabel(l, rotation=0, ha="right", va="center", fontsize=10, color=INK)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.5)
        if i < 11:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)")
    # evidence text panel
    tx = fig.add_subplot(gs[:, 1])
    tx.axis("off")
    lab = ev.get("_labels", "")
    lines = [("ECG %d  |  %s" % (eid, f"{ev['patient']['age']} y {ev['patient']['sex']}"), INK, "bold"),
             (f"Cardiologist labels: {lab}", INK2, "normal"), ("", INK, "normal"),
             ("CNN diagnoses", INK, "bold")]
    for c in ev["diagnoses"]:
        s = f"[{c['id']}] {c['finding']}  p={c['probability']:.2f}"
        if c["supported_by"]:
            s += f"  ✓ {','.join(c['supported_by'])}"
        if c["contradicted_by"]:
            s += f"  ✗ {','.join(c['contradicted_by'])}"
        lines.append((s, INK, "normal"))
    lines += [("", INK, "normal"), ("Measurements", INK, "bold")]
    for c in ev["findings"]:
        v = "" if c["value"] is None else f" {c['value']:.3g} {c['unit']}"
        lines.append((f"[{c['id']}] {c['finding']}{v}", INK, "normal"))
    if ev["flags"]:
        lines += [("", INK, "normal"), ("Flags", INK, "bold")] + [(f, RED, "normal") for f in ev["flags"]]
    y = 0.99
    for s, col, w in lines:
        import textwrap
        for k, part in enumerate(textwrap.wrap(s, 46) or [""]):
            tx.text(0.02, y, ("    " if k else "") + part, color=col, fontweight=w, fontsize=9, va="top",
                    transform=tx.transAxes)
            y -= 0.026
    fig.suptitle(f"Evidence for ECG {eid}: blue shading = where the CNN looked for "
                 f"'{main['finding']}' (Integrated Gradients); orange = measured ST/T/Q findings [M#]",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)


def agreement(cfg, meta, out):
    f = pd.read_parquet(cfg.path("artifacts") / "measurements" / "measurements_flat.parquet").set_index("ecg_id")
    r = load_12sl(cfg.path("ptbxl_plus") / "12sl_features.csv")
    test = [e for e in split_ids(meta, cfg)["test"] if f.at[e, "ok"] and e in r.index]
    specs = [("QRS duration (ms)", "qrs_ms", "QRS_Dur_Global", (40, 200)),
             ("PR interval (ms)", "pr_ms", "PR_Int_Global", (60, 360)),
             ("QT interval (ms)", "qt_ms", "QT_Int_Global", (250, 600)),
             ("ST level at J point, all leads (mV)", "st_j", "ST_Amp", (-0.6, 0.6))]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, (title, k, c, lim) in zip(axs, specs):
        if k == "st_j":
            a = np.concatenate([f.loc[test, f"st_j_{l}"].values for l in CANONICAL_LEADS])
            b = np.concatenate([r.loc[test, f"ST_Amp_{l}"].values for l in CANONICAL_LEADS])
        else:
            a, b = f.loc[test, k].values, r.loc[test, c].values
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        ax.scatter(b, a, s=5, color=BLUE, alpha=0.25, lw=0)
        ax.plot(lim, lim, color=INK2, lw=1, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(title)
        ax.set_xlabel("GE 12SL (reference)")
        ax.set_ylabel("ours")
        med = np.median(np.abs(a - b))
        ax.text(0.03, 0.95, f"median |diff| {med:.3g}\nn = {len(a):,}", transform=ax.transAxes, va="top",
                fontsize=9, color=INK)
    fig.suptitle("Step 3 · Measurement agreement on the held-out test fold (dashed = perfect agreement)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def cnn_perf(cfg, meta, ck, out_auc, out_curve):
    from claims.data import build_labels, cnn_targets, label_vocab, load_scp_statements
    scp = load_scp_statements(cfg)
    y, classes = cnn_targets(build_labels(meta, scp, 0), label_vocab(scp))
    cdir = cfg.path("artifacts") / "cnn"
    ids = np.load(cdir / "ids_val.npy")
    pv = np.load(cdir / "pred_val.npy")
    yv = y[meta.index.get_indexer(ids)]
    mac, per = macro_auc(yv, pv)
    from claims.claims_builder import CLASS_NAMES
    rows = sorted(((CLASS_NAMES[classes[j]], a, int(yv[:, j].sum())) for j, a in per.items()), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10, 10))
    names = [f"{n}  (n={k})" for n, _, k in rows]
    vals = [a for _, a, _ in rows]
    cols = [SEQ[1] if k < 20 else BLUE for _, _, k in rows]
    ax.barh(names, vals, color=cols, height=0.7)
    ax.axvline(mac, color=INK2, ls="--", lw=1)
    ax.text(mac + 0.003, len(rows) - 0.5, f"macro-AUC {mac:.3f}", color=INK, fontsize=9, va="top")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC AUC on validation fold 9")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_title("Step 4 · CNN AUC per class, validation fold\n(light bars: < 20 examples, statistically unreliable)")
    fig.tight_layout()
    fig.savefig(out_auc, dpi=80)
    plt.close(fig)

    h = pd.DataFrame(json.load(open(cdir / "cnn_final.history.json")))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(h.epoch, h.train_loss, color=BLUE, lw=2, label="train")
    axs[0].plot(h.epoch, h.val_loss, color=ORANGE, lw=2, label="validation")
    axs[0].legend(frameon=False)
    axs[0].set_title("Loss (binary cross-entropy)")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylim(0, 0.2)
    axs[1].plot(h.epoch, h.val_macro_auc, color=BLUE, lw=2)
    b = h.val_macro_auc.idxmax()
    axs[1].scatter([h.epoch[b]], [h.val_macro_auc[b]], s=60, color=BLUE, zorder=3, edgecolor=SURF, lw=2)
    axs[1].annotate(f"best: epoch {h.epoch[b]}, {h.val_macro_auc[b]:.3f}\n(model kept)", (h.epoch[b], h.val_macro_auc[b]),
                    xytext=(-130, -45), textcoords="offset points", fontsize=9, arrowprops={"arrowstyle": "-", "color": INK2})
    axs[1].set_ylim(0.85, 0.96)
    axs[1].set_title("Validation macro-AUC")
    axs[1].set_xlabel("epoch")
    fig.suptitle("Step 4 · CNN training (validation loss stops improving after ~epoch 25: later epochs overfit slightly)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_curve, dpi=80)
    plt.close(fig)


def rules(cfg, out):
    rep = json.load(open(cfg.path("artifacts") / "evidence" / "step5_report.json"))
    d = pd.DataFrame(rep["rule_vs_label"])
    nice = {"WIDE_QRS": "Wide QRS → bundle branch block", "LONG_PR": "Long PR → AV block",
            "LVH_V": "LVH voltage → LVH", "TACHY": "HR > 100 → tachycardia", "BRADY": "HR < 60 → bradycardia",
            "LAD": "Left axis dev. → fascicular block", "LOWV": "Low voltage → low voltage",
            "LONG_QT": "Long QTc → long QT", "QWAVE": "Q waves → myocardial infarction"}
    d["name"] = d.rule.map(nice)
    d = d.sort_values("sensitivity")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(d))
    for i, (s, p) in enumerate(zip(d.sensitivity, d.specificity)):
        ax.plot([s, p], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(d.sensitivity, y, s=70, color=ORANGE, label="sensitivity", zorder=2, edgecolor=SURF, lw=2)
    ax.scatter(d.specificity, y, s=70, color=BLUE, label="specificity", zorder=2, edgecolor=SURF, lw=2)
    ax.set_yticks(y, d.name)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    ax.set_xlabel("agreement with cardiologist labels (training folds)")
    ax.set_title("Step 5 · Do rule-based findings match what cardiologists labelled?")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    meta = load_metadata(cfg)
    sig = load_signals(cfg, meta)
    out = cfg.path("artifacts") / "figures"
    out.mkdir(exist_ok=True)
    labels = pd.read_pickle(cfg.path("cache") / "labels.pkl")
    model, ck = load_checkpoint(cfg.path("artifacts") / "cnn" / "cnn_final.pt")
    evs = {}
    for line in open(cfg.path("artifacts") / "evidence" / "evidence_test.jsonl", encoding="utf-8"):
        e = json.loads(line)
        evs[e["ecg_id"]] = e
    picks = [514]
    for want in ("CLBBB", "AFIB"):
        for eid, e in evs.items():
            L = labels.loc[eid]
            if want in set(L.subclass) | set(L.rhythm) and any(c["concept"] == want for c in e["diagnoses"]):
                picks.append(eid)
                break
    for eid in picks:
        e = evs[eid]
        L = labels.loc[eid]
        e["_labels"] = ", ".join(list(L.diagnostic) + list(L.rhythm))
        evidence_panel(cfg, meta, sig, e, model, ck, out / f"evidence_ecg{eid}.png")
        print("wrote evidence panel", eid, e["_labels"])
    agreement(cfg, meta, out / "step3_measurement_agreement.png")
    cnn_perf(cfg, meta, ck, out / "step4_cnn_auc_per_class.png", out / "step4_cnn_training.png")
    rules(cfg, out / "step5_rules_vs_cardiologists.png")
    print("done")


if __name__ == "__main__":
    main()
