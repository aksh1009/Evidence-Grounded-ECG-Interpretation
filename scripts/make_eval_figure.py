"""Step 8 figure: report systems compared on the test fold (reads artifacts/eval/metrics.json)."""
import json
import matplotlib.pyplot as plt
from claims.config import load_config

cfg = load_config()
R = json.load(open(cfg.path("artifacts") / "eval" / "metrics.json"))["reports"]
names = ["reference", "template", "zero_shot", "finetuned"]
labels = ["Cardiologist", "Template", "Zero-shot LLM", "Fine-tuned LLM"]
panels = [("Clinical micro-F1 (higher = better)", [R[n]["clinical_micro_f1"] for n in names], "{:.2f}"),
          ("Clinical macro-F1 (higher = better)", [R[n]["clinical_macro_f1"] for n in names], "{:.2f}"),
          ("Unsupported concepts (lower = better)", [R[n]["unsupported_concept_rate"] for n in names], "{:.1%}")]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (t, v, fmt) in zip(axs, panels):
    b = ax.barh(labels[::-1], v[::-1], color="#3a6fd8", height=0.55)
    for rect, x in zip(b, v[::-1]):
        ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, " " + fmt.format(x), va="center", fontsize=10)
    ax.set_title(t, fontsize=11, loc="left")
    ax.set_xlim(0, max(v) * 1.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
fig.suptitle(f"Report systems on the held-out test fold (n={R['n']} ECGs). The cardiologist row shows the limits of "
             "the text-concept extractor and of the evidence sheet's coverage.", fontsize=10, x=0.01, ha="left")
fig.tight_layout()
out = cfg.path("artifacts") / "figures" / "step8_report_systems.png"
fig.savefig(out, dpi=130)
print("wrote", out)
