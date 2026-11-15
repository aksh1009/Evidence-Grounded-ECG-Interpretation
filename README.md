# CLAIMS: evidence-grounded ECG reports

CLAIMS reads a 12-lead ECG and writes a short report in cardiologist style. Every statement in the report points to a piece of evidence you can check: a measurement such as the QRS duration, or a classifier finding together with the leads and time window it relates to. Anything the evidence doesn't support is cut, and findings the classifier is unsure about are worded as "possible".

## Problem

Language models can write fluent ECG reports, but they also invent findings, and a reader can't tell which sentences are backed by the signal. CLAIMS splits the job:

- **Signal processing and a CNN** decide *what is in the ECG*.
- **The LLM** only puts that evidence into words.
- **A rule-based verifier** checks each sentence against the evidence before anything is shown.

## Workflow

```
PTB-XL 12-lead ECG (500 Hz, 10 s)
   ├─► 1. Measurement engine  (HR, PR, QRS, QT/QTc, axis, ST/T per lead, P waves, Q waves)   → [M1], [M2] …
   ├─► 2. CNN, 35 classes     (23 diagnostic subclasses + 12 rhythms), Integrated Gradients   → [C1], [C2] …
   ▼
3. Evidence sheet  – guideline rules (4th UDMI ST elevation, Sokolow/Cornell LVH, QTc Fridericia, …),
                     CNN findings marked supported / contradicted by measurements,
                     ST/Q changes with wide QRS marked "secondary" (they do not support infarction)
   ▼
4. LLM (Qwen2.5-1.5B-Instruct + LoRA) writes the report from the evidence sheet only
   ▼
5. Verifier – extracts clinical concepts from each sentence, attaches citations [M#]/[C#],
              cuts unsupported phrases, hedges low-confidence findings ("possible …")
```

## Data

- **PTB-XL v1.0.3:** 21,799 ECGs from 18,869 patients, with cardiologist SCP-ECG labels and free-text reports (German, English, Swedish).
- **Split:** the official stratified folds, with no patient in more than one split:

  | Folds | Split | ECGs |
  |---|---|---|
  | 1–8 | train | 17,418 |
  | 9 | validation | 2,183 |
  | 10 | test | 2,198 |

  Folds 9 and 10 were all checked by a cardiologist.
- **PTB-XL+:** measurements from GE 12SL and Philips Uni-G. We use them **only as references** to check our measurement engine, never as training labels.
- **Reports:** German and Swedish reports were translated to English (glossary plus MarianMT `opus-mt-de-en`) to serve as LLM training targets.

## Results on the test fold

Fold 10 was used once, at the end.

### Measurements vs GE 12SL (median absolute difference)

| HR | PR | QRS | QT | Axis | ST at J point | T amplitude |
|---|---|---|---|---|---|---|
| 0.3 bpm | 10 ms | 4 ms | 16 ms | 5° | 0.012 mV | 0.013 mV |

For comparison, 12SL and Uni-G differ from each other by 4 ms (PR), 4 ms (QRS) and 6 ms (QT). That gap is the agreement ceiling between two commercial algorithms.

### CNN (XResNet1d, 100 Hz)

| Metric | Value |
|---|---|
| Macro-AUC, 35 classes | **0.934** (validation 0.945) |
| Macro-AUC, 5 superclasses | 0.922 |

Superclass AUCs: NORM 0.94, MI 0.92, STTC 0.93, CD 0.92, HYP 0.90.

### Reports

2,158 test ECGs have a cardiologist report. "Clinical F1" compares the diagnoses extracted from each report's text with the PTB-XL labels.

| System | Clinical micro-F1 | Macro-F1 | Unsupported concepts | Low-confidence findings stated as fact | ROUGE-L vs cardiologist |
|---|---|---|---|---|---|
| Cardiologist (ceiling) | 0.75 | 0.59 | 17.6%* | – | – |
| Template (no LLM) | 0.72 | 0.51 | 0% | 11% | 0.26 |
| Zero-shot LLM | 0.33 | 0.11 | 10.4% | 84% | 0.03 |
| Fine-tuned LLM | 0.69 | 0.43 | 2.0% | 78% | 0.51 |
| **Fine-tuned LLM + verifier (CLAIMS output)** | **0.70** | **0.43** | **0%** | **0%** | – |

\* The cardiologist row measures the limits of the evaluation itself, not cardiologist errors. Our text extractor misses some phrasings, and the evidence sheet doesn't capture everything a human reads (for example, comparisons with earlier ECGs).

**How to read this:**
- **Fine-tuning gives style and safety.** Wording similarity to the cardiologist rises from 0.03 to 0.51, and invented findings fall from 10.4% to 2.0%.
- **The verifier removes what remains.** Unsupported claims go from 2.0% to 0%, and accuracy rises slightly (0.69 → 0.70).
- **The LLM does not add diagnostic accuracy over the template.** Accuracy comes from the CNN and the measurements; the LLM turns them into readable text.

![report systems](artifacts/figures/step8_report_systems.png)

## Run it

```bash
pip install -e .
# put PTB-XL in data/ptb-xl and the PTB-XL+ feature CSVs in data/ptb-xl-plus (paths in config.yaml)

python scripts/s1_prepare_data.py          # labels, folds, int16 signal cache
python scripts/s3_measure.py all           # measurement engine on all ECGs + agreement vs 12SL / Uni-G
python scripts/s4_train_cnn.py final       # CNN (folds 1-8, model selection on fold 9)
python scripts/s4_train_cnn.py oof         # out-of-fold CNN predictions for the training folds
python scripts/s5_build_evidence.py        # evidence sheets + attributions
python scripts/s6a_translate_reports.py    # German/Swedish reports -> English
python scripts/s6_llm.py prepare           # evidence -> report pairs (faithful pairs only for training)
python scripts/s6_llm.py train             # LoRA fine-tuning (about 1 h on an 8 GB GPU)
python scripts/s6_llm.py generate          # template / zero-shot / fine-tuned reports for the test fold
python scripts/s8_evaluate.py              # all test-fold metrics -> artifacts/eval/metrics.json
python scripts/make_figures.py; python scripts/make_eval_figure.py
python scripts/s9_build_demo.py            # artifacts/demo/index.html
pytest                                     # 19 unit tests (no dataset or GPU needed)
```

Analyse a single ECG end to end:

```bash
claims analyze 180                 # PTB-XL record id
claims analyze 180 --no-llm        # measurements + CNN + template report, no GPU needed
claims analyze --record path/to/wfdb_record --age 61 --sex female    # any 500 Hz 12-lead WFDB record
claims analyze 180 --json out.json
```

The demo page (`artifacts/demo/index.html`) opens offline in any browser. It covers 15 curated test ECGs and shows:
- the 12-lead trace;
- every evidence item, which you can click to shade its time window on the cited leads;
- our measurements next to 12SL;
- the CNN probabilities and thresholds;
- the reports side by side.

## Design decisions to defend

| Decision | Why |
|---|---|
| Official PTB-XL folds; test fold used once | Patient-disjoint; comparable with published results; no tuning on test data |
| 12SL used only as a measurement reference | Labels come from cardiologists; using 12SL as labels would teach the system to copy a commercial algorithm |
| Out-of-fold CNN predictions for LLM training | Otherwise the LLM learns from over-confident, already-memorised predictions that it never sees at test time |
| LLM trained only on "faithful" pairs | If the target report mentions things not in the evidence, the model learns to invent |
| Loss only on report tokens | The model learns to write reports, not to repeat the prompt |
| Wide QRS makes ST/Q findings "secondary" | In bundle branch block and pacing, ST/Q changes are expected and do not indicate infarction |
| Verifier cuts phrases, not whole sentences | The LLM often writes one long sentence; dropping it all would discard supported findings too |
| Hedging when CNN p < 0.5 and no measurement support | A finding that is in the evidence but uncertain must not be stated as fact |

## Limitations

- **Measurement precision:** PR and QT are less precise than commercial algorithms (median error about 10 and 16 ms vs 12SL). QRS duration is overestimated in bundle branch block and pacing (+8 to +14 ms on median; 13–19% of these ECGs are off by more than 30 ms).
- **Rare findings:**
  - The CNN is weak on classes with few examples (for example, posterior MI has 2 test cases).
  - The LLM under-reports rare findings (LVH, RVH, right atrial enlargement): F1 for these is far below the template's.
  - It has memorised a stock German-derived phrase for ST/T changes ("… ischaemia or left ventricular strain").
- **Verifier blind spots:**
  - It checks that stated findings exist in the evidence. It does not check "normal" statements; for example, "normal axis" is not checked against a measured right axis deviation.
  - It relies on a rule-based concept extractor, which misses some phrasings.
- **Explanations:** Integrated Gradients highlights relevant leads on average (deleting the top lead hurts about 5× more than a random lead). For a single ECG it beats a random lead only 55% of the time, because the 12 leads carry overlapping information.
- **Labels:** PTB-XL long-QT labels are incomplete. Our Fridericia QTc rule has 3% PPV against them, which is no better or worse than GE 12SL.
- **Translation:** machine-translated reports add noise to the LLM training targets.
- **Not a medical device:** this is research software and has not been clinically validated.

## Repository layout

```
src/claims/   data.py  signal.py  measure.py  validate_measure.py  model.py  train_cnn.py  explain.py
              claims_builder.py (evidence)  report_text.py (translation)  llm.py  grounding.py (verifier)
              pipeline.py (one ECG end to end)  cli.py
scripts/      s1 … s9 pipeline steps, figures, demo builder
tests/        unit tests on synthetic ECGs
artifacts/    (generated) caches, models, evidence, reports, metrics, figures, demo
```
