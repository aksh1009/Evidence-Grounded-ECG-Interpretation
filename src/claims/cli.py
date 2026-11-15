"""Command line: ``claims analyze <ecg_id>`` or ``claims analyze --record path/to/wfdb_record``.

Examples
  claims analyze 180                      # PTB-XL record 180 (uses the local signal cache)
  claims analyze 180 --no-llm             # measurements + CNN + template report only (no GPU needed)
  claims analyze --record data/ptb-xl/records500/00000/00180_hr --age 81 --sex female
  claims analyze 180 --json out.json      # full machine-readable result
"""
from __future__ import annotations

import argparse
import json
import sys


def _print(res: dict, reference: str | None):
    ev = res["evidence"]
    p = ev["patient"]
    print(f"\nECG {res['ecg_id']}  |  {p['age'] if p['age'] is not None else '?'} y, {p['sex'] or 'sex unknown'}")
    print("\nMEASUREMENTS")
    for c in ev["findings"]:
        v = "" if c["value"] is None else f" {c['value']} {c['unit']}"
        leads = "" if c["leads"] in (["all"], ["global"]) else f"  leads {','.join(c['leads'])}"
        print(f"  [{c['id']}] {c['finding']}{v}{leads}  @ {c['window_s'][0]}-{c['window_s'][1]} s")
    print("\nCNN DIAGNOSES (above per-class threshold)")
    for c in ev["diagnoses"]:
        tag = ""
        if c["supported_by"]:
            tag += f"  supported by {','.join(c['supported_by'])}"
        if c["contradicted_by"]:
            tag += f"  CONTRADICTED by {','.join(c['contradicted_by'])}"
        print(f"  [{c['id']}] {c['finding']}  p={c['probability']:.2f}{tag}")
    if ev["flags"]:
        print("\nFLAGS\n  " + "\n  ".join(ev["flags"]))
    print("\nTEMPLATE REPORT\n  " + res["template_report"])
    if "llm_report" in res:
        g = res["grounding"]
        print("\nLLM REPORT (raw)\n  " + res["llm_report"])
        print("\nLLM REPORT (verified: cited, unsupported removed, low-confidence hedged)\n  " + g["verified_report"])
        print(f"  concepts {g['n_concepts']}, unsupported {g['n_unsupported']}, hedged {g['n_low_confidence_unhedged']}")
    if reference:
        print("\nCARDIOLOGIST REPORT (PTB-XL, English)\n  " + reference)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="claims", description="Evidence-grounded 12-lead ECG interpretation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse one ECG")
    a.add_argument("ecg_id", nargs="?", type=int, help="PTB-XL ecg_id")
    a.add_argument("--record", help="path to a WFDB record (500 Hz, 12 leads, 10 s) instead of a PTB-XL id")
    a.add_argument("--age", type=float)
    a.add_argument("--sex", choices=["male", "female"])
    a.add_argument("--no-llm", action="store_true", help="skip the LLM (template report only)")
    a.add_argument("--json", help="also write the full result to this JSON file")
    args = ap.parse_args(argv)
    if args.ecg_id is None and not args.record:
        ap.error("give a PTB-XL ecg_id or --record")

    from .config import load_config
    from .data import _read_one, load_metadata, load_signals
    from .pipeline import Analyzer

    cfg = load_config()
    reference = None
    if args.record:
        x = _read_one((args.record, 5000))
        eid = args.ecg_id or 0
        age = args.age
        sex = {"male": 0, "female": 1}.get(args.sex)
    else:
        meta = load_metadata(cfg)
        if args.ecg_id not in meta.index:
            ap.error(f"ecg_id {args.ecg_id} not in PTB-XL")
        x = load_signals(cfg, meta)[meta.index.get_loc(args.ecg_id)]
        eid = args.ecg_id
        row = meta.loc[eid]
        age = args.age if args.age is not None else row.age
        sex = {"male": 0, "female": 1}.get(args.sex, int(row.sex))
        rep_path = cfg.path("artifacts") / "reports_en.parquet"
        if rep_path.exists():
            import pandas as pd
            rep = pd.read_parquet(rep_path)
            if eid in rep.index:
                reference = rep.at[eid, "report_en"] if "report_en" in rep.columns else None
    res = Analyzer(cfg, use_llm=not args.no_llm).analyze(x, eid, age, sex)
    _print(res, reference)
    if args.json:
        res = {k: v for k, v in res.items() if k != "measurements"} | {"measurements": res["measurements"]["globals"]
                                                                     if res["measurements"].get("ok") else None}
        json.dump(res, open(args.json, "w", encoding="utf-8"), indent=1, default=float)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
