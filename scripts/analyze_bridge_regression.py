#!/usr/bin/env python3
"""Compare two ablation settings (default S3 → S4) on a finished run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _letter(x) -> str:
    s = str(x or "").strip().upper()
    if not s:
        return ""
    if s[0] in "ABCD":
        return s[0]
    for ch in s:
        if ch in "ABCD":
            return ch
    return s[:1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="smoke_medmcqa_0_50_bf16")
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "final_val_medmcqa",
    )
    p.add_argument("--base", default="S3_lifecycle")
    p.add_argument("--treat", default="S4_lifecycle_rr")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    details_path = args.runs_dir / args.run / "ablation_details.json"
    if not details_path.is_file():
        raise SystemExit(f"Missing {details_path}")
    data = json.loads(details_path.read_text())
    base = {r["example_id"]: r for r in data[args.base]}
    treat = {r["example_id"]: r for r in data[args.treat]}
    ids = sorted(set(base) & set(treat))

    flip_bad, flip_good = [], []
    both_right = both_wrong = 0
    for eid in ids:
        a, b = base[eid], treat[eid]
        gold = _letter(a.get("gold_answer"))
        pa, pb = _letter(a.get("pred_answer")), _letter(b.get("pred_answer"))
        ok_a, ok_b = pa == gold, pb == gold
        if ok_a and ok_b:
            both_right += 1
        elif (not ok_a) and (not ok_b):
            both_wrong += 1
        elif ok_a and not ok_b:
            flip_bad.append(
                {
                    "example_id": eid,
                    "gold": gold,
                    "base_pred": pa,
                    "treat_pred": pb,
                    "question": (a.get("question") or "")[:200],
                }
            )
        else:
            flip_good.append(
                {
                    "example_id": eid,
                    "gold": gold,
                    "base_pred": pa,
                    "treat_pred": pb,
                    "question": (a.get("question") or "")[:200],
                }
            )

    n = len(ids)
    summary = {
        "run": args.run,
        "base": args.base,
        "treat": args.treat,
        "n": n,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "flip_to_wrong": len(flip_bad),
        "flip_to_right": len(flip_good),
        "net_em_delta": (len(flip_good) - len(flip_bad)) / n if n else 0.0,
        "flip_to_wrong_examples": flip_bad[:30],
        "flip_to_right_examples": flip_good[:30],
    }
    out = args.out or (args.runs_dir / args.run / f"bridge_regression_{args.base}_vs_{args.treat}.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{args.run}: {args.base}→{args.treat} n={n} "
        f"flip→wrong={len(flip_bad)} flip→right={len(flip_good)} "
        f"netΔ={summary['net_em_delta']:+.4f}"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
