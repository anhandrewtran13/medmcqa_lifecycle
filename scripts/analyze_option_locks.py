#!/usr/bin/env python3
"""Measure Option-title lock rate in an ablation_details.json (S3/full)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flow_rag.medmcqa_domain import is_option_title_lock  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--details",
        type=Path,
        required=True,
        help="Path to ablation_details.json",
    )
    p.add_argument(
        "--setting",
        default="S3_lifecycle",
        help="Ablation setting key (default S3_lifecycle)",
    )
    args = p.parse_args()
    data = json.loads(args.details.read_text())
    rows = data.get(args.setting)
    if not rows:
        raise SystemExit(f"Missing setting {args.setting!r} in {args.details}")

    locks = Counter()
    option_n = 0
    empty = 0
    soft_fb = 0
    for r in rows:
        md = r.get("metadata") or {}
        lv = md.get("locked_value")
        if not lv:
            empty += 1
        else:
            locks[str(lv)[:120]] += 1
            if is_option_title_lock(lv):
                option_n += 1
        # Soft-fallback annotation lives on trace in newer runs
        for tr in r.get("trace") or []:
            if isinstance(tr, dict) and tr.get("medmcqa_soft_lock_fallback"):
                soft_fb += 1
                break

    n = len(rows)
    print(f"setting={args.setting} n={n}")
    print(f"empty_lock={empty} ({empty/n:.1%})")
    print(f"option_title_locks={option_n} ({option_n/n:.1%})")
    print(f"soft_lock_fallback_traces={soft_fb}")
    print("top locked_value:")
    for k, v in locks.most_common(15):
        flag = " [OPTION]" if is_option_title_lock(k) else ""
        print(f"  {v:4d}  {k}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
