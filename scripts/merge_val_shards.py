#!/usr/bin/env python3
"""Merge MedMCQA validation shard ablation_df.csv into a micro-averaged summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    "medmcqa_val_medical_0_1000_bf16",
    "medmcqa_val_medical_1000_2000_bf16",
    "medmcqa_val_medical_2000_3000_bf16",
    "medmcqa_val_medical_3000_4183_bf16",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "final_val_medmcqa",
    )
    p.add_argument("--run-names", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    out_dir = args.out_dir or (args.runs_dir / "medmcqa_val_merged_bf16")
    out_dir.mkdir(parents=True, exist_ok=True)

    acc: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "em_sum": 0.0,
            "relaxed_sum": 0.0,
            "f1_sum": 0.0,
            "locked_sum": 0.0,
            "verified_sum": 0.0,
            "fallback_sum": 0.0,
        }
    )
    used, missing = [], []
    for name in args.run_names:
        csv_path = args.runs_dir / name / "ablation_df.csv"
        if not csv_path.is_file():
            missing.append(name)
            continue
        used.append(name)
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                setting, n = row["setting"], float(row["n"])
                b = acc[setting]
                b["n"] += n
                b["em_sum"] += float(row["EM"]) * n
                b["relaxed_sum"] += float(row["relaxed_EM"]) * n
                b["f1_sum"] += float(row["F1"]) * n
                b["locked_sum"] += float(row.get("avg_locked") or 0) * n
                b["verified_sum"] += float(row.get("pct_verified_llm") or 0) * n
                b["fallback_sum"] += float(row.get("pct_fallback") or 0) * n

    if not used:
        raise SystemExit(f"No shard results under {args.runs_dir}")
    if missing:
        print(f"WARNING: missing shards: {missing}")

    order = [
        "naive_rag",
        "S1_multihop",
        "S2_multihop_modularqa",
        "S3_lifecycle",
        "S4_lifecycle_rr",
        "S5_lifecycle_rr_sf",
        "full",
    ]
    rows = []
    for setting in order:
        if setting not in acc:
            continue
        b, n = acc[setting], acc[setting]["n"]
        rows.append(
            {
                "setting": setting,
                "EM": b["em_sum"] / n,
                "relaxed_EM": b["relaxed_sum"] / n,
                "F1": b["f1_sum"] / n,
                "n": int(n),
                "avg_locked": b["locked_sum"] / n,
                "pct_verified_llm": b["verified_sum"] / n,
                "pct_fallback": b["fallback_sum"] / n,
            }
        )

    out_csv = out_dir / "ablation_df.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "merged_at": datetime.now().isoformat(timespec="seconds"),
        "shards_used": used,
        "shards_missing": missing,
        "n_total": rows[0]["n"] if rows else 0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {out_csv}")
    for r in rows:
        print(f"  {r['setting']:24s} EM={r['EM']:.4f}  n={r['n']}")


if __name__ == "__main__":
    main()
