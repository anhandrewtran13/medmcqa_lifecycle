#!/usr/bin/env python3
"""Error analysis: why S2 (multihop+ModularQA) beats S3 (lifecycle) on MedMCQA.

Walks finished shard ablation_details.json files and writes:
  - summary counts / letter confusion
  - sample S2-correct & S3-wrong cases with locked claims / traces
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARDS = [
    "medmcqa_val_medical_0_1000_bf16",
    "medmcqa_val_medical_1000_2000_bf16",
    "medmcqa_val_medical_2000_3000_bf16",
    "medmcqa_val_medical_3000_4183_bf16",
]


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


def _ok(pred, gold) -> bool:
    return _letter(pred) == _letter(gold) and bool(_letter(gold))


def _trace_summary(row: dict) -> dict:
    meta = row.get("metadata") or {}
    hops = row.get("hops") or []
    locked = []
    for h in hops if isinstance(hops, list) else []:
        if not isinstance(h, dict):
            continue
        locked.append(
            {
                "hop": h.get("hop") or h.get("hop_idx"),
                "subq": (h.get("subquestion") or h.get("query") or "")[:160],
                "value": (h.get("value") or h.get("claim") or h.get("answer") or "")[:120],
                "locked": h.get("locked"),
                "score": h.get("score") or h.get("lock_score"),
            }
        )
    return {
        "pred": _letter(row.get("pred_answer")),
        "method": row.get("method"),
        "metadata_keys": sorted(meta.keys())[:20] if isinstance(meta, dict) else [],
        "avg_locked": meta.get("avg_locked") or meta.get("locked"),
        "status": meta.get("status") or meta.get("answer_status"),
        "hops": locked[:4],
        "trace0": (row.get("trace") or [None])[0] if row.get("trace") else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "final_val_medmcqa",
    )
    p.add_argument("--shards", nargs="+", default=DEFAULT_SHARDS)
    p.add_argument("--base", default="S2_multihop_modularqa")
    p.add_argument("--treat", default="S3_lifecycle")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: experiments/s2_vs_s3_analysis.json",
    )
    p.add_argument("--samples", type=int, default=25)
    args = p.parse_args()
    out = args.out or (REPO_ROOT / "experiments" / "s2_vs_s3_analysis.json")

    n = 0
    both_right = both_wrong = 0
    s2_only = s3_only = 0
    pred_s2, pred_s3, gold_c = Counter(), Counter(), Counter()
    flip_s2_win, flip_s3_win = [], []
    letter_flip = Counter()  # (s2,s3,gold)
    status_s3 = Counter()

    for name in args.shards:
        path = args.runs_dir / name / "ablation_details.json"
        if not path.is_file():
            print(f"WARNING: skip missing {path}")
            continue
        data = json.loads(path.read_text())
        a_map = {r["example_id"]: r for r in data[args.base]}
        b_map = {r["example_id"]: r for r in data[args.treat]}
        for eid in set(a_map) & set(b_map):
            a, b = a_map[eid], b_map[eid]
            gold = _letter(a.get("gold_answer"))
            pa, pb = _letter(a.get("pred_answer")), _letter(b.get("pred_answer"))
            ok_a, ok_b = _ok(pa, gold), _ok(pb, gold)
            n += 1
            gold_c[gold] += 1
            pred_s2[pa] += 1
            pred_s3[pb] += 1
            meta = b.get("metadata") or {}
            if isinstance(meta, dict):
                status_s3[str(meta.get("status") or meta.get("answer_status") or "")] += 1
            if ok_a and ok_b:
                both_right += 1
            elif (not ok_a) and (not ok_b):
                both_wrong += 1
            elif ok_a and not ok_b:
                s2_only += 1
                letter_flip[(pa, pb, gold)] += 1
                if len(flip_s2_win) < args.samples:
                    flip_s2_win.append(
                        {
                            "example_id": eid,
                            "shard": name,
                            "gold": gold,
                            "s2_pred": pa,
                            "s3_pred": pb,
                            "question": (a.get("question") or "")[:280],
                            "s2": _trace_summary(a),
                            "s3": _trace_summary(b),
                        }
                    )
            else:
                s3_only += 1
                if len(flip_s3_win) < args.samples:
                    flip_s3_win.append(
                        {
                            "example_id": eid,
                            "shard": name,
                            "gold": gold,
                            "s2_pred": pa,
                            "s3_pred": pb,
                            "question": (a.get("question") or "")[:280],
                            "s2": _trace_summary(a),
                            "s3": _trace_summary(b),
                        }
                    )

    summary = {
        "base": args.base,
        "treat": args.treat,
        "shards": args.shards,
        "n": n,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "s2_only_correct": s2_only,
        "s3_only_correct": s3_only,
        "em_base": (both_right + s2_only) / n if n else 0.0,
        "em_treat": (both_right + s3_only) / n if n else 0.0,
        "net_em_delta_treat_minus_base": (s3_only - s2_only) / n if n else 0.0,
        "gold_dist": dict(gold_c),
        "pred_dist_s2": dict(pred_s2),
        "pred_dist_s3": dict(pred_s3),
        "s3_status_dist": dict(status_s3),
        "top_letter_flips_s2_correct_s3_wrong": [
            {"s2": a, "s3": b, "gold": g, "count": c}
            for (a, b, g), c in letter_flip.most_common(15)
        ],
        "samples_s2_win": flip_s2_win,
        "samples_s3_win": flip_s3_win,
        "takeaways": [
            "If s2_only_correct >> s3_only_correct, lifecycle hop-locking is over-constraining MedMCQA.",
            "Inspect samples_s2_win[*].s3.hops for Option-title locks / wrong locked claims.",
            "On MedMCQA, prefer S2 (or S3 without Hotpot-trained bridge) unless lifecycle is redesigned.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{args.base} vs {args.treat}  n={n}")
    print(f"  both right={both_right}  both wrong={both_wrong}")
    print(f"  S2-only={s2_only}  S3-only={s3_only}")
    print(f"  EM S2={summary['em_base']:.4f}  S3={summary['em_treat']:.4f}  Δ={summary['net_em_delta_treat_minus_base']:+.4f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
