"""Shared eval-index helper for FLOW-RAG range / batch shards.

Works for any example list with ``example_id`` (HotpotQA or MedMCQA adapters).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


def _fingerprint(example_ids: Sequence[str], meta: dict) -> str:
    payload = json.dumps(
        {"ids": list(example_ids), "meta": meta},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_or_load_eval_index(
    pool: Sequence[Any],
    index_dir: str | Path,
    split: str = "validation",
    subset: str = "default",
    filter_bridge_only: bool = False,
    batch_size: int = 100,
    rebuild: bool = False,
    shared_index_path: str | Path | None = None,
) -> dict:
    """Build or reuse a stable ordered id list for sharded GPU runs."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    local_path = index_dir / "eval_index.json"
    shared_path = Path(shared_index_path) if shared_index_path else None

    example_ids = [str(ex.example_id) for ex in pool]
    meta = {
        "split": split,
        "subset": subset,
        "filter_bridge_only": bool(filter_bridge_only),
        "batch_size": int(batch_size),
        "n_total": len(example_ids),
    }
    built = {
        "example_ids": example_ids,
        "n_total": len(example_ids),
        "batch_size": int(batch_size),
        "split": split,
        "subset": subset,
        "filter_bridge_only": bool(filter_bridge_only),
        "fingerprint": _fingerprint(example_ids, meta),
    }

    def _load(path: Path) -> dict | None:
        if not path.is_file():
            return None
        with path.open() as f:
            return json.load(f)

    existing = None
    if not rebuild:
        if shared_path is not None:
            existing = _load(shared_path)
        if existing is None:
            existing = _load(local_path)

    if existing is not None and not rebuild:
        if existing.get("n_total") != len(example_ids):
            print(
                f"[index] WARNING: loaded index n_total={existing.get('n_total')} "
                f"but pool has {len(example_ids)} — set INDEX_REBUILD=True to rebuild"
            )
        print(f"[index] loaded existing index fingerprint={existing.get('fingerprint')}")
        return existing

    for path in filter(None, [local_path, shared_path]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(built, f, indent=2)
        print(f"[index] wrote {path} fingerprint={built['fingerprint']} n={built['n_total']}")
    return built


def load_range_examples(pool: Sequence[Any], eval_index: dict, start: int, end: int) -> list:
    by_id = {str(ex.example_id): ex for ex in pool}
    ids = list(eval_index["example_ids"])
    start = max(0, int(start))
    end = min(len(ids), int(end))
    out = []
    missing = []
    for eid in ids[start:end]:
        ex = by_id.get(str(eid))
        if ex is None:
            missing.append(eid)
        else:
            out.append(ex)
    if missing:
        raise KeyError(f"{len(missing)} index ids missing from pool, e.g. {missing[:3]}")
    return out


def load_batch_examples(pool: Sequence[Any], eval_index: dict, batch_id: int) -> list:
    batch_size = int(eval_index.get("batch_size") or 100)
    start = int(batch_id) * batch_size
    end = start + batch_size
    return load_range_examples(pool, eval_index, start, end)


def range_run_name(start: int, end: int, suffix: str = "") -> str:
    return f"range_{int(start):04d}_{int(end):04d}{suffix}"


def list_completed_shards(runs_root: str | Path) -> list[str]:
    root = Path(runs_root)
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        if d.is_dir() and (d / "ablation_details.json").is_file():
            out.append(d.name)
    return out
