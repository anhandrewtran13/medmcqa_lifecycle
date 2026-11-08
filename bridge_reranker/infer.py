"""
Bridge-entity reranker inference (LightGBM).

Used by FLOW_RAG_Starter_V20 hop-1 lock selection. Feature schema must match
prepare_training_data._feature_row + train_reranker FEATURE_COLS_*.

Can run in Colab (after Drive mount) or be exec()'d into the notebook kernel.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

FEATURE_COLS_NUMERIC = [
    "feature_nli_grounding_score",
    "feature_bridge_quality_bonus",
    "feature_combined_heuristic_score",
    "feature_rank_by_heuristic",
    "feature_group_size",
    "feature_mutex_score",
    "feature_has_mutex_score",
    "feature_in_conflict",
    "feature_claim_text_len",
    "feature_value_len",
    "feature_value_word_count",
    "feature_has_digit_in_value",
    "feature_entity_overlap_with_question",
    "feature_claim_text_overlap_with_question",
    "feature_title_in_question",
    "feature_retrieval_rank",
    "feature_hop1_query_score",
    "feature_is_title_slot",
    "feature_is_wide_span",
    "feature_rankpool_bypass",
]
FEATURE_COL_CATEGORICAL = "feature_slot_prefix"

_STATE = {
    "booster": None,
    "feature_cols": None,
    "slot_map": None,
    "loaded_from": None,
    "error": None,
}


def _tok(text):
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _jaccard(a, b):
    a, b = _tok(a), _tok(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _wide_slot_type(claim) -> str:
    slot = (getattr(claim, "slot", "") or "").strip().lower()
    if slot.startswith("title("):
        return "title"
    if slot.startswith("bridge_span("):
        return "bridge_span"
    if slot.startswith("quote("):
        return "quote"
    if slot.startswith("phrase("):
        return "phrase"
    return "other"


def feature_row_for_claim(
    claim,
    nli_score: float,
    bonus: float,
    question: str,
    entity: str,
    hop1_paras: list,
    mutex_score,
    in_conflict: bool,
    rank_in_group: int,
    group_size: int,
    hop1_query_score,
    rankpool_bypass: bool = False,
) -> dict:
    """Same schema as prepare_training_data._feature_row."""
    value = claim.value or ""
    text = claim.text or ""
    evidence_titles = set()
    para_by_id = {p.para_id: p for p in hop1_paras}
    retrieval_rank = None
    for i, pid in enumerate(claim.evidence_para_ids or []):
        p = para_by_id.get(pid)
        if p:
            evidence_titles.add(p.title)
            if retrieval_rank is None:
                retrieval_rank = i
    title_in_question = (
        any(t.lower() in (question or "").lower() for t in evidence_titles)
        if evidence_titles
        else False
    )
    wide_typ = _wide_slot_type(claim)
    return {
        "feature_nli_grounding_score": float(nli_score),
        "feature_bridge_quality_bonus": float(bonus),
        "feature_combined_heuristic_score": float(nli_score) + float(bonus),
        "feature_rank_by_heuristic": int(rank_in_group),
        "feature_group_size": int(group_size),
        "feature_mutex_score": float(mutex_score) if mutex_score is not None else -1.0,
        "feature_has_mutex_score": bool(mutex_score is not None),
        "feature_in_conflict": bool(in_conflict),
        "feature_claim_text_len": len(text),
        "feature_value_len": len(value),
        "feature_value_word_count": len(value.split()),
        "feature_has_digit_in_value": bool(re.search(r"\d", value)),
        "feature_entity_overlap_with_question": _jaccard(claim.entity or "", question),
        "feature_claim_text_overlap_with_question": _jaccard(text, question),
        "feature_title_in_question": bool(title_in_question),
        "feature_retrieval_rank": retrieval_rank if retrieval_rank is not None else 99,
        "feature_slot_prefix": (claim.slot or "").split("(")[0],
        "feature_hop1_query_score": float(hop1_query_score) if hop1_query_score is not None else -1.0,
        "feature_is_title_slot": bool(wide_typ == "title"),
        "feature_is_wide_span": bool(wide_typ in {"bridge_span", "quote", "phrase"}),
        "feature_rankpool_bypass": bool(rankpool_bypass),
    }


def _rebuild_slot_map_from_train_jsonl(train_jsonl: str, val_frac: float = 0.15, seed: int = 42) -> dict:
    """Reproduce train_reranker group_split + categorical fit (train qids only)."""
    import numpy as np

    rows = []
    with open(train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    qids = sorted({r["qid"] for r in rows})
    rng = np.random.RandomState(seed)
    qids_arr = np.array(qids, dtype=object)
    rng.shuffle(qids_arr)
    n_val = max(1, int(len(qids_arr) * val_frac))
    train_qids = set(qids_arr[n_val:])
    cats = sorted(
        {
            (r.get(FEATURE_COL_CATEGORICAL) or "none")
            for r in rows
            if r["qid"] in train_qids
        }
    )
    return {c: i for i, c in enumerate(cats)}


def load_bridge_reranker(
    model_path: str,
    report_path: Optional[str] = None,
    train_jsonl: Optional[str] = None,
    slot_map_path: Optional[str] = None,
):
    """Load LightGBM booster + feature columns + slot-prefix mapping."""
    import lightgbm as lgb

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Bridge reranker model not found: {model_path}")

    booster = lgb.Booster(model_file=model_path)
    feature_cols = list(FEATURE_COLS_NUMERIC) + ["feature_slot_prefix_code"]
    if report_path and os.path.isfile(report_path):
        with open(report_path) as f:
            report = json.load(f)
        if report.get("feature_cols"):
            feature_cols = list(report["feature_cols"])

    slot_map = None
    if slot_map_path and os.path.isfile(slot_map_path):
        with open(slot_map_path) as f:
            slot_map = {str(k): int(v) for k, v in json.load(f).items()}
    elif train_jsonl and os.path.isfile(train_jsonl):
        slot_map = _rebuild_slot_map_from_train_jsonl(train_jsonl)
    else:
        slot_map = {}

    _STATE["booster"] = booster
    _STATE["feature_cols"] = feature_cols
    _STATE["slot_map"] = slot_map
    _STATE["loaded_from"] = model_path
    _STATE["error"] = None
    return _STATE


def is_bridge_reranker_ready() -> bool:
    return _STATE.get("booster") is not None and _STATE.get("feature_cols") is not None


def _vectorize(feat_rows: list) -> list:
    import numpy as np

    cols = _STATE["feature_cols"]
    slot_map = _STATE["slot_map"] or {}
    mat = []
    for fr in feat_rows:
        row = []
        for c in cols:
            if c == "feature_slot_prefix_code":
                prefix = fr.get(FEATURE_COL_CATEGORICAL) or "none"
                row.append(float(slot_map.get(prefix, -1)))
            else:
                v = fr.get(c, 0.0)
                if isinstance(v, bool):
                    v = float(v)
                row.append(float(v) if v is not None else 0.0)
        mat.append(row)
    return np.asarray(mat, dtype=float)


def score_candidates(feat_rows: list) -> list:
    """Return list of float scores aligned with feat_rows."""
    if not is_bridge_reranker_ready():
        raise RuntimeError("Bridge reranker not loaded")
    if not feat_rows:
        return []
    X = _vectorize(feat_rows)
    scores = _STATE["booster"].predict(X)
    return [float(s) for s in scores]


def select_claim_by_reranker(claims_with_meta: list) -> tuple:
    """
    claims_with_meta: list of dicts with keys:
      claim, nli_score, bonus, mutex_score, in_conflict, feat (optional precomputed)
    Plus shared context passed via kwargs on each row: question, entity, hop1_paras,
    hop1_query_score, rank_in_group, group_size — or supply feat already.

    Returns (best_claim, debug_info).
    """
    if not claims_with_meta:
        return None, {"error": "empty"}

    # Ensure heuristic ranks exist for features
    ranked = sorted(
        claims_with_meta,
        key=lambda d: -(float(d.get("nli_score", 0.0)) + float(d.get("bonus", 0.0))),
    )
    group_size = len(ranked)
    feat_rows = []
    for rank, d in enumerate(ranked, start=1):
        if d.get("feat"):
            fr = dict(d["feat"])
            fr["feature_rank_by_heuristic"] = rank
            fr["feature_group_size"] = group_size
            feat_rows.append(fr)
        else:
            feat_rows.append(
                feature_row_for_claim(
                    d["claim"],
                    float(d["nli_score"]),
                    float(d["bonus"]),
                    d["question"],
                    d.get("entity") or "",
                    d.get("hop1_paras") or [],
                    d.get("mutex_score"),
                    bool(d.get("in_conflict")),
                    rank,
                    group_size,
                    d.get("hop1_query_score"),
                    rankpool_bypass=bool(d.get("rankpool_bypass")),
                )
            )

    scores = score_candidates(feat_rows)
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return ranked[best_i]["claim"], {
        "winner_value": getattr(ranked[best_i]["claim"], "value", None),
        "winner_score": scores[best_i],
        "ranking": [
            {
                "value": getattr(ranked[i]["claim"], "value", None),
                "score": scores[i],
            }
            for i in order
        ],
    }
