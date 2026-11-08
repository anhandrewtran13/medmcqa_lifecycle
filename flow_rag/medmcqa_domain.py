"""MedMCQA / medical-domain adapters for the Lifecycle (FLOW-RAG) pipeline.

HotpotQA assumed Wikipedia entity bridges. MedMCQA paragraphs are the four
answer options (titles historically ``Option A``…``Option D``), so lifecycle
locking those titles poisons Hop-2. This module:

1. Reloads MedMCQA with option *text* as titles + optional explanation paras
2. Rejects option-letter / option-title / meta-option bridges
3. Mines medical concepts from the stem (not option titles) for hop-1
4. Soft-falls back to S2-style multihop when no valid medical lock exists

Applied via :func:`apply_medmcqa_domain_patches` after the pipeline bytecode loads.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# Option / meta-option detection
# ---------------------------------------------------------------------------

_OPTION_TITLE_RE = re.compile(
    r"^(?:option\s+)?([ABCD])(?:\s*[).:\-–—]\s*.*)?$",
    re.IGNORECASE,
)
_BARE_LETTER_RE = re.compile(r"^[ABCD]$", re.IGNORECASE)
_META_OPTION_RE = re.compile(
    r"^(all|none|both|either)\s+of\s+the\s+above\b|"
    r"^(all|none)\s+are\s+correct\b|"
    r"^not\s+known\b|"
    r"^all$",
    re.IGNORECASE,
)
_OPTION_PREFIX_RE = re.compile(r"^[ABCD][).:\-–—]\s*", re.IGNORECASE)
_EXPLANATION_TITLE_RE = re.compile(r"^explanation(?:\s*\d+)?$", re.IGNORECASE)
_NUMERIC_ONLY_RE = re.compile(r"^[\d.,%]+$")
_WEAK_LOCK_VALUES = {
    "explanation",
    "all",
    "none",
    "both",
    "either",
    "impulse",  # common fragment from myelinated-nerve stems; too vague alone
    "option",
    "options",
    "answer",
    "following",
}

# Strip MCQ chrome before entity / concept mining
_ANSWER_INSTR_RE = re.compile(
    r"\n\s*Answer with a single letter.*$",
    re.IGNORECASE | re.DOTALL,
)
_OPTIONS_BLOCK_RE = re.compile(
    r"\n\s*[ABCD][).:\-–—]\s+.+(?:\n\s*[ABCD][).:\-–—]\s+.+)*\s*$",
    re.IGNORECASE | re.DOTALL,
)

def is_option_title_lock(value: Any) -> bool:
    """True if *value* is an option letter/title (must not be locked as bridge)."""
    if value is None:
        return False
    v = str(value).strip()
    if not v:
        return False
    if _BARE_LETTER_RE.match(v):
        return True
    m = _OPTION_TITLE_RE.match(v)
    if m and (v.lower().startswith("option") or len(v) <= 3):
        return True
    # "Option C" exact / "option c"
    if re.fullmatch(r"option\s+[ABCD]", v, flags=re.IGNORECASE):
        return True
    # Letter-prefixed option text: "A. Zygomatic bone", "C. PCA"
    if _OPTION_PREFIX_RE.match(v):
        return True
    return False


def is_meta_option_value(value: Any) -> bool:
    if value is None:
        return False
    return bool(_META_OPTION_RE.match(str(value).strip()))


def is_explanation_title_lock(value: Any) -> bool:
    if value is None:
        return False
    return bool(_EXPLANATION_TITLE_RE.match(str(value).strip()))


def is_invalid_medmcqa_bridge_lock(value: Any) -> bool:
    """Unified gate for soft-fallback / plausibility (smoke n=20 failure modes)."""
    if value is None:
        return True
    v = str(value).strip()
    if not v:
        return True
    if is_option_title_lock(v) or is_meta_option_value(v) or is_explanation_title_lock(v):
        return True
    if _NUMERIC_ONLY_RE.match(v):
        return True
    if v.lower() in _WEAK_LOCK_VALUES:
        return True
    # Full / near-full question stems accidentally mined as bridges
    if v.lower().startswith(
        ("which of the following", "all of the following", "what is the", "what are the")
    ):
        return True
    if len(v) > 120:
        return True
    # Single very short token (often chrome), allow known clinical acronyms ≥2 caps
    toks = v.split()
    if len(toks) == 1 and len(v) <= 3 and not v.isupper():
        return True
    return False


def strip_option_letter_prefix(value: str) -> str:
    return _OPTION_PREFIX_RE.sub("", value.strip()).strip()


def medmcqa_question_stem(question: str) -> str:
    """Question text without options block / answer instruction."""
    q = question or ""
    q = _ANSWER_INSTR_RE.sub("", q)
    q = _OPTIONS_BLOCK_RE.sub("", q)
    return q.strip()


_QUESTION_STOP = {
    "all",
    "which",
    "what",
    "where",
    "when",
    "how",
    "true",
    "false",
    "none",
    "following",
    "except",
    "most",
    "best",
    "common",
    "not",
    "the",
    "a",
    "an",
    "in",
    "of",
    "for",
    "and",
    "or",
    "with",
    "answer",
    "option",
    "options",
    "choose",
    "select",
    "correct",
    "incorrect",
    "statement",
    "statements",
    "regarding",
    "concerning",
    "about",
    "drug",
    "choice",
    "feature",
    "features",
    "cause",
    "causes",
    "type",
    "types",
    "true",
    "false",
}

_QUESTION_PREFIX_RE = re.compile(
    r"^(?:all\s+of\s+the\s+following|which\s+of\s+the\s+following|"
    r"which\s+one\s+of\s+the\s+following|what\s+is\s+the|"
    r"what\s+are\s+the|which\s+is\s+(?:the\s+)?(?:most|best|true|false|correct)|"
    r"all\s+are\s+true\s+except)\b[\s:,\-]*",
    re.IGNORECASE,
)


def _is_weak_entity(cand: str) -> bool:
    toks = [t.lower() for t in re.findall(r"[A-Za-z0-9']+", cand)]
    if not toks:
        return True
    if all(t in _QUESTION_STOP for t in toks):
        return True
    if toks[0] in _QUESTION_STOP and len(toks) <= 3:
        return True
    # Reject stems that are mostly interrogative chrome
    joined = " ".join(toks)
    if joined.startswith(
        ("which of", "what is", "what are", "all of the following", "which one")
    ):
        return True
    return False


def extract_medical_bridge_entity(question: str) -> Optional[str]:
    """Prefer a clinical concept from the MCQ stem (not a Hotpot film/director cue)."""
    stem = medmcqa_question_stem(question)
    if not stem:
        return None
    stem_core = _QUESTION_PREFIX_RE.sub("", stem).strip(" ?:.-")

    # High-precision clinical patterns
    for pat in (
        r"(?:except|excluding)\s*[:\-]?\s*(.+?)\s*$",
        r"(?:regarding|concerning|about)\s+(.+?)(?:\s*[?:\.]|\s*$)",
        r"(?:developed from|derived from|caused by|produced by|secreted by)\s+(.+?)(?:\s*[?:\.]|\s*$)",
        r"(?:treatment of|management of|diagnosis of|feature(?:s)? of|function of|"
        r"drug of choice for|DOC for|treatment for)\s+(.+?)(?:\s*[?:\.]|\s*$)",
        r"(?:not true for|true for|characteristic of|seen in|found in)\s+(.+?)(?:\s*[?:\.]|\s*$)",
    ):
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            cand = _clean_entity(m.group(1))
            if cand and not is_option_title_lock(cand) and not _is_weak_entity(cand):
                return cand

    # Capitalized multi-word medical phrases in stem (prefer content after chrome)
    search_in = stem_core or stem
    caps = re.findall(
        r"\b([A-Z][A-Za-z0-9'’\-/]*(?:\s+[A-Z][A-Za-z0-9'’\-/]*){0,5})\b",
        search_in,
    )
    ranked = []
    for c in caps:
        if _is_weak_entity(c) or is_option_title_lock(c) or is_meta_option_value(c):
            continue
        ranked.append((len(c.split()), len(c), c))
    if ranked:
        ranked.sort(reverse=True)
        return ranked[0][2]

    # Lowercase clinical head nouns after "for/of/in"
    m = re.search(
        r"\b(?:for|of|in|with)\s+([a-z][a-z0-9'’\-/]+(?:\s+[a-z0-9'’\-/]+){0,4})\b",
        search_in,
        flags=re.IGNORECASE,
    )
    if m:
        cand = _clean_entity(m.group(1))
        if cand and not _is_weak_entity(cand):
            return cand.title() if cand.islower() else cand

    return None


def _clean_entity(raw: str) -> str:
    v = (raw or "").strip(" ?:.,;\"'()[]")
    v = re.sub(r"\s+", " ", v)
    if len(v) < 3 or len(v) > 80:
        return ""
    if is_option_title_lock(v) or is_meta_option_value(v):
        return ""
    return v


def is_plausible_medical_bridge_value(
    value: Any,
    entity: Any = None,
    question: Any = None,
    *,
    base_fn: Optional[Callable[..., bool]] = None,
) -> bool:
    """Hotpot plausibility + MedMCQA bans (option titles / meta-options / chrome)."""
    v = str(value or "").strip()
    if not v:
        return False
    if is_invalid_medmcqa_bridge_lock(v):
        return False
    if base_fn is not None:
        try:
            return bool(base_fn(value, entity, question))
        except TypeError:
            return bool(base_fn(value))
    return len(v) >= 2


# ---------------------------------------------------------------------------
# Span mining (hop-1)
# ---------------------------------------------------------------------------

def mine_medical_hop1_bridge_spans(
    question: str,
    entity: Any,
    paragraphs: Iterable[Any],
    max_n: int = 36,
    *,
    base_fn: Optional[Callable[..., list]] = None,
) -> list:
    """Mine hop-1 candidates; drop option-title spans; boost stem concepts."""
    max_n = int(max_n or 36)
    out: list = []
    seen = set()

    def _push(value: str, slot: str, para_id: Any, text: str, pri: float) -> None:
        raw = str(value or "").strip()
        # Keep letter-prefixed values invalid (do not strip into a fake "concept")
        if is_invalid_medmcqa_bridge_lock(raw):
            return
        v = strip_option_letter_prefix(raw)
        if not v or v.lower() in seen or is_invalid_medmcqa_bridge_lock(v):
            return
        slot_s = str(slot or "")
        if (
            is_option_title_lock(slot_s)
            or slot_s.lower().startswith("title(option")
            or slot_s.lower().startswith("title(explanation")
            or "explanation" in slot_s.lower()[:22]
        ):
            return
        seen.add(v.lower())
        out.append((v, slot, para_id, text, pri))

    # 1) Concept from stem / entity
    stem = medmcqa_question_stem(question or "")
    ent = extract_medical_bridge_entity(question or "") or (
        str(entity).strip() if entity else ""
    )
    if ent and not is_option_title_lock(ent):
        _push(ent, f"med_concept({ent[:40]})", None, f"Medical concept: {ent}", 100.0)

    # Extra capitalized phrases from stem
    for m in re.finditer(
        r"\b([A-Z][A-Za-z0-9'’\-/]*(?:\s+[a-z]{2,12}){0,2}(?:\s+[A-Z][A-Za-z0-9'’\-/]*){0,3})\b",
        stem,
    ):
        _push(m.group(1), f"stem_span({m.group(1)[:40]})", None, f"Stem: {m.group(1)}", 80.0)

    # 2) Base Hotpot miner, then filter
    if base_fn is not None:
        try:
            base_spans = base_fn(question, entity, paragraphs, max_n=max_n) or []
        except TypeError:
            base_spans = base_fn(question, entity, paragraphs) or []
        for item in base_spans:
            if not item:
                continue
            if len(item) >= 5:
                value, slot, para_id, text, pri = item[:5]
            elif len(item) == 4:
                value, slot, para_id, text = item
                pri = 1.0
            else:
                continue
            slot_s = str(slot or "")
            # Drop option / explanation title mining
            if is_invalid_medmcqa_bridge_lock(value):
                continue
            if slot_s.lower().startswith("title("):
                inner = slot_s[6:].rstrip(")")
                if is_invalid_medmcqa_bridge_lock(inner) or is_explanation_title_lock(
                    inner
                ):
                    continue
            if slot_s.lower().startswith("title(option") or slot_s.lower().startswith(
                "title(explanation"
            ):
                continue
            _push(
                str(value),
                slot_s or f"bridge_span({value})",
                para_id,
                str(text or ""),
                float(pri or 1.0),
            )

    # 3) Option *texts* as weak evidence spans (not locks of "Option X" / "A. …")
    for para in paragraphs or []:
        title = str(getattr(para, "title", "") or "")
        text = str(getattr(para, "text", "") or "").strip()
        pid = getattr(para, "para_id", None)
        if not text:
            continue
        if (
            is_option_title_lock(title)
            or is_explanation_title_lock(title)
            or str(pid or "").startswith("opt_")
        ):
            # Keep option text as a candidate value only if it looks clinical
            if not is_invalid_medmcqa_bridge_lock(text) and len(text.split()) <= 12:
                _push(text, f"option_text({pid})", pid, text, 25.0)
        elif str(pid or "").startswith("exp::") or title.lower().startswith("explanation"):
            # Short noun phrases from explanation sentences — never lock title "Explanation"
            for m in re.finditer(
                r"\b([A-Z][A-Za-z0-9'’\-/]*(?:\s+[A-Za-z0-9'’\-/]+){0,4})\b",
                text,
            ):
                span = m.group(1)
                if is_explanation_title_lock(span):
                    continue
                _push(span, f"exp_span({span[:40]})", pid, text, 40.0)

    out.sort(key=lambda x: (-float(x[4]), -len(str(x[0]))))
    return out[:max_n]


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def make_load_medmcqa(ns: dict) -> Callable[..., list]:
    """Build a MedMCQA loader that uses medical-friendly paragraph titles."""

    Paragraph = ns["Paragraph"]
    HotpotExample = ns["HotpotExample"]
    cop_to_idx = ns["_medmcqa_cop_to_idx"]
    split_exp = ns["_split_explanation_paras"]
    load_dataset = ns["load_dataset"]

    def load_medmcqa(
        split: str = "validation",
        max_samples: Optional[int] = None,
        use_explanation_context: bool = True,
        answer_as: str = "letter",
    ):
        """Map MedMCQA → HotpotExample with medical retrieval corpus.

        Retrieval corpus = four options (title = ``A. <text>``) plus optional
        explanation sentences when ``use_explanation_context`` is true.
        """
        dataset_ids = ["openlifescienceai/medmcqa", "medmcqa"]
        last_err = None
        ds = None
        for dataset_id in dataset_ids:
            try:
                ds = load_dataset(dataset_id, split=split)
                print(f"Loaded dataset: {dataset_id} ({split})")
                break
            except Exception as e:  # noqa: BLE001 — mirror pipeline behavior
                last_err = e
        if ds is None:
            raise last_err  # type: ignore[misc]

        letters = "ABCD"
        opt_keys = ("opa", "opb", "opc", "opd")
        out: list = []
        skipped = 0

        for i, row in enumerate(ds):
            if max_samples is not None and len(out) >= max_samples:
                break
            choices = [str(row.get(k) or "").strip() for k in opt_keys]
            if sum(1 for c in choices if c) < 2:
                skipped += 1
                continue
            try:
                gold_idx = cop_to_idx(row.get("cop"))
            except Exception:
                skipped += 1
                continue
            if not (0 <= gold_idx < 4) or not choices[gold_idx]:
                skipped += 1
                continue

            gold_letter = letters[gold_idx]
            gold_text = choices[gold_idx]
            answer = gold_letter if answer_as == "letter" else gold_text

            paras = []
            for j in range(4):
                if not choices[j]:
                    continue
                letter = letters[j]
                # Title carries letter + clinical text (never bare "Option X")
                title = f"{letter}. {choices[j]}"
                paras.append(
                    Paragraph(
                        f"opt_{letter}",
                        title,
                        0,
                        choices[j],
                    )
                )

            if use_explanation_context:
                paras.extend(split_exp(row.get("exp") or ""))

            options_block = "\n".join(
                f"{letters[j]}. {choices[j]}" for j in range(4) if choices[j]
            )
            q_raw = str(row["question"]).strip()
            question = (
                f"{q_raw}\n\n{options_block}\n\n"
                "Answer with a single letter (A, B, C, or D)."
            )
            subject = str(row.get("subject_name") or "unknown")
            topic = str(row.get("topic_name") or "")
            level = f"{subject}/{topic}" if topic else subject
            ex_id = str(row.get("id") or i)

            # Match pipeline HotpotExample(example_id, question, answer,
            # question_type, level, paragraphs, supporting_para_ids)
            out.append(
                HotpotExample(
                    ex_id,
                    question,
                    answer,
                    "bridge",
                    level,
                    paras,
                    [],
                )
            )

        print(
            f"MedMCQA loaded={len(out)} skipped={skipped} answer_as={answer_as} "
            f"use_explanation_context={bool(use_explanation_context)}"
        )
        return out

    return load_medmcqa


# ---------------------------------------------------------------------------
# Soft lock → multihop fallback
# ---------------------------------------------------------------------------

def wrap_phase_b1_to_lock(ns: dict) -> Callable:
    """If lifecycle would lock chrome/options (or fail grounding), use S2 multihop."""

    original = ns["_phase_b1_to_lock"]
    multihop = ns["_phase_b1_multihop_nolife"]

    def _phase_b1_to_lock(example, config, ab_cfg, llm_cache):
        use_life = bool(getattr(ab_cfg, "use_lifecycle", True))
        if not use_life:
            return original(example, config, ab_cfg, llm_cache)

        soft = bool(ns.get("MEDMCQA_SOFT_LOCK_FALLBACK", True))
        ban = bool(ns.get("MEDMCQA_BAN_OPTION_TITLE_LOCKS", True))
        result = original(example, config, ab_cfg, llm_cache)
        if not soft and not ban:
            return result

        status = (result or {}).get("status")
        locked = (result or {}).get("locked")
        value = getattr(locked, "value", None) if locked is not None else None

        bad_lock = ban and is_invalid_medmcqa_bridge_lock(value)
        failed = status in {"fallback", "hop1_not_grounded", "bridge_resolution_failed"} or (
            status != "ok" and locked is None
        )

        if bad_lock or (soft and failed):
            fb = multihop(example, config, ab_cfg, llm_cache)
            try:
                tr = fb.get("trace") if isinstance(fb, dict) else None
                if isinstance(tr, list):
                    tr.append(
                        {
                            "medmcqa_soft_lock_fallback": True,
                            "rejected_locked_value": value,
                            "prior_status": status,
                            "reason": (
                                "invalid_medical_bridge_lock"
                                if bad_lock
                                else "no_valid_medical_lock"
                            ),
                        }
                    )
            except Exception:
                pass
            return fb
        return result

    return _phase_b1_to_lock


def wrap_split_explanation_paras(ns: dict) -> Callable:
    """Avoid locking the literal title 'Explanation'; use a text snippet title."""

    Paragraph = ns["Paragraph"]
    base = ns["_split_explanation_paras"]

    def _split_explanation_paras(exp, max_sents=6):
        paras = base(exp, max_sents) if max_sents is not None else base(exp)
        out = []
        for p in paras or []:
            text = str(getattr(p, "text", "") or "").strip()
            snippet = re.sub(r"\s+", " ", text)[:48].rstrip() or "exp"
            out.append(
                Paragraph(
                    getattr(p, "para_id", "exp::0"),
                    snippet,
                    int(getattr(p, "sent_idx", 0) or 0),
                    text,
                )
            )
        return out

    return _split_explanation_paras


def _sync_patches_into_live_modules(ns: dict, keys: list[str]) -> list[str]:
    """Pipeline bytecode uses LOAD_GLOBAL on the exec'd module — patch that dict too.

    Earlier smoke showed Option-title rate dropped only because ``load_medmcqa`` is
    called via ``ns[...]``; plausibility / soft-fallback patches on ``ns`` alone did
    not affect in-module globals.
    """
    import sys

    synced = []
    names = []
    for n in (
        ns.get("__name__"),
        "flow_rag_pipeline",
        "flow_rag.pipeline_export",
        "flow_rag_pipeline_export",
    ):
        if n and n not in names:
            names.append(n)
    payload = {k: ns[k] for k in keys if k in ns}
    # Always sync domain flags used by wrappers
    for flag in (
        "MEDMCQA_DOMAIN",
        "MEDMCQA_BAN_OPTION_TITLE_LOCKS",
        "MEDMCQA_SOFT_LOCK_FALLBACK",
        "MEDMCQA_USE_EXPLANATION_CONTEXT",
        "USE_HOP1_TITLE_GROUND_BYPASS",
        "DATASET",
        "is_option_title_lock",
        "is_meta_option_value",
        "is_explanation_title_lock",
        "is_invalid_medmcqa_bridge_lock",
        "is_plausible_medical_bridge_value",
        "extract_medical_bridge_entity",
        "medmcqa_question_stem",
    ):
        if flag in ns:
            payload[flag] = ns[flag]
    for name in names:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        d = getattr(mod, "__dict__", None)
        if d is None or d is ns:
            continue
        d.update(payload)
        synced.append(name)
    return synced


def wrap_extract_bridge_entity(ns: dict) -> Callable:
    base = ns["extract_bridge_entity"]

    def extract_bridge_entity(question: str):
        if ns.get("DATASET") == "medmcqa" or ns.get("MEDMCQA_DOMAIN", False):
            med = extract_medical_bridge_entity(question)
            if med:
                return med
        return base(question)

    return extract_bridge_entity


# ---------------------------------------------------------------------------
# Patch entrypoint
# ---------------------------------------------------------------------------

def apply_medmcqa_domain_patches(ns: dict) -> list[str]:
    """Monkey-patch pipeline namespace for medical MCQ. Returns patch names."""
    applied: list[str] = []
    ns["MEDMCQA_DOMAIN"] = True
    ns.setdefault("MEDMCQA_BAN_OPTION_TITLE_LOCKS", True)
    ns.setdefault("MEDMCQA_SOFT_LOCK_FALLBACK", True)
    # Title bypass is Hotpot-specific; keep off for MedMCQA
    ns["USE_HOP1_TITLE_GROUND_BYPASS"] = False

    # Helpers exposed for tests / audits
    ns["is_option_title_lock"] = is_option_title_lock
    ns["is_meta_option_value"] = is_meta_option_value
    ns["is_explanation_title_lock"] = is_explanation_title_lock
    ns["is_invalid_medmcqa_bridge_lock"] = is_invalid_medmcqa_bridge_lock
    ns["medmcqa_question_stem"] = medmcqa_question_stem
    ns["extract_medical_bridge_entity"] = extract_medical_bridge_entity
    ns["is_plausible_medical_bridge_value"] = is_plausible_medical_bridge_value

    if "is_plausible_bridge_value" in ns:
        base_plausible = ns["is_plausible_bridge_value"]

        def is_plausible_bridge_value(value, entity=None, question=None):
            return is_plausible_medical_bridge_value(
                value, entity, question, base_fn=base_plausible
            )

        ns["is_plausible_bridge_value"] = is_plausible_bridge_value
        applied.append("is_plausible_bridge_value")

    if "mine_wide_hop1_bridge_spans" in ns:
        base_mine = ns["mine_wide_hop1_bridge_spans"]

        def mine_wide_hop1_bridge_spans(question, entity, paragraphs, max_n=36):
            return mine_medical_hop1_bridge_spans(
                question, entity, paragraphs, max_n=max_n, base_fn=base_mine
            )

        ns["mine_wide_hop1_bridge_spans"] = mine_wide_hop1_bridge_spans
        applied.append("mine_wide_hop1_bridge_spans")

    if "extract_bridge_entity" in ns:
        ns["extract_bridge_entity"] = wrap_extract_bridge_entity(ns)
        applied.append("extract_bridge_entity")

    if "_split_explanation_paras" in ns and "Paragraph" in ns:
        ns["_split_explanation_paras"] = wrap_split_explanation_paras(ns)
        applied.append("_split_explanation_paras")

    if all(
        k in ns
        for k in (
            "Paragraph",
            "HotpotExample",
            "_medmcqa_cop_to_idx",
            "_split_explanation_paras",
            "load_dataset",
        )
    ):
        ns["load_medmcqa"] = make_load_medmcqa(ns)
        applied.append("load_medmcqa")

    if "_phase_b1_to_lock" in ns and "_phase_b1_multihop_nolife" in ns:
        ns["_phase_b1_to_lock"] = wrap_phase_b1_to_lock(ns)
        applied.append("_phase_b1_to_lock")

    synced = _sync_patches_into_live_modules(ns, applied)
    print(
        "[medmcqa_domain] patches:",
        ", ".join(applied) or "(none)",
        f"| synced_modules={synced or ['(ns-only)']} "
        f"| ban_option_locks={ns.get('MEDMCQA_BAN_OPTION_TITLE_LOCKS')} "
        f"soft_fallback={ns.get('MEDMCQA_SOFT_LOCK_FALLBACK')} "
        f"explain_ctx={ns.get('MEDMCQA_USE_EXPLANATION_CONTEXT')}",
    )
    return applied
