"""Pretrained hop-1 subquestion provider for FLOW-RAG (Mode A).

Replaces Gemma empty-context hop-1 subquestions with a frozen decomposition model.
Downstream retrieval, claims, NLI, reranker, and answer generation are unchanged.

Backends
--------
- ``modularqa`` (recommended): ModularQA NextGen BART (HotpotQA+DROP trained).
  Model: https://ai2-public-datasets.s3.amazonaws.com/modularqa/nextgen_model.zip
- ``cache``: DecompRC released HotpotQA-dev predictions with ``queries``
  (official Drive file is currently unavailable).
- ``t5``: HuggingFace ``thenHung/question_decomposer_t5`` (generic; weaker HotpotQA fit).

Usage in a notebook::

    from decomrc_hop1 import get_pretrained_hop1_decomposer, ensure_modularqa_nextgen
    model_dir = ensure_modularqa_nextgen("/content/drive/MyDrive/FLOW_RAG_runs/modularqa")
    decomposer = get_pretrained_hop1_decomposer(backend="modularqa", model_name=model_dir)
    q1 = decomposer.hop1_query(question, example_id=example_id)
"""
from __future__ import annotations

import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

_GLOBAL: "PretrainedHop1Decomposer | None" = None

# Official DecompRC HotpotQA-dev final predictions (includes reasoning + queries).
DECOMPRC_PRED_DRIVE_ID = "19h5afipdv0nc9uPhfDCDSWadojQpe0I_"
DECOMPRC_PRED_FILENAME = "decomrc_hotpot_dev_predictions.json"

# ModularQA NextGen (Khot et al. 2021) — HotpotQA + DROP trained BART.
MODULARQA_NEXTGEN_URL = (
    "https://ai2-public-datasets.s3.amazonaws.com/modularqa/nextgen_model.zip"
)
MODULARQA_NEXTGEN_DIRNAME = "nextgen_model"

# ModularQA sequence markers (from modularqa.con_gen.constants).
_COMPQ_MARKER = " QC: "
_SIMPQ_MARKER = " QS: "
_MODEL_PREFIX_RE = re.compile(
    r"^\((?:squad|math|bool|if_then|squad_list)\)\s*",
    re.IGNORECASE,
)


def _clean_query(text: str) -> str:
    q = re.sub(r"\s+", " ", (text or "").strip())
    if not q:
        return q
    q = q.rstrip("!. ")
    if not q.endswith("?"):
        q += "?"
    return q


def _strip_modularqa_prefix(text: str) -> str:
    """Drop ``(squad)`` / ``(math)`` router prefixes from NextGen output."""
    q = (text or "").strip()
    q = _MODEL_PREFIX_RE.sub("", q).strip()
    # Stop at EOQ / next marker if model over-generates
    for stop in ("[EOQ]", " QI:", " QC:", " A:", " QS:"):
        if stop in q:
            q = q.split(stop, 1)[0].strip()
    return _clean_query(q)


def _find_model_dir(root: Path) -> Path | None:
    """Find a HF model directory (has config.json) under root."""
    if (root / "config.json").is_file():
        return root
    for p in sorted(root.rglob("config.json")):
        return p.parent
    return None


def ensure_modularqa_nextgen(
    dest_dir: str | Path,
    force_download: bool = False,
) -> str:
    """Download + unzip ModularQA NextGen BART if needed. Returns model directory."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_dir = dest_dir / MODULARQA_NEXTGEN_DIRNAME
    found = _find_model_dir(model_dir) if model_dir.exists() else None
    if found is not None and not force_download:
        print(f"[hop1-pretrained] ModularQA NextGen ready → {found}", flush=True)
        return str(found)

    # Also accept an already-extracted tree elsewhere under dest_dir
    if not force_download:
        found = _find_model_dir(dest_dir)
        if found is not None and "nextgen" in str(found).lower():
            print(f"[hop1-pretrained] found ModularQA NextGen → {found}", flush=True)
            return str(found)

    zip_path = dest_dir / "nextgen_model.zip"
    print(
        f"[hop1-pretrained] downloading ModularQA NextGen (~1.5GB) → {zip_path}\n"
        f"  source: {MODULARQA_NEXTGEN_URL}",
        flush=True,
    )
    tmp_zip = zip_path.with_suffix(".zip.partial")
    try:
        urllib.request.urlretrieve(MODULARQA_NEXTGEN_URL, str(tmp_zip))
        tmp_zip.replace(zip_path)
    except Exception as e:
        if tmp_zip.exists():
            tmp_zip.unlink()
        raise FileNotFoundError(
            "Failed to download ModularQA NextGen from AI2 S3.\n"
            f"  URL: {MODULARQA_NEXTGEN_URL}\n"
            f"  Manual: wget that URL, unzip into {dest_dir}/\n"
            f"Original error: {e}"
        ) from e

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hop1-pretrained] unzipping → {model_dir}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(model_dir)

    found = _find_model_dir(model_dir)
    if found is None:
        raise FileNotFoundError(
            f"Unzipped NextGen zip but found no config.json under {model_dir}"
        )
    print(f"[hop1-pretrained] ModularQA NextGen ready → {found}", flush=True)
    return str(found)


def _find_existing_decomrc_predictions(dest_dir: Path) -> Path | None:
    """Look for an already-uploaded DecompRC predictions JSON under common names."""
    names = [
        DECOMPRC_PRED_FILENAME,
        "dev_predictions.json",
        "decomrc_dev_predictions.json",
        "hotpot_dev_predictions.json",
        "final_predictions.json",
        "predictions.json",
    ]
    search_dirs = [dest_dir, dest_dir.parent, Path("/content/drive/MyDrive/FLOW_RAG_runs")]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for name in names:
            p = d / name
            if p.is_file():
                return p
        for p in sorted(d.glob("*.json")):
            try:
                raw = json.loads(p.read_text())
            except Exception:
                continue
            if isinstance(raw, dict) and raw:
                sample = next(iter(raw.values()))
                if isinstance(sample, dict) and ("queries" in sample or "reasoning" in sample):
                    return p
    return None


def ensure_decomrc_predictions(
    dest_dir: str | Path,
    filename: str = DECOMPRC_PRED_FILENAME,
    drive_id: str = DECOMPRC_PRED_DRIVE_ID,
    force_download: bool = False,
) -> str:
    """Ensure DecompRC prediction JSON exists locally.

    Prefer an already-uploaded Drive file. Auto-download via gdown is best-effort
    (Google often blocks programmatic Drive downloads; official file may be gone).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / filename
    if out.is_file() and not force_download:
        return str(out)

    existing = _find_existing_decomrc_predictions(dest_dir)
    if existing is not None and not force_download:
        print(f"[hop1-pretrained] found existing DecompRC preds → {existing}", flush=True)
        return str(existing)

    url = f"https://drive.google.com/uc?id={drive_id}"
    print(f"[hop1-pretrained] downloading DecompRC predictions → {out}", flush=True)
    try:
        import gdown
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown

    try:
        gdown.download(url, str(out), quiet=False, fuzzy=True)
    except Exception as e:
        raise FileNotFoundError(
            "Auto-download from Google Drive failed (Drive file may be deleted).\n"
            "Prefer PRETRAINED_HOP1_BACKEND='modularqa' instead.\n"
            "Manual DecompRC fix (if you have the JSON):\n"
            f"  upload to {out}\n"
            f"Original error: {e}"
        ) from e

    if not out.is_file():
        raise FileNotFoundError(
            f"Failed to download DecompRC predictions to {out}. "
            "Use backend='modularqa' instead."
        )
    return str(out)


class PretrainedHop1Decomposer:
    """Frozen hop-1 Q1 generator (Mode A: retrieval query only)."""

    def __init__(
        self,
        backend: str = "modularqa",
        model_name: str = "thenHung/question_decomposer_t5",
        cache_path: str | None = None,
        device: str | None = None,
        fallback_to_question: bool = False,
    ):
        self.backend = (backend or "modularqa").strip().lower()
        if self.backend == "nextgen":
            self.backend = "modularqa"
        self.model_name = model_name
        self.cache_path = cache_path
        self.device = device
        self.fallback_to_question = bool(fallback_to_question)
        self._cache_by_id: dict[str, str] = {}
        self._tokenizer = None
        self._model = None
        self._load()

    def _load(self) -> None:
        if self.backend == "cache":
            path = self.cache_path
            if not path or not Path(path).is_file():
                raise FileNotFoundError(
                    f"DECOMPRC_PREDICTIONS_PATH not found: {path!r}\n"
                    "Use PRETRAINED_HOP1_BACKEND='modularqa', or place DecompRC "
                    f"predictions at that path."
                )
            raw = json.loads(Path(path).read_text())
            self._cache_by_id = _parse_predictions_cache(raw)
            print(
                f"[hop1-pretrained] DecompRC cache loaded {len(self._cache_by_id)} ids "
                f"from {path}",
                flush=True,
            )
            return

        if self.backend == "modularqa":
            self._load_modularqa()
            return

        if self.backend != "t5":
            raise ValueError(
                f"Unknown PRETRAINED_HOP1 backend: {self.backend!r} "
                "(expected modularqa | cache | t5)"
            )

        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        dev = self.device
        if not dev:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        print(f"[hop1-pretrained] loading T5 decomposer {self.model_name} on {dev}...", flush=True)
        self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self._model.to(dev)
        self._model.eval()
        print("[hop1-pretrained] T5 decomposer ready", flush=True)

    def _load_modularqa(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_path = self.model_name
        if not model_path or not Path(model_path).exists():
            # Treat model_name as parent dir and auto-download
            parent = model_path or "/content/drive/MyDrive/FLOW_RAG_runs/modularqa"
            model_path = ensure_modularqa_nextgen(parent)
            self.model_name = model_path

        found = _find_model_dir(Path(model_path))
        if found is None:
            raise FileNotFoundError(
                f"ModularQA NextGen model not found at {model_path!r}. "
                "Run ensure_modularqa_nextgen(...) first."
            )
        model_path = str(found)
        self.model_name = model_path

        dev = self.device
        if not dev:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        print(
            f"[hop1-pretrained] loading ModularQA NextGen from {model_path} on {dev}...",
            flush=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self._model.to(dev)
        self._model.eval()
        print("[hop1-pretrained] ModularQA NextGen ready", flush=True)

    def coverage(self, example_ids: list[str]) -> dict[str, Any]:
        ids = [str(x) for x in example_ids]
        hit = [eid for eid in ids if eid in self._cache_by_id]
        miss = [eid for eid in ids if eid not in self._cache_by_id]
        return {
            "n": len(ids),
            "hit": len(hit),
            "miss": len(miss),
            "miss_ids": miss[:10],
            "pct": (100.0 * len(hit) / len(ids)) if ids else 0.0,
        }

    def hop1_query(self, question: str, example_id: str | None = None) -> str:
        q = (question or "").strip()
        if not q:
            return q
        if self.backend == "cache":
            if example_id and example_id in self._cache_by_id:
                return self._cache_by_id[example_id]
            if self.fallback_to_question:
                return q
            raise KeyError(f"No DecompRC hop-1 query for example_id={example_id!r}")
        if self.backend == "modularqa":
            return self._modularqa_hop1(q)
        return self._t5_hop1(q)

    def batch_hop1_queries(
        self,
        items: list[tuple[str, str]],
        batch_size: int = 8,
    ) -> dict[str, str]:
        """Map example_id -> hop-1 query for a list of (example_id, question)."""
        out: dict[str, str] = {}
        if self.backend == "cache":
            missing = []
            for eid, q in items:
                try:
                    out[eid] = self.hop1_query(q, example_id=eid)
                except KeyError:
                    missing.append(eid)
                    if self.fallback_to_question:
                        out[eid] = (q or "").strip()
            if missing and not self.fallback_to_question:
                raise KeyError(
                    f"DecompRC cache missing {len(missing)}/{len(items)} ids "
                    f"(e.g. {missing[:3]})"
                )
            if missing:
                print(
                    f"[hop1-pretrained] WARNING: {len(missing)} ids missing in DecompRC "
                    "cache; fell back to original question",
                    flush=True,
                )
            return out

        if self.backend == "modularqa":
            for i in range(0, len(items), max(1, int(batch_size))):
                chunk = items[i : i + batch_size]
                qs = [q for _, q in chunk]
                gens = self._modularqa_hop1_batch(qs)
                for (eid, _), q1 in zip(chunk, gens):
                    out[eid] = q1
            return out

        for i in range(0, len(items), max(1, int(batch_size))):
            chunk = items[i : i + batch_size]
            for eid, q in chunk:
                out[eid] = self._t5_hop1(q)
        return out

    def _modularqa_prompt(self, question: str) -> str:
        # First-hop generation: QC: <q> QS:
        return f"{_COMPQ_MARKER}{(question or '').strip()}{_SIMPQ_MARKER}"

    def _modularqa_hop1(self, question: str) -> str:
        return self._modularqa_hop1_batch([question])[0]

    def _modularqa_hop1_batch(self, questions: list[str]) -> list[str]:
        import torch

        assert self._tokenizer is not None and self._model is not None
        prompts = [self._modularqa_prompt(q) for q in questions]
        # Older ModularQA tokenizer may not define pad; fall back to eos
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        enc = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            outputs = self._model.generate(
                **enc,
                max_new_tokens=40,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
            )
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [_strip_modularqa_prefix(t) for t in decoded]

    def _t5_hop1(self, question: str) -> str:
        import torch

        assert self._tokenizer is not None and self._model is not None
        input_text = f"decompose question: {question}"
        input_ids = self._tokenizer(
            input_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        with torch.inference_mode():
            outputs = self._model.generate(
                input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        parts = [p.strip() for p in decoded.split(" [SEP] ") if p.strip()]
        q1 = parts[0] if parts else decoded.strip()
        return _clean_query(q1)


def _parse_predictions_cache(raw: Any) -> dict[str, str]:
    """Accept DecompRC-style dicts or id->record mappings."""
    out: dict[str, str] = {}
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        records = raw["data"]
    elif isinstance(raw, dict):
        if all(isinstance(v, dict) for v in raw.values()):
            for eid, rec in raw.items():
                q1 = _queries_to_q1(rec)
                if q1:
                    out[str(eid)] = q1
            return out
        records = [raw]
    elif isinstance(raw, list):
        records = raw
    else:
        records = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        eid = str(rec.get("_id") or rec.get("id") or rec.get("example_id") or "")
        if not eid:
            continue
        q1 = _queries_to_q1(rec)
        if q1:
            out[eid] = q1
    return out


def _queries_to_q1(rec: dict) -> str | None:
    queries = rec.get("queries")
    if queries is None:
        return None
    if isinstance(queries, (list, tuple)) and len(queries) >= 2:
        cand = queries[1]
        if isinstance(cand, str) and cand.strip() and cand.strip().lower() != "none":
            return _clean_query(cand)
        for item in queries:
            if isinstance(item, str) and item.strip() and item.strip().lower() != "none":
                return _clean_query(item)
    if isinstance(queries, str) and queries.strip() and queries.strip().lower() != "none":
        parts = re.split(r"\?\s+", queries.strip())
        if parts:
            return _clean_query(parts[0] + "?")
    subq = rec.get("sub_question") or rec.get("hop1_query") or rec.get("q1")
    if isinstance(subq, str) and subq.strip():
        return _clean_query(subq)
    return None


def get_pretrained_hop1_decomposer(
    backend: str | None = None,
    model_name: str | None = None,
    cache_path: str | None = None,
    device: str | None = None,
    force_reload: bool = False,
    fallback_to_question: bool = False,
) -> PretrainedHop1Decomposer:
    global _GLOBAL
    if _GLOBAL is not None and not force_reload:
        return _GLOBAL
    _GLOBAL = PretrainedHop1Decomposer(
        backend=backend or "modularqa",
        model_name=model_name
        or "/content/drive/MyDrive/FLOW_RAG_runs/modularqa",
        cache_path=cache_path,
        device=device,
        fallback_to_question=fallback_to_question,
    )
    return _GLOBAL
