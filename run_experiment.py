#!/usr/bin/env python3
"""MedMCQA Lifecycle (FLOW-RAG) experiment runner for GPU servers.

Examples:
  export HF_TOKEN=...
  python run_experiment.py --config configs/smoke.json
  python run_experiment.py --config configs/val_0_1000.json --model-path /data/models/gemma
  python run_experiment.py --dataset medmcqa --index-start 0 --index-end 50 --load-in-4bit

CLI flags override values from --config. Results land under runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_config_file(path: Path) -> dict:
    raw = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "PyYAML required for .yaml configs: pip install pyyaml"
            ) from e
        data = yaml.safe_load(raw) or {}
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a JSON/YAML object: {path}")
    return data


def _apply_config_defaults(argv: list[str] | None) -> list[str]:
    """Pre-parse --config so file values become argparse defaults."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in argv:
        return argv
    i = argv.index("--config")
    if i + 1 >= len(argv):
        raise SystemExit("--config requires a path")
    cfg_path = Path(argv[i + 1])
    if not cfg_path.is_file():
        alt = REPO_ROOT / cfg_path
        if alt.is_file():
            cfg_path = alt
        else:
            raise SystemExit(f"Config not found: {argv[i + 1]}")
    cfg = _load_config_file(cfg_path)
    optional_bool = {
        "load-in-4bit",
        "use-bridge-reranker",
        "use-sf-labeler",
        "use-pretrained-hop1",
        "filter-bridge-only",
        "use-pipeline",
        "medmcqa-use-explanation-context",
        "medmcqa-ban-option-locks",
        "medmcqa-soft-lock-fallback",
        "medmcqa-domain-patches",
    }
    injected: list[str] = []
    skip = {"comment", "description", "notes"}
    for key, val in cfg.items():
        if key in skip or val is None:
            continue
        flag = "--" + str(key).replace("_", "-")
        name = flag[2:]
        if flag in argv or f"--no-{name}" in argv:
            continue
        if isinstance(val, bool):
            if name in optional_bool:
                injected.append(flag if val else f"--no-{name}")
            elif val:
                injected.append(flag)
        else:
            injected.extend([flag, str(val)])
    return injected + argv


def _default_rr_paths(bridge_dir: Path):
    """Prefer repo-shipped ModularQA LightGBM weights."""
    candidates = [
        bridge_dir / "models_modularqa",
        bridge_dir / "models_modularqa" / "scaling_500_1000_2529" / "n2529",
        bridge_dir / "models",
        bridge_dir / "models_v3_symbolic" / "scaling_500_1000_3000" / "n1000",
    ]
    for d in candidates:
        model = d / "bridge_reranker_lightgbm.txt"
        slot = d / "slot_prefix_mapping.json"
        report = d / "comparison_report.json"
        if model.is_file() and slot.is_file():
            return (model, slot, report if report.is_file() else report)
    d = candidates[0]
    return (
        d / "bridge_reranker_lightgbm.txt",
        d / "slot_prefix_mapping.json",
        d / "comparison_report.json",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = _apply_config_defaults(argv)
    rr_model, rr_slot, rr_report = _default_rr_paths(REPO_ROOT / "bridge_reranker")
    p = argparse.ArgumentParser(
        description="MedMCQA Lifecycle experiments on a GPU server (no Colab/Jupyter)."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON/YAML experiment config (CLI flags override file values)",
    )
    p.add_argument("--dataset", choices=["medmcqa", "hotpot"], default="medmcqa")
    p.add_argument("--split", default="validation", help="HF split (MedMCQA: validation)")
    p.add_argument("--subset", default=None, help="Hotpot subset (default: distractor)")
    p.add_argument("--index-start", type=int, default=0)
    p.add_argument("--index-end", type=int, default=200)
    p.add_argument("--index-rebuild", action="store_true", help="Rebuild eval_index.json")
    p.add_argument("--batch-size", type=int, default=100, help="Index batch metadata size")
    p.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    p.add_argument("--run-name", default=None, help="Output folder name under runs-dir")
    p.add_argument("--model", default="google/gemma-3-4b-it", help="HF model id")
    p.add_argument("--model-path", type=Path, default=None, help="Local HF model directory")
    p.add_argument("--llm-backend", choices=["local", "huggingface", "heuristic"], default="local")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "",
    )
    p.add_argument("--hf-model", default="google/gemma-4-31B-it")
    p.add_argument("--hf-provider", default="auto")
    p.add_argument("--bridge-reranker-dir", type=Path, default=REPO_ROOT / "bridge_reranker")
    p.add_argument("--bridge-reranker-model", type=Path, default=rr_model)
    p.add_argument("--bridge-reranker-slot-map", type=Path, default=rr_slot)
    p.add_argument("--bridge-reranker-report", type=Path, default=rr_report)
    p.add_argument("--use-bridge-reranker", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-sf-labeler", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sf-dir", type=Path, default=REPO_ROOT / "sf_labeler")
    p.add_argument(
        "--use-pretrained-hop1",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="ModularQA hop-1 (default: off for medmcqa, on for hotpot)",
    )
    p.add_argument("--modularqa-dir", type=Path, default=REPO_ROOT / "modularqa")
    p.add_argument(
        "--medmcqa-use-explanation-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include MedMCQA explanation sentences in the retrieval corpus (medical domain default: on)",
    )
    p.add_argument("--medmcqa-answer-as", choices=["letter", "text"], default="letter")
    p.add_argument(
        "--medmcqa-domain-patches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply medical-domain patches (ban Option-title locks, medical bridges, soft fallback)",
    )
    p.add_argument(
        "--medmcqa-ban-option-locks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject Option A–D / letter locks as hop-1 bridges",
    )
    p.add_argument(
        "--medmcqa-soft-lock-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If lifecycle cannot lock a medical concept, fall back to S2-style multihop",
    )
    p.add_argument("--filter-bridge-only", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--llm-batch-size", type=int, default=2)
    p.add_argument("--phase-b-llm-batch-size", type=int, default=1)
    p.add_argument("--cpu-workers", type=int, default=4)
    p.add_argument("--use-pipeline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke-test-llm", action="store_true")
    p.add_argument("--skip-ablation", action="store_true", help="Load data/models only")
    p.add_argument(
        "--export-pipeline",
        action="store_true",
        help="Re-export flow_rag/pipeline_export.py from the parent notebook, then exit",
    )
    return p.parse_args(argv)


def _ensure_pipeline_export() -> Path:
    export_path = REPO_ROOT / "flow_rag" / "pipeline_export.py"
    if not export_path.is_file():
        # Allow bytecode-only recovery bootstrap / missing source
        pyc = (
            REPO_ROOT
            / "flow_rag"
            / "__pycache__"
            / f"pipeline_export.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
        )
        if pyc.is_file() and not export_path.is_file():
            # bootstrap file will be written by recovery; if still missing, error
            pass
        if not export_path.is_file():
            raise SystemExit(
                f"Missing {export_path}; run: python tools/export_flow_rag_pipeline.py"
            )
    return export_path


def _build_namespace(args: argparse.Namespace) -> dict:
    runs = args.runs_dir.resolve()
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "phase_a_cache").mkdir(parents=True, exist_ok=True)

    dataset = args.dataset
    use_hop1 = (
        args.use_pretrained_hop1
        if args.use_pretrained_hop1 is not None
        else (dataset != "medmcqa")
    )
    filter_bridge = (
        args.filter_bridge_only
        if args.filter_bridge_only is not None
        else (dataset == "hotpot")
    )
    subset = args.subset or ("distractor" if dataset == "hotpot" else "default")
    load_from_local = bool(args.model_path)
    model_path = str(args.model_path.resolve()) if args.model_path else None

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", args.hf_token)

    run_name = args.run_name or (
        f"{dataset}_range_{args.index_start:04d}_{args.index_end:04d}_v5c_progressive"
    )
    final_runs = runs / ("final_val_medmcqa" if dataset == "medmcqa" else "final_val_bridge")
    final_runs.mkdir(parents=True, exist_ok=True)
    index_dir = runs / (
        "final_eval_index_medmcqa" if dataset == "medmcqa" else "final_eval_index"
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    shared_index = runs / (
        "eval_index_medmcqa.json" if dataset == "medmcqa" else "eval_index.json"
    )

    ns = {
        "__name__": "flow_rag_pipeline",
        "__file__": str(_ensure_pipeline_export()),
        "DATASET": dataset,
        "LLM_BACKEND": args.llm_backend,
        "LOCAL_MODEL": args.model,
        "LOAD_IN_4BIT": bool(args.load_in_4bit),
        "LOAD_FROM_DRIVE": load_from_local,
        "LOCAL_MODEL_PATH": model_path,
        "HF_MODEL": args.hf_model,
        "HF_PROVIDER": args.hf_provider,
        "HF_TOKEN": args.hf_token,
        "NLI_BATCH_SIZE": 32,
        "NLI_TOP_PARAS": 3,
        "NLI_MEMOIZE": True,
        "USE_PIPELINE": bool(args.use_pipeline),
        "HOP1_MAX_VALUE_LEN": 100,
        "HOP1_MAX_VALUE_WORDS": 16,
        "HOP1_MAX_KEEP": 6,
        "HOP1_LOCK_MARGIN": 0.0,
        "HOP1_LOCK_MIN_SCORE": 0.0,
        "BRIDGE_CANDIDATES": 3,
        "REPAIR_SCORE_THRESHOLD": 0.55,
        "REPAIR_MIN_IMPROVE": 0.05,
        "USE_WIDE_HOP2_CANDIDATES": True,
        "WIDE_HOP2_MAX_SPANS": 36,
        "HOP2_SOFT_PREFER": True,
        "HOP2_PHRASE_SUBJECT": True,
        "HOP2_EXPAND_EXTRACTORS": True,
        "USE_WIDE_HOP1_CANDIDATES": True,
        "WIDE_HOP1_MAX_SPANS": 36,
        "USE_HOP1_WIDE_RANK_POOL": True,
        "USE_HOP1_TITLE_GROUND_BYPASS": False if dataset == "medmcqa" else True,
        "HOP1_TITLE_BYPASS_MIN_SCORE": 0.2,
        "HOP1_WIDE_BYPASS_MIN_SCORE": 0.35,
        "MEDMCQA_DOMAIN": dataset == "medmcqa",
        "MEDMCQA_BAN_OPTION_TITLE_LOCKS": bool(
            getattr(args, "medmcqa_ban_option_locks", True)
        ),
        "MEDMCQA_SOFT_LOCK_FALLBACK": bool(
            getattr(args, "medmcqa_soft_lock_fallback", True)
        ),
        "MEDMCQA_DOMAIN_PATCHES": bool(
            getattr(args, "medmcqa_domain_patches", True)
        ),
        "USE_ANSWER_POSTPROCESS": True,
        "CPU_WORKERS": int(args.cpu_workers),
        "ABLATION_CACHE_LLM": True,
        "PHASE_A_CACHE_PERSIST": True,
        "PHASE_A_CACHE_DIR": str(runs / "phase_a_cache"),
        "LLM_BATCH_SIZE": int(args.llm_batch_size),
        "PHASE_B_LLM_BATCH_SIZE": int(args.phase_b_llm_batch_size),
        "USE_PRETRAINED_HOP1": bool(use_hop1),
        "PRETRAINED_HOP1_BACKEND": "modularqa",
        "MODULARQA_DIR": str(args.modularqa_dir.resolve()),
        "PRETRAINED_HOP1_MODEL": str(args.modularqa_dir.resolve()),
        "DECOMPRC_DIR": str((REPO_ROOT / "decomrc").resolve()),
        "DECOMPRC_PREDICTIONS_PATH": str(
            (REPO_ROOT / "decomrc" / "decomrc_hotpot_dev_predictions.json").resolve()
        ),
        "PRETRAINED_HOP1_BATCH_SIZE": 8,
        "PRETRAINED_HOP1_FALLBACK_TO_QUESTION": False,
        "CACHE_MAX_NEW_TOKENS": 128,
        "CACHE_SKIP_COT": False,
        "LLM_VERBOSE": False,
        "INCLUDE_NAIVE_RAG_BASELINE": True,
        "RUN_FLOW_RAG": False,
        "RUN_NAIVE_RAG": False,
        "ABLATION_SAVE_DIR": str(final_runs),
        "BRIDGE_RERANKER_DIR": str(args.bridge_reranker_dir.resolve()),
        "BRIDGE_RERANKER_BACKEND": "lightgbm",
        "BRIDGE_RERANKER_MODEL_PATH": str(args.bridge_reranker_model.resolve()),
        "BRIDGE_RERANKER_SLOT_MAP_PATH": str(args.bridge_reranker_slot_map.resolve()),
        "BRIDGE_RERANKER_REPORT_PATH": str(args.bridge_reranker_report.resolve()),
        "BRIDGE_RERANKER_TRAIN_JSONL": str(
            (args.bridge_reranker_dir / "bridge_train_data.jsonl").resolve()
        ),
        "USE_BRIDGE_RERANKER": bool(args.use_bridge_reranker),
        "EVAL_SPLIT": args.split,
        "EVAL_SUBSET": subset,
        "FILTER_BRIDGE_ONLY": bool(filter_bridge),
        "MEDMCQA_USE_EXPLANATION_CONTEXT": bool(args.medmcqa_use_explanation_context),
        "MEDMCQA_ANSWER_AS": args.medmcqa_answer_as,
        "USE_V21_ID_OVERRIDE": False,
        "v21_ids": [],
        "SHARED_INDEX_PATH": str(shared_index),
        "INDEX_START": int(args.index_start),
        "INDEX_END": int(args.index_end),
        "INDEX_REBUILD": bool(args.index_rebuild) or (not shared_index.is_file()),
        "FINAL_INDEX_DIR": str(index_dir),
        "FINAL_RUNS_ROOT": str(final_runs),
        "RUN_NAME": run_name,
        "ABLATION_RUN_NAME": run_name,
        "BATCH_SIZE": int(args.batch_size),
        "BATCH_ID": None,
        "SAVE_ABLATION_RESULTS": True,
        "LOAD_ABLATION_RESULTS": False,
        "REUSE_IN_MEMORY_ABLATION": False,
        "USE_SF_LABELER": bool(args.use_sf_labeler),
        "SF_MODE": "argmax",
        "SF_TOP_K": 5,
        "SF_ARGMAX_MARGIN": 0.0,
        "SF_LOCK_FREEZE": True,
        "SF_DIR": str(args.sf_dir.resolve()),
        "SF_LABELER_MODEL_PATH": str(
            (
                args.sf_dir / "models" / "scaling" / "n3500" / "sf_labeler_lightgbm.txt"
            ).resolve()
        ),
        "SF_LABELER_REPORT_PATH": str(
            (
                args.sf_dir / "models" / "scaling" / "n3500" / "comparison_report.json"
            ).resolve()
        ),
        "SF_LABELER_SLOT_MAP_PATH": str(
            (
                args.sf_dir / "models" / "scaling" / "n3500" / "slot_prefix_mapping.json"
            ).resolve()
        ),
        "SF_RETRACT_TAU": 0.25,
        "SF_RETRACT_MARGIN": 0.05,
        "METHODS": {},
    }
    return ns


def _exec_pipeline(ns: dict) -> None:
    """Exec pipeline_export into ns, registered in sys.modules.

    Python 3.9 dataclasses look up sys.modules[cls.__module__] while processing
    fields; a bare exec() never registers that name and raises AttributeError.
    """
    path = Path(ns["__file__"])
    code = path.read_text()
    mod_name = ns.get("__name__", "flow_rag_pipeline")
    module = types.ModuleType(mod_name)
    module.__dict__.update(ns)
    sys.modules[mod_name] = module
    exec(compile(code, str(path), "exec"), module.__dict__)
    ns.update(module.__dict__)


def _load_llm(ns: dict, smoke: bool = False) -> None:
    backend = ns["LLM_BACKEND"]
    if backend != "local":
        print(f"LLM load skipped for backend={backend}")
        return
    LocalHuggingFaceLLM = ns["LocalHuggingFaceLLM"]
    local_path = ns.get("LOCAL_MODEL_PATH")
    token = ns.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    if local_path:
        if not Path(local_path).exists():
            raise FileNotFoundError(f"--model-path not found: {local_path}")
        LocalHuggingFaceLLM(
            ns["LOCAL_MODEL"],
            token=token,
            load_in_4bit=ns["LOAD_IN_4BIT"],
            local_path=local_path,
        )
        print(f"Loaded local model from {local_path}")
    else:
        if not token and "gemma" in str(ns["LOCAL_MODEL"]).lower():
            print(
                "Warning: HF_TOKEN not set; gated models (Gemma) usually require export HF_TOKEN=..."
            )
        LocalHuggingFaceLLM(
            ns["LOCAL_MODEL"],
            token=token,
            load_in_4bit=ns["LOAD_IN_4BIT"],
        )
        print(f"Loaded Hub model {ns['LOCAL_MODEL']}")

    if smoke:
        llm = LocalHuggingFaceLLM(
            ns["LOCAL_MODEL"],
            token=token,
            load_in_4bit=ns["LOAD_IN_4BIT"],
            local_path=local_path,
        )
        reply = llm.answer("What is 2+2?", "Math fact: 2+2 equals 4.")
        print("Smoke-test reply:", repr(reply))
        text = (reply or "").strip()
        if (not text) or text.lower() == "unknown":
            raise SystemExit(
                "Smoke-test LLM reply was empty/unknown — fix generation before ablation "
                "(see LocalHuggingFaceLLM._chat_batch truncation/decode)."
            )


def _load_examples(ns: dict):
    from final_eval_index import (
        build_or_load_eval_index,
        list_completed_shards,
        load_range_examples,
    )

    dataset = ns["DATASET"]
    if dataset == "medmcqa":
        print(f"Loading MedMCQA {ns['EVAL_SPLIT']}...")
        all_examples = ns["load_medmcqa"](
            split=ns["EVAL_SPLIT"],
            max_samples=None,
            use_explanation_context=ns["MEDMCQA_USE_EXPLANATION_CONTEXT"],
            answer_as=ns["MEDMCQA_ANSWER_AS"],
        )
        pool = list(all_examples)
    else:
        print(f"Loading HotpotQA {ns['EVAL_SPLIT']}/{ns['EVAL_SUBSET']}...")
        all_examples = ns["load_hotpot"](
            split=ns["EVAL_SPLIT"],
            subset=ns["EVAL_SUBSET"],
            max_samples=None,
        )
        if ns["FILTER_BRIDGE_ONLY"]:
            pool = [ex for ex in all_examples if ex.question_type == "bridge"]
            print(f"Bridge-only pool: {len(pool)} / {len(all_examples)}")
        else:
            pool = list(all_examples)

    ExperimentConfig = ns["ExperimentConfig"]
    config = ExperimentConfig(
        top_k_retrieve=5,
        max_hops=2,
        llm_backend=ns["LLM_BACKEND"],
        llm_model=(
            ns["LOCAL_MODEL"] if ns["LLM_BACKEND"] == "local" else ns["HF_MODEL"]
        ),
        hf_provider=ns["HF_PROVIDER"],
        load_in_4bit=ns["LOAD_IN_4BIT"],
        local_model_path=(
            ns["LOCAL_MODEL_PATH"] if ns["LOAD_FROM_DRIVE"] else None
        ),
        flow_rag_llm_fallback=True,
        flow_rag_verified_llm_answer=True,
        use_llm_claims=True,
        use_nli=True,
        nli_entailment_threshold=0.55,
        nli_weak_value_fallback=False,
        use_hybrid_retrieval=True,
        vericot_llm_final_answer=True,
    )
    ns["config"] = config

    eval_index = build_or_load_eval_index(
        pool,
        index_dir=ns["FINAL_INDEX_DIR"],
        split=ns["EVAL_SPLIT"],
        subset=ns["EVAL_SUBSET"],
        filter_bridge_only=ns["FILTER_BRIDGE_ONLY"],
        batch_size=ns["BATCH_SIZE"],
        rebuild=ns["INDEX_REBUILD"],
        shared_index_path=ns["SHARED_INDEX_PATH"],
    )
    ns["eval_index"] = eval_index
    examples = load_range_examples(
        pool, eval_index, ns["INDEX_START"], ns["INDEX_END"]
    )
    ns["examples"] = examples
    ns["ABLATION_N"] = len(examples)
    print(
        f"RANGE [{ns['INDEX_START']}, {ns['INDEX_END']}) → {len(examples)} examples "
        f"| fingerprint={eval_index.get('fingerprint')}"
    )
    print("IDs (first 5):", [ex.example_id for ex in examples[:5]])
    completed = list_completed_shards(ns["FINAL_RUNS_ROOT"])
    if ns["ABLATION_RUN_NAME"] in completed:
        print(
            f"WARNING: {ns['ABLATION_RUN_NAME']} already has results under {ns['FINAL_RUNS_ROOT']}"
        )
    return examples


def _save_ablation(ns: dict, ablation_df, ablation_details) -> None:
    save_dir = Path(ns["ABLATION_SAVE_DIR"])
    run_name = ns["ABLATION_RUN_NAME"]
    out = save_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(out / "ablation_df.csv", index=False)
    details_json = {}
    for setting, results in ablation_details.items():
        details_json[setting] = [
            r if isinstance(r, dict) else getattr(r, "__dict__", r) for r in results
        ]
        # Prefer to_dict if present
        serialized = []
        for r in results:
            if hasattr(r, "to_dict"):
                serialized.append(r.to_dict())
            elif isinstance(r, dict):
                serialized.append(r)
            else:
                serialized.append(
                    {
                        "example_id": getattr(r, "example_id", None),
                        "question": getattr(r, "question", None),
                        "gold_answer": getattr(r, "gold_answer", getattr(r, "answer", None)),
                        "pred_answer": getattr(r, "pred_answer", getattr(r, "prediction", None)),
                        "method": getattr(r, "method", setting),
                        "hops": getattr(r, "hops", None),
                        "graph": getattr(r, "graph", None),
                        "trace": getattr(r, "trace", None),
                        "metadata": getattr(r, "metadata", None),
                    }
                )
        details_json[setting] = serialized
    (out / "ablation_details.json").write_text(
        json.dumps(details_json, indent=2, default=str) + "\n"
    )
    meta = {
        "project": "medmcqa_lifecycle",
        "run_name": run_name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": ns["DATASET"],
        "index_start": ns["INDEX_START"],
        "index_end": ns["INDEX_END"],
        "n_examples": ns.get("ABLATION_N"),
        "settings": list(ablation_details.keys()),
        "llm_backend": ns["LLM_BACKEND"],
        "model": ns["LOCAL_MODEL"],
        "fingerprint": (ns.get("eval_index") or {}).get("fingerprint"),
        "config": ns.get("_CONFIG_PATH", ""),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved ablation run → {out}")


def _run_ablation(ns: dict, examples) -> None:
    config = ns["config"]
    n = len(examples)
    if ns["USE_PIPELINE"] and ns["LLM_BACKEND"] == "local":
        print(
            f"Pipelined ablation | cpu_workers={ns['CPU_WORKERS']} "
            f"llm_batch={ns['LLM_BATCH_SIZE']} n={n}",
            flush=True,
        )
        ablation_df, ablation_details = ns["run_ablation_suite_pipelined"](
            examples, config, max_examples=n, cpu_workers=ns["CPU_WORKERS"]
        )
    else:
        ablation_df, ablation_details = ns["run_ablation_suite"](
            examples, config, max_examples=n
        )
    ns["ablation_df"] = ablation_df
    ns["ablation_details"] = ablation_details
    print("Ablation EM / relaxed_EM / F1 by setting:")
    try:
        print(ablation_df.set_index("setting")[["EM", "relaxed_EM", "F1"]])
    except Exception:
        print(ablation_df)
    print("\nProgressive Δ EM:")
    order = (
        "naive_rag",
        "S1_multihop",
        "S2_multihop_modularqa",
        "S3_lifecycle",
        "S4_lifecycle_rr",
        "S5_lifecycle_rr_sf",
        "full",
    )
    try:
        em = {r["setting"]: float(r["EM"]) for _, r in ablation_df.iterrows()}
        prev = None
        for k in order:
            if k not in em:
                continue
            if prev is None:
                print(f"  {k:28s} EM={em[k]:.4f}")
            else:
                print(f"  {k:28s} EM={em[k]:.4f}  Δ={em[k] - prev:+.4f}")
            prev = em[k]
    except Exception as e:
        print("[progressive deltas skipped]", e)
    if ns["SAVE_ABLATION_RESULTS"]:
        _save_ablation(ns, ablation_df, ablation_details)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.export_pipeline:
        from tools.export_flow_rag_pipeline import main as export_main

        export_main()
        return 0

    print(f"Repo root: {REPO_ROOT}")
    print(
        f"Config: dataset={args.dataset} range=[{args.index_start},{args.index_end}) "
        f"backend={args.llm_backend} model={args.model_path or args.model}"
    )
    ns = _build_namespace(args)
    ns["_CONFIG_PATH"] = str(args.config.resolve()) if args.config else ""
    print("Loading pipeline export (definitions + optional component loaders)...")
    _exec_pipeline(ns)
    if args.dataset == "medmcqa" and bool(args.medmcqa_domain_patches):
        from flow_rag.medmcqa_domain import apply_medmcqa_domain_patches

        # Keep ns flags authoritative after pipeline defaults load
        ns["MEDMCQA_BAN_OPTION_TITLE_LOCKS"] = bool(args.medmcqa_ban_option_locks)
        ns["MEDMCQA_SOFT_LOCK_FALLBACK"] = bool(args.medmcqa_soft_lock_fallback)
        ns["MEDMCQA_USE_EXPLANATION_CONTEXT"] = bool(
            args.medmcqa_use_explanation_context
        )
        apply_medmcqa_domain_patches(ns)
    elif args.dataset == "medmcqa":
        print("[medmcqa_domain] patches disabled (--no-medmcqa-domain-patches)")
    _load_llm(ns, smoke=args.smoke_test_llm)
    examples = _load_examples(ns)
    if args.skip_ablation:
        print(f"Skip ablation (--skip-ablation). Loaded {len(examples)} examples.")
        return 0
    if not examples:
        print("No examples in range — nothing to run.")
        return 1
    _run_ablation(ns, examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
