# 2026-11-06 Medical-domain MedMCQA adaptations

## Problem

Full-val (n=4183) best system was **S2_multihop_modularqa EM=0.437**. Lifecycle (S3+) ~0.39 because ~75% of S3 locks were bare `Option A–D` titles (Hotpot-style title bridging on MCQ option paras).

## Implemented (`flow_rag/medmcqa_domain.py`)

1. **Ban option-title locks** — rejects `Option X` / `A. …` / bare letters / meta-options / `Explanation` / numeric-only / question-stem locks.
2. **Medical hop-1 mining** — stem concepts + explanation spans; drop `title(Option *)` / explanation titles.
3. **Medical entity extract** — MCQ stem concepts (e.g. Meckel's cartilage) instead of Hotpot film/director cues.
4. **Loader** — option titles become `A. <text>` (not `Option A`); **explanation context ON by default**; explanation para titles are text snippets.
5. **Soft lock fallback** — invalid medical lock or failed grounding → S2-style `_phase_b1_multihop_nolife`.
6. **Module sync** — patches are written into the exec'd `flow_rag_pipeline` module globals (required for bytecode `LOAD_GLOBAL`).

Wired automatically for `--dataset medmcqa` unless `--no-medmcqa-domain-patches`.

### Smoke `smoke_medical_domain_0_20_bf16` (pre-sync fix)

- EM: naive/S1 0.50, S2–S5 0.40, full 0.35 (n=20 noise)
- Option-title locks: **0%** (loader change worked)
- Remaining bad locks: `Explanation`, `A./D. …` — filters had not synced into module globals; fixed in follow-up patch.

## Run (lab Python 3.9 + GPU)

```bash
MODEL_PATH=$HF_HOME/models--google--gemma-3-4b-it/snapshots/<snapshot>

# Smoke
python run_experiment.py --config configs/smoke_medical_domain.json \
  --model-path "$MODEL_PATH" --llm-batch-size 1 --no-load-in-4bit

# Primary comparison shard
python run_experiment.py --config configs/val_medical_domain_0_1000.json \
  --model-path "$MODEL_PATH" --llm-batch-size 1 --no-load-in-4bit

# Lock-rate audit
python scripts/analyze_option_locks.py \
  --details runs/final_val_medmcqa/medmcqa_val_medical_domain_0_1000_bf16/ablation_details.json
python scripts/analyze_s2_vs_s3.py  # point at new run if needed
```

## Success criteria (n=1000 slice)

| Check | Target |
|--|--|
| Option-title lock rate (S3) | ≪ 75% (aim &lt; 5% hard locks; rest soft-fallback) |
| S3 EM vs old S3 (~0.39) | improve |
| S3 EM vs S2 | close the −4.3 gap; ideally S3 ≥ S2 |
| verified_llm | stay ~high |

## Still later (not in this patch)

- External medical corpus (PubMed / textbooks) beyond MedMCQA `exp`
- MedMCQA-trained bridge reranker (keep Hotpot RR **off**)
- Stronger / medical LLM (MedGemma, larger instruct)
- Redesign hop-2 to score options against locked *concept* explicitly
