# MedMCQA Lifecycle (Neurosymbolic)

FLOW-RAG / Lifecycle Model on **MedMCQA** (GPU server), config-driven.

## Layout

```
medmcqa_lifecycle/
  run_experiment.py          # main CLI
  final_eval_index.py        # shared eval sharding
  requirements.txt
  configs/                   # experiment presets
  flow_rag/                  # pipeline (see note on bytecode bootstrap)
  bridge_reranker/           # LightGBM bridge weights + infer.py
  scripts/                   # SLURM + analysis helpers
  experiments/               # notes / analyses
  runs/                      # outputs (gitignored)
```

## Setup

```bash
cd medmcqa_lifecycle
pip install -r requirements.txt
# Gemma-3 needs transformers>=4.50
export HF_HOME=/network/rit/lab/Lai_ReSecureAI/kiel/hf_cache   # lab
export HF_TOKEN=...   # optional if using --model-path snapshot
```

**Gemma-3:** use **bf16** (`load_in_4bit: false`). 4-bit / wrong model class → empty answers.

## Latest full-val result (2026-11-06)

n=4183 · best **S2_multihop_modularqa EM=0.437** · lifecycle ~0.39  
Details: `experiments/20261106_full_val_results.md`  
Merged CSV: `runs/final_val_medmcqa/medmcqa_val_merged_bf16/`

**MedMCQA reporting tip:** Hotpot-trained bridge does not help; use S3 / `configs/ablation_no_reranker.json` for lifecycle numbers.

**Medical-domain patches (default ON for MedMCQA):** ban `Option A–D` hop-1 locks, mine stem/explanation concepts, explanation context ON, soft-fallback to S2 multihop when no valid lock. See `flow_rag/medmcqa_domain.py` and `experiments/20261106_medical_domain_plan.md`.

## Run

```bash
MODEL_PATH=$HF_HOME/models--google--gemma-3-4b-it/snapshots/<snapshot>

# Smoke
python run_experiment.py --config configs/smoke.json \
  --model-path "$MODEL_PATH" --llm-batch-size 1 --no-load-in-4bit

# Medical-domain smoke + 0–1000 comparison (RR off, explanation context on)
python run_experiment.py --config configs/smoke_medical_domain.json \
  --model-path "$MODEL_PATH" --llm-batch-size 1 --no-load-in-4bit
python run_experiment.py --config configs/val_medical_domain_0_1000.json \
  --model-path "$MODEL_PATH" --llm-batch-size 1 --no-load-in-4bit
python scripts/analyze_option_locks.py \
  --details runs/final_val_medmcqa/medmcqa_val_medical_domain_0_1000_bf16/ablation_details.json

# Full val shards (SLURM)
bash scripts/submit_val_shards.sh
python scripts/merge_val_shards.py
python scripts/analyze_s2_vs_s3.py
python scripts/analyze_bridge_regression.py --run medmcqa_val_0_1000_bf16
```

## Pipeline source note

`flow_rag/pipeline_export.py` may be a **bytecode bootstrap** loading
`flow_rag/__pycache__/pipeline_export.cpython-39.pyc` (lab Python 3.9).
Prefer re-exporting from the parent FLOW-RAG notebook when available:

```bash
python tools/export_flow_rag_pipeline.py   # if exporter + notebook restored
```

## Defaults / caveats

| Setting | Default / note |
|--------|----------------|
| Dataset | MedMCQA `validation` (4183 labeled) |
| Metric | Letter accuracy A–D (EM) |
| Bridge | ON in val configs historically; **prefer OFF** for MedMCQA |
| SF labeler | OFF |
| ModularQA hop-1 | OFF for MedMCQA |
| Precision | bf16 for Gemma-3 |
