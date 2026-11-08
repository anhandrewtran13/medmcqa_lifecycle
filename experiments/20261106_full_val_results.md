# 2026-11-06 MedMCQA full validation (bf16 Gemma-3-4B)

## Setup

- Model: `google/gemma-3-4b-it` local snapshot, **bf16** (`load_in_4bit=false`)
- Split: MedMCQA `validation`, n=**4183** (4 shards)
- Merged: `runs/final_val_medmcqa/medmcqa_val_merged_bf16/ablation_df.csv`
- Analysis: `experiments/s2_vs_s3_analysis.json`

## Main table (micro-averaged EM = letter accuracy)

| setting | EM | verified_llm |
|---------|-----|--------------|
| naive_rag | 0.422 | 0 |
| S1_multihop | 0.422 | 0 |
| **S2_multihop_modularqa** | **0.437** | 0 |
| S3_lifecycle | 0.394 | 0.993 |
| S4_lifecycle_rr | 0.390 | 0.993 |
| S5_lifecycle_rr_sf | 0.390 | 0.993 |
| full | 0.389 | 0.993 |

## Takeaways

1. **Best MedMCQA system here is S2**, not full lifecycle.
2. Lifecycle LLM path is healthy (~99% verified answers); the gap vs S2 is modeling/locking, not empty generations.
3. Hotpot-trained **bridge reranker** is roughly neutral/slightly negative on full val (S3→S4 net ≈ −0.36 EM). Prefer **bridge OFF** for MedMCQA reporting (`configs/ablation_no_reranker.json`, or S3 as primary lifecycle number).
4. SF path does not move EM when disabled/on this setup (S4≈S5).

## S2 vs S3 error analysis

Script: `scripts/analyze_s2_vs_s3.py` → `experiments/s2_vs_s3_analysis.json`

| | count |
|--|--|
| both right | 1209 |
| both wrong | 1917 |
| **S2 only correct** | **619** |
| S3 only correct | 438 |
| net Δ (S3−S2) | **−4.33 EM** |

Letter flips when S2 is right and S3 wrong are spread across A/B/C/D (no single-letter collapse).

Among the **619** S2-win / S3-lose cases:

- **484** lock `Option X` title-like values (options-as-paras grounder)
- Symbolic letter: empty 51 / wrong 432 / right 136 — even when symbolic is right, final pred often diverges

Root cause direction: hop title-bypass / option-title locking still pollutes MedMCQA graphs; free S2 multihop avoids that.

## Follow-ups

- Redesign hop lock / option grounding for MedMCQA (or ship **S2 as MedMCQA best system**).
- Report lifecycle as **S3** (bridge OFF): `configs/ablation_no_reranker.json`, `configs/val_s3_no_rr_0_1000.json`.
- Harden repo (git) — pipeline still bootstraps from `pipeline_export.cpython-39.pyc`.
- Optional: stronger LLM or MedMCQA-trained bridge if lifecycle must beat S2.
