# Question Answering Lifecycles on MedMCQA
## Extension Report — Medical Domain Experiments

**Date:** 2026-11-08  
**Base system:** FLOW-RAG / Lifecycle Model (neurosymbolic multi-hop QA)  
**Companion report:** *Final Report — Question Answering Lifecycles* (HotpotQA bridge1000)  
**Primary LLM:** `google/gemma-3-4b-it` (bf16)

---

## 1. Abstract

The Lifecycle Model was originally validated on HotpotQA bridge questions, where grounding and locking Wikipedia entity bridges improved exact match (EM) from 0.381 (naive RAG) to **0.433**, matching VeriCoT with a lighter, more auditable architecture. This report extends that system to **MedMCQA**, a medical multiple-choice benchmark.

On the full MedMCQA validation set (**n = 4183**), the best configuration is **not** the full lifecycle: **S2_multihop_modularqa** reaches **EM = 0.437**, while lifecycle settings fall to ~**0.39**. Error analysis shows a domain mismatch: ~75% of lifecycle locks were bare `Option A–D` titles—Hotpot-style title bridging applied to MCQ option paragraphs.

Medical-domain patches (ban option-title locks, mine stem/explanation concepts, enable explanation context, soft-fallback to multihop) eliminate option-title locks (**0.0%** on a 1,000-example slice) and raise absolute EM for all systems (e.g., S2 **0.600** vs baseline shard S2 **0.446**). Lifecycle still trails free multihop (**S3 0.524** vs **S2 0.600**, Δ = −7.6 EM). Relative to HotpotQA, lifecycle locking and the Hotpot-trained Bridge Reranker **help** on Wikipedia bridges and **hurt or remain neutral** on MedMCQA letter answers.

---

## 2. Motivation and Setup

### 2.1 Why MedMCQA

The HotpotQA report’s future work called for expanding lifecycle testing beyond Wikipedia bridge questions. MedMCQA tests whether the same claim-graph lifecycle (proposed → grounded → locked / retracted) transfers to medical MCQs, where:

- Answers are **letters A–D**, not free-form entity strings.
- “Context” paragraphs are the **four options** (plus optional explanation text), not gold + distractor Wikipedia passages.
- Intermediate “bridges” are clinical concepts, not hyperlinked Wikipedia entities.

### 2.2 Experimental protocol

| Item | Setting |
|------|---------|
| Dataset | `openlifescienceai/medmcqa`, validation split |
| Metric | Letter exact match (A–D); relaxed EM ≡ EM for letters |
| Model | Gemma-3-4B-IT, **bf16** (4-bit yields empty answers) |
| Ablation ladder | naive RAG → S1 multihop → S2 + ModularQA → S3 lifecycle → S4 + Bridge Reranker → S5 + SF → full |
| Full-val run | 4 shards merged, n = **4183** (2026-11-06) |
| Domain-patch slice | indices `[0, 1000)`, n = **1000** |

Configs and artifacts: `configs/`, `runs/final_val_medmcqa/`, `experiments/20261106_full_val_results.md`.

---

## 3. MedMCQA Results

### 3.1 Full validation (baseline lifecycle, n = 4183)

Merged run: `runs/final_val_medmcqa/medmcqa_val_merged_bf16/`

| Setting | EM | pct verified LLM |
|---------|-----:|-----------------:|
| naive_rag | 0.422 | 0 |
| S1_multihop | 0.422 | 0 |
| **S2_multihop_modularqa** | **0.437** | 0 |
| S3_lifecycle | 0.394 | 0.993 |
| S4_lifecycle_rr | 0.390 | 0.993 |
| S5_lifecycle_rr_sf | 0.390 | 0.993 |
| full | 0.389 | 0.993 |

**Takeaways**

1. Best MedMCQA system in this setup is **S2**, not full lifecycle.
2. Lifecycle LLM answering is healthy (~99% verified answers); the gap vs S2 is locking/modeling, not empty generations.
3. Hotpot-trained **Bridge Reranker** is slightly negative (S3 → S4 ≈ −0.36 EM). Prefer **bridge OFF** for MedMCQA reporting.
4. SF labeler does not move EM (S4 ≈ S5).

### 3.2 S2 vs S3 error analysis (full val)

| Outcome | Count |
|---------|------:|
| Both correct | 1,209 |
| Both wrong | 1,917 |
| **S2 only correct** | **619** |
| S3 only correct | 438 |
| Net Δ (S3 − S2) | **−4.33 EM** |

Among the 619 S2-win / S3-lose cases, **484** locked `Option X`-style titles. Symbolic letter recovery was often empty or wrong; even when symbolic was right, the final prediction frequently diverged. Root cause: Hotpot-style hop title-bypass / option-title locking pollutes MedMCQA graphs; free S2 multihop avoids that constraint.

### 3.3 Medical-domain adaptations

Implemented in `flow_rag/medmcqa_domain.py` (auto-applied for `--dataset medmcqa`):

1. **Ban option-title locks** — reject `Option X`, `A. …`, bare letters, meta-options, `Explanation`, numeric-only, question-stem locks.
2. **Medical hop-1 mining** — stem concepts + explanation spans; drop option/explanation titles.
3. **Medical entity extract** — clinical concepts (e.g., Meckel’s cartilage) instead of Hotpot film/director cues.
4. **Loader** — option titles as `A. <text>`; **explanation context ON**; explanation paras use text snippets as titles.
5. **Soft lock fallback** — invalid medical lock or failed grounding → S2-style multihop.

### 3.4 Domain-patch slice vs matched baseline shard (n = 1000, indices 0–1000)

| Setting | Baseline shard EM | Medical-domain EM | Δ |
|---------|------------------:|------------------:|---:|
| naive_rag | 0.420 | 0.561 | +0.141 |
| S1_multihop | 0.427 | **0.604** | +0.177 |
| S2_multihop_modularqa | 0.446 | 0.600 | +0.154 |
| S3_lifecycle | 0.401 | 0.524 | +0.123 |
| S4_lifecycle_rr | 0.404 | 0.524 | +0.120 |
| full | 0.396 | 0.540 | +0.144 |

**Lock audit (S3)**

| Metric | Baseline `[0,1000)` | Medical-domain `[0,1000)` |
|--------|--------------------:|--------------------------:|
| Option-title lock rate | **74.9%** | **0.0%** |
| Soft-fallback traces | 0 | 8 |
| Empty locks | 0.8% | 0.9% |

Option-title locking is solved. Absolute accuracy rises for every setting—largely from **explanation context** plus cleaner retrieval titles. Relative ranking is unchanged: free multihop (S1/S2) still beats lifecycle (S3), and the S2−S3 gap **widens** (−4.5 → −7.6 EM on this slice).

Paired outcomes (medical-domain slice): both right 441, both wrong 317, S2-only 159, S3-only 83. Residual bad locks are now contentful but often wrong or too weak (e.g., vague phrases, occasional `Answer- A` chrome), not bare `Option D`.

Smoke (n=20, post-sync): S2 0.55, S3–S5 0.50, full 0.60—directionally consistent but too small to rank systems.

---

## 4. Comparison to HotpotQA Results

Numbers below are from *Final Report — Question Answering Lifecycles* (HotpotQA **bridge1000**, n = 1000) unless noted.

### 4.1 Head-to-head system quality

| Method | HotpotQA EM | HotpotQA Rel. EM | HotpotQA F1 | MedMCQA full-val EM (n=4183) | MedMCQA med-domain EM (n=1000) |
|--------|------------:|-----------------:|------------:|-----------------------------:|-------------------------------:|
| Naive LLM | 0.079 | 0.130 | 0.130 | — | — |
| Naive RAG | 0.376 / 0.381† | 0.512 / 0.540† | 0.512 / 0.513† | 0.422 | 0.561 |
| + Multihop | 0.390† | 0.543† | 0.517† | 0.422 (S1) | **0.604** (S1) |
| + ModularQA | 0.400† | 0.557† | 0.532† | **0.437** (S2) | 0.600 (S2) |
| + Lifecycle | 0.417† | 0.578† | 0.548† | 0.394 (S3) | 0.524 (S3) |
| + Bridge Reranker | 0.427† | 0.585† | 0.559† | 0.390 (S4) | 0.524 (S4) |
| Full Lifecycle | **0.433** | **0.602** | 0.566 | 0.389 | 0.540 |
| VeriCoT | 0.435 | 0.577 | 0.577 | — | — |

† Ablation Table 2 in the HotpotQA report (same bridge1000 pool). Table 1 reports Naive RAG EM 0.376 for the main comparison table.

### 4.2 Ablation direction reverses across datasets

| Component | HotpotQA (bridge1000) | MedMCQA |
|-----------|----------------------|---------|
| Multihop retrieval | Small gain (+0.009 EM) | Small / large gain once explanations available |
| ModularQA hop-1 | Helps (+0.010) | Helps on full val (+0.015 vs naive); mixed on med-domain slice (S1 slightly > S2) |
| **Lifecycle locking** | **Helps** (+0.017) | **Hurts** (−0.043 full val; −0.076 med-domain slice) |
| **Bridge Reranker** | **Helps** (+0.010) | **Neutral / slight hurt** (−0.004 full val; 0.000 med-domain with RR off in config) |
| Path / repair / rich hop-1 / solver | Helps (+0.006 to full) | No gain vs S3; full still below S2 |

On HotpotQA, each lifecycle stage is cumulative and positive. On MedMCQA, the ladder peaks at **S2** (or **S1** with explanation context) and declines once locking is enabled.

### 4.3 Why the transfer fails

| Factor | HotpotQA | MedMCQA |
|--------|----------|---------|
| Bridge semantics | Wikipedia entity connecting two gold passages | Clinical concept; “paragraphs” are answer options |
| Gold intermediate | Often a title span the reranker can rank | Locking an option title shortcuts to a letter without medical grounding |
| Bridge Reranker training | Hotpot train-bridge; top-1 acc ~0.72 | Out-of-domain features; no MedMCQA-trained reranker |
| Answer form | Free-form string (EM / relaxed EM / F1) | Single letter (EM = letter accuracy) |
| Failure mode addressed | Wrong Hop-1 entity → coherent wrong Hop-2 | Wrong/locked option chrome → constrained wrong letter |

The HotpotQA report’s core claim—that verifying and locking Hop-1 bridges prevents confident wrong Hop-2 answers—holds when bridges are real entities. On MedMCQA without domain patches, the same machinery **locks distractor option titles**. With domain patches, locks are medical but still over-constrain relative to unconstrained multihop + explanation context.

### 4.4 What transfers successfully

- Shared FLOW-RAG ablation harness and auditable traces (hop queries, lock status, rerank scores).
- High **verified LLM** rates on MedMCQA lifecycle paths (~99%), analogous to HotpotQA’s reliable Hop-2 answering once a bridge is fixed.
- ModularQA / multihop structure remains a useful baseline on both datasets.
- Lightweight CPU-side components (NLI, LightGBM) deploy on MedMCQA; they simply need **domain-appropriate training and lock policies**.

---

## 5. Discussion

1. **Best MedMCQA reporting number:** S2_multihop_modularqa **EM = 0.437** (full val, n=4183). With medical-domain context on the first 1,000 examples, S1 reaches **0.604** and S2 **0.600**—treat these as context-augmented upper bounds for that slice, not yet full-val.
2. **Lifecycle on MedMCQA:** Report **S3 (bridge OFF)** if a lifecycle number is required; do not claim Hotpot-style gains from locking or the Hotpot Bridge Reranker.
3. **Domain patches are necessary but not sufficient:** They fix the option-title failure mode (74.9% → 0%) and raise absolute EM, but do not make lifecycle beat S2.
4. **Cross-dataset conclusion:** The Lifecycle Model is **dataset-sensitive**. Gains on HotpotQA bridge questions do not automatically appear on medical MCQs; bridge definition, corpus shape, and reranker training must be redesigned for the new domain.

---

## 6. Limitations

- Full-val (n=4183) numbers are **pre–medical-domain-patch** for lifecycle behavior; medical-domain results are currently a **1,000-example** slice (plus n=20 smokes).
- MedMCQA and HotpotQA metrics are not directly commensurate (letter EM vs free-form EM/F1); comparisons emphasize **ablation direction** and relative ranking.
- Same Gemma-3-4B backbone for MedMCQA; HotpotQA report baselines may use a different generation stack—absolute Hotpot vs MedMCQA EM should not be read as a pure dataset difficulty score.
- Pipeline still bootstraps from exported bytecode (`pipeline_export.cpython-39.pyc`) in lab environments.

---

## 7. Future Work

1. Re-run **full MedMCQA validation** with medical-domain patches + explanation context; publish merged S1/S2/S3.
2. Train a **MedMCQA Bridge Reranker** (or drop reranking) instead of reusing Hotpot LightGBM weights.
3. Redesign Hop-2 to **score options against a locked clinical concept**, rather than free-form generation conditioned on a lock.
4. Stronger / medical LLMs (e.g., MedGemma) and external corpora (PubMed / textbooks) beyond MedMCQA `exp`.
5. Continue residual chrome filters (`Answer- A`, boolean fragments) identified in the domain-patch lock audit.

---

## 8. Conclusion

Extending the Lifecycle Model from HotpotQA to MedMCQA shows that **neurosymbolic locking is not universally helpful**. On HotpotQA bridge1000, lifecycle + Bridge Reranker raise EM to **0.433** (best relaxed EM **0.602**). On MedMCQA full validation, unconstrained multihop (**S2 EM 0.437**) beats lifecycle (~**0.39**), because Hotpot-style locks latch onto option titles. Medical-domain patches remove that failure mode and lift absolute accuracy when explanations are in context, but lifecycle still trails S1/S2 on the patched 1,000-example slice. The productive path forward is domain-specific bridge semantics and reranking—not transferring the HotpotQA lock policy unchanged.

---

## Appendix A — Artifact index

| Artifact | Path |
|----------|------|
| HotpotQA final report | `Final Report- Question Answering Lifecycles.pdf` |
| Full-val notes | `experiments/20261106_full_val_results.md` |
| Domain-patch plan | `experiments/20261106_medical_domain_plan.md` |
| Merged ablation CSV | `experiments/20261106_merged_ablation_df.csv` |
| S2 vs S3 JSON | `experiments/s2_vs_s3_analysis.json` |
| Full-val merged run | `runs/final_val_medmcqa/medmcqa_val_merged_bf16/` |
| Baseline shard 0–1000 | `runs/final_val_medmcqa/medmcqa_val_0_1000_bf16/` |
| Medical-domain 0–1000 | `runs/final_val_medmcqa/medmcqa_val_medical_domain_0_1000_bf16/` |
| Domain adapters | `flow_rag/medmcqa_domain.py` |

## Appendix B — Progressive Δ EM (MedMCQA full val)

| Setting | EM | Δ vs previous |
|---------|-----:|--------------:|
| naive_rag | 0.422 | — |
| S1_multihop | 0.422 | +0.000 |
| S2_multihop_modularqa | 0.437 | +0.015 |
| S3_lifecycle | 0.394 | −0.043 |
| S4_lifecycle_rr | 0.390 | −0.004 |
| S5_lifecycle_rr_sf | 0.390 | 0.000 |
| full | 0.389 | −0.001 |

Contrast with HotpotQA Table 2, where every step from naive RAG → full lifecycle is non-negative and lifecycle/reranker steps are among the largest gains.
