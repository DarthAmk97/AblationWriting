# AblationWriting, Humanization by Ablation (H2)

Training-free selective ablation of post-training writing collapse. Five frozen safety cones on five small instruct models, tested against dose-matched random controls, lexical controls, and 5,000 blind human-vs-model judgments.

Status: H2 frozen and sealed. The cone beats dose-matched random 5 of 5. Lexical deletion ties or beats the cone on Q08 and Q20. Blind quality preference never improves. Conditional claim: mixed. Details in claims_ledger.md.

## Headline numbers

Core TEST MMD recovery: L1 +0.108, O1 +0.214, Q08 +0.022, Q20 +0.240, G2 +0.134.

Blind JMQ, refusal-filtered ablated win share: L1 0.343, O1 0.370, Q08 0.472, Q20 0.363, G2 0.490. None better than baseline.

v5 TEST MMD, cone vs dose-matched random: G2 +0.193 vs -0.067, L1 +0.092 vs -0.036, O1 +0.332 vs -0.045, Q08 +0.020 vs -0.134, Q20 +0.411 vs -0.091. Random-vs-cone Holm p = 0.0005, all five.

## Repo map, what is here and why

| Path | Why it exists |
|---|---|
| src/ | the entire pipeline, nothing trimmed: geometry sweeps, TEST, judging, MMD, controls, verification, watchdogs |
| metrics/tables/ | all 11 rendered results tables, the human-readable face of the results |
| metrics/*.parquet | the same results machine-readable, with CIs and Holm p-values |
| claims_ledger.md | every paper claim mapped to evidence and a verdict, the adjudication record |
| CONTROL_PROTOCOL.md | the frozen v5 control spec, including v5-patch1 and the decision rule |
| failures.md, lab_notebook.md, state_ledger.md | provenance: what failed, what was tried, what it cost |
| figures/ | PNG, SVG, and the data parquets behind each figure |
| exports/ | reviewer packs: prompt-level review and the 150-pair case probe |
| paper/ | section drafts, appendix, references; main manuscript compiles from these |
| manifest.yaml, DUMP_SPEC.md | frozen revisions, splits, and sampler, the identity of the study |
| judgments/ | 5,000 blind pairwise judgments in four shards plus the refusal audit |
| artifacts/geometry/H2/ | per-model small markers and metrics; bulk embeddings and bases live on HF |

## Data, here vs HF

This repo holds code, paper, tables, and small results. Bulk data, about 12 GB, lives in the private HF dump amkkk/AblationWriting-H2-dump, organized under the AblationWriting collection with per-folder data cards. Per-model repos hold each frozen cone basis, generations, metrics, and a model card:

- amkkk/AblationWriting-H2-L1-cone, Llama-3.2-1B-Instruct
- amkkk/AblationWriting-H2-O1-cone, OLMo-2-0425-1B-Instruct
- amkkk/AblationWriting-H2-Q08-cone, Qwen3.5-0.8B
- amkkk/AblationWriting-H2-Q20-cone, Qwen3.5-2B
- amkkk/AblationWriting-H2-G2-cone, Gemma-4-E2B-it

No GGUFs by design: the cone hook removes only the positive projection on generated tokens, which no static weight edit can express exactly. Each model repo documents this with the numbers. The exact intervention is ~35 lines, see below.

## Usage

Exact cone, transformers plus 30 lines: rebuild each layer basis with src/h2_controls.py build_control_bases from full_layer npz files plus frozen.json, apply src/h2_controls.py GeneratedTokenHook at temp 0.8, top-p 0.95, top-k 50. Same seeds as controls_plan.json reproduce our 400 TEST outputs exactly.

Analysis only, no GPU: every generation, metric, and judgment is a parquet. Pandas reproduces every table.

Lexical control: the exact deletion rule lives in controls_lexical_policy.json, K token ids at rate q, hash-decided per position, with our LEX-MATCH outputs beside it for comparison.

## Repro map

combined_test_jmq.md from combined_test_jmq.parquet via src/combined_results.py and src/test_headline.py. paired_mmd_jmq_l2.md and grouped_distribution_shift.md likewise. jmq.md, jmq_domain.md, jmq_truncation.md, style_bigword.md, jmq_rationale_themes.md from their parquets via src/jmq_stats.py. mmd.md from per-model mmd_metrics.json via src/mmd_score.py. controls.md from control_metrics.parquet via src/control_stats.py. Prompt review via src/export_jmq_review.py. A one-command verifier is tracked as repro/, in progress.

## License and privacy

Private research repo. GEN human references are CC-BY-NC. No model weights here. Pinned env: Python 3.12.14, Torch 2.11.0+cu128, Transformers 5.16.1.
