# JMQ prompt-level review export

- `jmq_prompt_review.xlsx`: easiest option for Excel; includes prompt reviews, model summary, and README sheets.
- `jmq_prompt_review.parquet`: recommended for Colab/Pandas; complete multiline text with compact storage.
- `jmq_prompt_review.csv`: Excel-compatible UTF-8 CSV. Multiline text is quoted correctly.
- `jmq_prompt_review.jsonl`: one complete JSON object per prompt/model pair.
- `AblationWriting_JMQ_Review.ipynb`: upload this notebook to Colab, then upload the Parquet file when prompted.

The export has 1,000 rows: 200 direct H2-vs-baseline pairs for each of five frozen models. Rows where both candidates refused remain visible but have `both_refused_excluded=true` and no per-row ablated score. Schema v2 reports order-corrected TEST L2 recovery and verified TEST MMD fields. `model_stats_json` contains the complete repeated VAL2, TEST, JMQ, preservation, and MMD record.
