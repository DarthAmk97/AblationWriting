# repro/ - one-command verification for every results table

`python repro/verify.py` re-renders all Tier 1 tables from frozen inputs with the
real pipeline code and compares them to the committed files. Exit 0 means every
number checks out. `--only` selects a subset. `--write-manifest` re-records
`repro/manifest.json` after review, never silently.

## Tiers

Tier 1, byte-exact re-render: controls.md, combined_test_jmq.md,
paired_mmd_jmq_l2.md, grouped_distribution_shift.md, jmq.md, mmd.md, plus the
figure-data parquets behind combined_test_jmq and jmq_overall (frame equality).

Tier 2, hash-pinned: jmq_domain, jmq_truncation, style_bigword,
jmq_rationale_themes, preliminary_summary. Rendered once by ad-hoc analysis;
pins live in manifest.json with provenance notes. Rerunning the analyses is
future work, not a gap in the frozen results.

PNGs are hash-recorded only. Exact bytes depend on the matplotlib build;
the data behind them is Tier 1.

## Normalizations, documented not hidden

- jmq.md carries a generation timestamp on line 1. The verifier drops it.
- VAST renders LF, Windows writers emit CRLF. Tables compare on content;
  raw shas stay pinned in the manifest.

## Inputs

Everything needed sits in this repo or the pinned dump commit recorded in
manifest.json (amkkk/AblationWriting-H2-dump). Missing inputs are reported,
never silently fetched into authoritative paths; staging follows
LOCAL_SYNC_POLICY.md.

## Env

Recorded in manifest.json on every --write-manifest run. Reference: Python
3.12.14, Torch 2.11.0+cu128, Transformers 5.16.1. Verification itself needs
only numpy, pandas, scipy, matplotlib.