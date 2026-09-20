# state_ledger.md — autonomous H2 driver (do NOT wait on user except BLOCKED)
# TEST PRINCIPLE (2026-09-11, user): TEST runs regardless; safety overlap = insight for H3, not a block.
# ORDERING PRINCIPLE (2026-09-09, user): all GPU steps first on VAST; all decouplable time-heavy non-GPU work goes LOCAL with dump lookups.
# GPU chain (VAST): smokes -> Screen0/S1/S2/S3 -> preservation-full -> TEST -> MMD-embed + StoryScope-GEN + controls-gen.
# LOCAL chain (dump-fed, no GPU): JMQ DONE -> bootstrap/Holm DONE -> MMD verified DONE -> joint figures/exports DONE -> paper active -> controls pending.
# SYNC SAFETY (2026-09-16): local is ahead; while `.local_ahead.json` exists, never download HF directly into `dump_mirror`. Stage separately and preserve local-root authority per `LOCAL_SYNC_POLICY.md`.
# NEW VAST (2026-09-16): RTX 3090 at 154.64.230.67:25326, recovery key `vast_ed25519_20260916_recovery`; minimal 32GB reconstruction under `/root/AblationWriting`. MMD completed 5/5 and was published to the private HF dump at commit `9da4909fc4c1ded93b929e336f14aa421ab4931f`. Core H2 remains archived locally/HF and must not be rerun.
# Figure-data tables stage into dump INCREMENTALLY (runs.jsonl already suffices for draft S1-S3 figures — local paper work starts before TEST).
# Goal: complete H2 per protocol, push to HF, mark H3-awaiting partials. Updated every stage.
# Scope: /root/AblationWriting ONLY. Last update: 2026-09-18 by agent (L1 v5 finals pulled+verified+promoted locally; global evidence registry built with 1832 entries across 17 categories; O1 running on VAST).

## 0. Initial instruction (what the first prompt told me)
- VAST `ssh -p 42632 root@173.239.92.155 -L 8080:localhost:8080`, copying from another instance → create `AblationWriting` there, operate ONLY inside it.
- Valuable work → HuggingFace: collection `AblationWriting` + `writeup.html`. Token `hf_oWb...cbR` (user amkkk, validated).
- JMQ via opencode-go key `sk-2q7C...9me` model `muse-spark-1.3-contributor` → RESOLVED to `POST https://opencode.ai/zen/v1/responses` model `muse-spark-1.3-contributor-free` Responses API + `User-Agent Mozilla/5.0` (VAST IP was Cloudflare-1010 without it; verified local+VAST).
- H3 NOT available: finish H2 first → HF repo+html, then H3 separate repo, then completion + html append. Mark partials `awaiting-H3`.
- Full H2 protocol (25 secs): low-rank cone (NOT rank-1; rank-1 appendix baseline only), 8-model panel exact SHAs, GEN primary + dmitva external + StoryScope structural, immutable CAL/VAL1/VAL2/TEST/TEST-JMQ/INTERP, sampler 0.8/0.95/50 seed-matched cap 1024 generated-only locus, teacher-force means + paired Δ, whiten + SVD r∈{2,4,8,16,32}, RepIt protected ρ∈{0.5,0.75,1.0}, independence matrix, positive-cone α∈{0.25,0.5,0.75,1.0} (full-subspace control), successive halving 24→12→4→1, metrics L2-1/2/3 (fixed Qwen3.5-0.8B tok) + JSD + MMD (nvidia-embed-8b RBF median-VAL) + R recovery + JMQ_H=2P(beat human) + StoryScope 304-feat, preservation gates IFEval95/capab97/refusal95/path+2pp, Pareto decision, bootstrap 10k + Holm, 11 controls, H4 10-criteria, manifest/runs/geometry/records, figures4papers style, canonical paper/, 30-70 GPU-h on 1×3090, smoke Q08+L1 512/128 (L1 gated → O1 sub).
- Poll with ticker every 30 min when stable: remote `ticker_loop.sh→ticker.log` DONE + chat `/loop 30m ablation-ticker` ARMED (state proves runs 03:10/03:40).

## 1. Abbreviations
| Abbr | Meaning | Notes |
|---|---|---|
| Q08/Q20 | Qwen3.5-0.8B/2B `Qwen3_5ForConditionalGeneration` multimodal hybrid | hook language backbone only, text-only |
| G2/G4 | Gemma-4-E2B/E4B-it `Gemma4ForConditionalGeneration` multimodal | one-at-a-time on 24GB |
| L1 | Llama-3.2-1B-Instruct 16L/2048d text | UNLOCKED 05:00 UTC (was gated, user accepted) — smoke RUNNING packed with Q08 |
| O1 | OLMo-2-0425-1B-Instruct 16L/2048d text | smoke DONE |
| LF12 | LiquidAI LFM2-1.2B 16L/2048d hybrid-conv | text ForCausalLM, pack with Q08 |
| S3 | SmolLM3-3B 36L/2048d text | pack with Q20 |
| CAL/VAL1/VAL2/TEST | GEN matched prompt IDs 2048/256/512/2000 | immutable, TEST never touched in selection |
| TEST-JMQ/INTERP | 400 TEST subset / 128 CAL-only token-plots | JMQ blind, INTERP never headline |
| GEN/PAIRED | szyszy/GEN (human/ai_generated/ai_edited) / dmitva 1M (external 1000) | don't pool PAIRED into selection |
| W1/W2 | depth windows W1 25-44% layers[4-7], W2 50-69% layers[8-11] (O1) | smoke only; full = Screen-0 stability |
| r/a/ρ | rank {2,4,8,16,32} / alpha {0.25,0.5,0.75,1.0} / protection {0.5,0.75,1.0} | smoke ρ=0.75 only |
| R/κ/rem/pos | human-gap recovery / throughput mult vs baseline / removed norm per tok / % positive coords | κ target ~1.0 (<1.20 gate) |
| MMD/JMQ | embed distribution distance / blind judge win-rate | MMD nvidia-8b; JMQ opencode-go/muse-spark-1.3-contributor via CLI |
| H4 | no-compact-erasable verdict (10 criteria, needs H2+H3) | awaiting-H3 |

## 2. Stage table (smoke =/= headline)
| Stage | Scope | Status | Evidence |
|---|---|---|---|
| Env + splits | torch2.11cu128/trans5.16, CAL2048/VAL1256/VAL2512/TEST2000/JMQ400/INTERP128/EXT1000 | DONE | manifest + split_summary, 5-domain balanced |
| Funnel + freeze | CAL2048, Screen0→S1→S2→S3→VAL2 | DONE | L1/O1/Q08/Q20/G2 frozen; LF12 uniformly negative; S3 reverted on VAL2; G4 stopped after Screen0 |
| Preservation | IFEval full, MMLU-Pro500/GSM8K500, safety200+200, KL256, pathologies | DONE | instruction/capability gates pass; harmful-refusal retention reported as overlap insight |
| Held-out TEST | 2,000 prompts/model, frozen configs only | DONE 5/5 | raw L2/JSD valid; downstream R repaired with order-matched frozen VAL2 floors |
| JMQ | overall 400+400+200/model | DONE | 5,000 valid judgments; refusal-filtered 10k bootstrap, exact tests, Holm complete |
| MMD | pinned embedder, VAL2-only bandwidth, VAL2+TEST | DONE 5/5 | 10 embeddings + metrics/done verified by SHA; HF dump commit `9da4909f...` |
| Joint stats/figures/exports | corrected L2 R + MMD + JMQ + preservation | DONE | combined parquet/table/PNG/SVG and 1,000-row review export v2 |
| Controls | H2-MRSC-400-v5 RAND-only; lex-merge fixed, L1 v4 progress adopted | RUNNING | v5 epoch HF b58cf663, v4 state archived, launched alone as PID 9719, first model L1; 72 GPU-h hard cap |
| Paper | findings, methods, caveats, appendix | ACTIVE | Proposed title recorded 2026-09-17: `Humanization by Ablation` (user proposal; needs claim-scope check before finalizing); MMD and TEST floor correction integrated; full compile still pending |
| Final HF publish | repo `amkkk/AblationWriting-H2` + collection + writeup.html | QUEUED after controls | private dump contains verified core + MMD additions |

Smoke ≠ headline: smoke uses 512/128, max_new256, ρ-only-0.75, simplified 4-pair protect, L2-only cheap metric. Headline uses full CAL/VAL2, 1024 cap, full ρ/α/r grids, 6-behaviour protect, MMD+JMQ+StoryScope + gates.

## 3. Learnings so far
1. Hook must handle KV-cache: `T==1` decode = generated (fixed; was rem=0 bug, 5 nulls discarded, logged failures.md).
2. Native mapping must be `Winv@V` (sqrt Sigma), not `W@V`.
3. Depth matters more than rank: W1 neg→W2 r4 positive→r8 neg = elbow at 4 (H2-like, not H4 monotonic).
4. κ≈1.04-1.08 healthy (<1.20); 0 pathologies; JSD can improve while L2 worsens (need both + MMD).
5. VAST IP blocked by Zen Cloudflare without UA; fixed with UA header (JMQ READY both sides).
6. /workspace HF cache empty (syncthing incomplete) → isolated `.hf_cache` mandatory.
7. L1 gated needs user license; Qwen/Gemma multimodal need backbone-only hooks.

## 4. Next steps (autonomous — do not wait)
- [1] Keep the verified HF-downloaded MMD artifacts authoritative locally; never replace `dump_mirror` directly while `.local_ahead.json` exists.
- [2] Finish paper/state updates and validate the corrected joint table, figure, review exports, and watchdog.
- [3] Hydrate the frozen H2-MRSC-400 inputs and guarded runner through the private HF dump; verify all source hashes on VAST.
- [4] Run H2-MRSC-400 alone on the VAST GPU, publish artifacts through the private HF dump, and verify locally by hash.
- [5] Close claims, compile the manuscript, publish `amkkk/AblationWriting-H2`, and append `writeup.html`.
- BLOCKED-only waits: an explicit control-scope decision if the minimal default is rejected, H3 unavailable (mark awaiting-H3), disk>90%, SSH down, or HF authentication failure.

## 5. Discussion log (paper-enriching decisions with user, Sep 8-9)
- R% legend: share of human gap closed on word-distribution vs human-human floor. +N% = voice gap erased; -N% = pushed away; 0 = untouched. Phenotype = distributional habits (tics, scaffolding, headers, preambles), NOT quality (JMQ), depth (StoryScope), or safety (gates).
- Qwen3.5-2B keeps full sweep despite flat smoke: anti-cherry-picking (panel scored 6/8, missing != negative), Qwen3.5-0.8B precedent (smoke flat -5.5% to full late +20.6%), H4 needs full negatives. VAL2-freeze skips ONLY on uniformly-negative full evidence (LiquidAI/LFM2-1.2B rule: 62/62 <=0).
- Rank scales with size: 1B peaks rank-4, SmolLM3-3B rank-32, all late-ish. Cone-vs-full: positive-only beats full removal everywhere tested.
- Dropped for timeline (logged, not forgotten): multi-pass re-probing, winsorization, KL-rollback, bias check. Best H4 adjudicator among them remains multi-pass if reviewers demand it.
- OBLITERATUS read: we mirror their serious ideas (mean-diff baseline, SVD/whitened-SVD, cones, CoT-protect, strength sweep, reversible steering); forbidden: weight surgery + probe-refined dirs; stolen: nothing yet (multi-pass parked).
- Ordering principle: GPU chain first on VAST, local chain dump-fed; figure-data incremental; StoryScope extraction local.
- Full canonical names in messages; abbreviations ledger-only lookup.
- PHENOTYPE NOTE (paper-bound, Sep 9): ablation operates on tracked human-AI distance metrics (L2-1/2/3, JSD, later MMD) — n-gram movement is byproduct AND readout, not target. Evidence: correlated multi-order moves (SmolLM3 L2-1/2/3 together); Llama bigram-flat dissociation (unigrams+JSD move, bigrams don't) = representation-level signature a word-ban cannot produce. Explicit top-k contributor suppression adopted as CONTROL arm at matched shift (VAL2, frozen models) — if cone beats ban on preservation/generalization at equal L2, cone story strengthens. Length audit: abl length ~= base << human (no length confound; length gap itself an unablated phenotype dimension, on the books).
- PASS RULE (2026-09-13, user): preservation `pass=true` when the ONLY miss is harmful-refusal retention (refusal_rel<0.95); refusal overlap is a writing/refusal cone-overlap insight for H3, not an H2 block. IFEval>=95% rel, capability>=97% rel, false-refusal<=2pp and pathology<=2pp still gate. Coded in `preserve_full.py` (`pass_rule: refusal-exempt-2026-09-13`); L1/O1/Q08/Q20 JSONs recomputed to pass=true with `refusal_only_miss: true`.

## 6. Shutdown / local-takeover checklist (2026-09-13, user: end VAST spend, replicate 1:1 locally)
- GPU remainder (VAST): Q08/Q20 TEST (~13-15h packed) -> G2 preserve (4-10h) -> G2 TEST (~20-30h) ->
  G4 S1/S2/S3 + preserve + TEST (~4-7 days, critical path) -> MMD-embed (2-4h) + StoryScope-gen (4-10h, needs
  released prompts) + controls-gen (scope TBD, see below). Budget ~2 more weeks of the box.
- LOCAL chain (no GPU): JMQ judging from dump feed -> stats/bootstrap (10k, Holm) -> figures -> paper -> push.
- Dump retrieve-clean-repeat: VAST `stage_dump.py` pushes on frozen-change/6h; LOCAL downloads dump, runs JMQ/stats,
  scp's `judgments/*.parquet` + `metrics/*` back to VAST same paths; next auto-push carries them. Repeat until
  `dump_manifest.json` pending shows only `figures/data/*` resolved + judgments present, then stop VAST.
- Replication contract (all in dump): splits jsonl + SHAs (manifest.yaml) + src/*.py/*.sh incl. local_jmq.py copy +
  code_hashes.json + runs.jsonl (+ .bak audit trail) + geometry (npz/parquet/json/test_done) + preservation/TEST JSONs +
  judging feed + judgments + metrics/tables/*.md (all 3 tables, stamped per run via `table_query.py --out`) +
  paper tex. Excluded by design: weights (SHA-pinned), .hf_cache, secrets, logs/ (diagnostics; decisions live in
  failures.md + PROGRESS.md, both dumped), transient test_progress.parquet, operational dotfiles.
- Collection plan (after H2): per successful writer model in `amkkk/ablationwriting-*` collection — card shows frozen
  VAL2 row + TEST row + preservation row (the 3 archived tables), links frozen run_id, geometry npz, generation
  parquets, refusal note where applicable. Weak/reverted models (LF12, S3) get cards marked as such, not dropped.
- Open decisions that move the date: (1) controls cut to a NAMED arm list with per-arm GPU budgets (11-vs-12 count
  unresolved: top-k word-ban is the 12th); (2) JMQ sharding (sequential CLI ~20s/call x ~7.5k calls = ~2-4 days —
  needs parallel shards per model/dim); (3) StoryScope prompts unreleased -> stays awaiting-H3, never blocks publication.
- G4 SKIP (2026-09-13, user): google/gemma-4-E4B-it removed from QUEUE + parked in .blocked_models. Stop point: smoke
  12/12 done (best W2-r4-a0.5 R+0.180) + Screen0 done (windows W-A layers 10-15, W-B 21-26, ranks 2-32); S1/S2/S3,
  preserve, TEST never started. Afternote for paper appendix: stopped on cost/benefit, revisit only if H2 looks thin
  without it. G4 numbers stay in smoke/stages tables, out of headline claims.
- STORYSCOPE SKIP (2026-09-13, user): never in scope, not "skipped" — no generation, no figure, no skip-note in paper.
  Russell et al cited for phenotype/dimension framing only (claims_ledger row updated).
