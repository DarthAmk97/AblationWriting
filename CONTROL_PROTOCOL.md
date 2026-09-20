# H2-MRSC-400-v5 — frozen RAND-only controls (v4 lex-merge defect fixed)

Frozen 2026-09-17 as epoch `H2-MRSC-400-v5-rand-only`. This supersedes v4, which completed all L1 generations but died merging CPU LEX rows through the archived-row merge path (a deterministic code defect, not a scientific failure): L1's 1200 TEST generations survive in the progress file and are adopted after strict hash verification, with equivalence, calibration, and LEX policy regenerated. v1 (bit-replay), v2/v3 (PC1 dose) remain permanently blocked.

Frozen 2026-09-17 as epoch `H2-MRSC-400-v4-rand-only`. This superseded v3, which passed the same-runtime equivalence gate but failed dose calibration on L1 even under total-dose matching: the nested PC1 direction needed alpha 16.83 (cap 4.0), carrying only ~1/17 of the frozen cone's positive-part removal. This is a post-hoc mechanistic suite, not preregistered confirmation. It supports only a conditional claim about the five frozen headline models.

## v5 change (lex-merge defect fixed; v4 progress adopted)

v4's post-LEX merge passed CPU LEX rows through the archived-row merge helper, whose fixed-lookup comparison can never contain not-yet-merged rows: deterministic failure after all generations complete. v5 merges LEX rows through a dedicated validated-append path (with a regression test), and adopts v4's surviving L1 progress/anchors after strict text-hash verification while regenerating equivalence, calibration, and LEX policy. Generation semantics are unchanged.

## v4 change (PC1 dropped as dose-infeasible)

Per-layer matching (v2) needed alpha 25.15 at L1 layer 8; total-dose matching (v3) still needed 16.83. No sane-strength single direction reproduces the 4-direction cone's removal burden, so PC1-DOSE is **dropped as a generation arm** and recorded as an infeasibility finding — itself evidence that the top direction alone does not explain the cone effect. v4 generates only the matched-rank random-subspace arm (plus contemporaneous anchors, CPU LEX-MATCH, and reused FULL-SYM). The primary contrast is frozen cone vs dose-matched RAND. v1 (archived bit-replay) and v2/v3 (PC1 dose) remain permanently blocked.

## Why v2

The v1 epoch required exact bit-reproduction of archived sampled text on a different NVIDIA driver (original `580.173.02` vs current `595.71.05`). All 8/8 unhooked baselines mismatched before any hook logic ran, so the failure is a runtime-identity mismatch, not hook evidence. v2 retains archived replay as a recorded environment diagnostic but never uses it as hook-equivalence evidence.

## Scope

- Models: `L1 O1 Q08 Q20 G2`, exact revisions in `manifest.yaml`.
- Main split: fixed `TEST-JMQ` 400 prompts/model; no filtering after generation.
- Contemporaneous v5 runtime (`contemporaneous-v5-runtime`): generate `ANCHOR-BASE`, `ANCHOR-CONE`, and `RAND-RANK-DOSE` together in the same per-model process/runtime. `HUMAN` reuses archive by exact text hash. `LEX-MATCH` is CPU-applied to contemporaneous `ANCHOR-BASE`. (PC1-DOSE dropped as dose-infeasible; see above.)
- Archived secondary stratum (`archived-original-runtime`): VAL2-512 `HUMAN`/`VAL2-BASE`/`VAL2-CONE`/`FULL-SYM` only. Never pool absolute outputs across strata.
- VAL1-96 contemporaneous anchors: generate `ANCHOR-BASE` and `ANCHOR-CONE` in the same runtime before TEST; calibrate `LEX-MATCH` only from these.
- Sampler: temperature 0.8, top-p 0.95, top-k 50, batch 1, BF16, no quantization, cap 1024, generated-token residual stream only.
- Generation seed: `(1000 + int(sha256(prompt)[:8],16)) % (2**31 - 1)`.
- Nominal budget: 58–64 GPU-hours; hard cap: 72 GPU-hours (contemporaneous anchors added; unchanged by the v2→v3 matching change).

## New representation controls

`PC1-DOSE` uses exactly the first column of the final frozen cleaned/QR basis at every frozen layer. `RAND-RANK-DOSE` uses the frozen rank and layers with independently keyed Gaussian bases in whitened coordinates, mapped and protected by the implemented native-space procedure.

Random basis seed per model/layer:

```text
(17011 + int(sha256("H2-MRSC|RAND|<model>|<layer>")[:8],16)) % 2147483647
```

Both arms are dose-matched per layer to the contemporaneous frozen cone on a deterministic 64-prompt VAL1 calibration pass. A non-finite, non-positive, or greater-than-four calibrated alpha fails the arm; it is never silently clipped.

## Same-runtime equivalence gate

Before calibration or new generation, require exact 8-prompt same-runtime equivalence on TEST-JMQ:

- immutable `smoke_h2.generate_batch` vs new no-hook path;
- immutable `smoke_h2.generate_with_hooks` / `h2_lib.ConeAblationHook` vs new `GeneratedTokenHook`, using the same reconstructed frozen basis and sampler.

Save full expected/actual text hashes, first-divergence diagnostics, code hashes, basis hash, and runtime fingerprint. Fail if old/new current-runtime paths differ. Archived replay mismatches are saved with status `archive-bit-replay-mismatch-expected-runtime-drift` and are not success evidence.

## Lexical comparator

Rank baseline-excess unigrams on contemporaneous VAL1-96 anchors by squared excess over human. Search `K={1,2,4,8,16,32}` and deletion rate `q={.125,.25,.5,.75,1}`. Each selected occurrence is deleted when the SHA-256 integer of

```text
H2-MRSC|LEX|<model>|<prompt_id>|<zero-based token position>
```

falls below `q*2**256`. Select the pair closest to the contemporaneous H2 absolute L2-1 shift, breaking ties by fewer deletions, lower `q`, then lower `K`. Apply the frozen rule unchanged to contemporaneous TEST `ANCHOR-BASE`. Call it matched only when the TEST shift has the same direction and an absolute-magnitude ratio in `[0.8,1.25]`.

## Runtime and completion gates

Progress checkpoints are atomic every 16 prompts and resume only under identical source, plan, epoch, and runtime fingerprints. Core MMD artifacts are read-only. Failure diagnostics are preserved atomically before raising.

Completion requires per-model VAL1 anchors, lexical policy, generation, calibration, equivalence/replay records, embeddings, metrics, and hash markers; 10,000 paired prompt-attribution bootstraps; Holm correction across the five frozen models within each arm/metric comparison; and a global suite marker.

## v5-patch1 (stats fail-soft for degenerate JSD denominators; generations/embeds untouched)

`control_stats.py` aborted fail-closed on `L1 test_jmq/ANCHOR-BASE/JSD` (later also `Q20`):
10k-bootstrap `validfrac` 0.9157 (L1) / 0.9634 (Q20) vs the 0.99 guard. Measured cause
(`--diag`): TEST unigram-JSD baseline sits almost on the shared human–human floor
(floor 0.0924; L1 point 0.0987, denom 0.0063), so resamples routinely cross the floor
and the recovery ratio divides by ~zero. Not a data bug — the metric carries no
recovery signal there. Patch: rows with finite bootstrap frac < 0.99 or a non-finite
point estimate record flagged NaNs (`recovery_undefined_reason=degenerate_denominator`)
with raw distances preserved and the block kept visible; every other row is
bit-identical (same seeds, same 10k bootstraps, same Holm). Decision rule keys on MMD
and is unaffected. Code change is stats-only; all generation/embed hashes stand.

## Decision rule

The conditional representation/selectivity claim is supported only if:

1. Contemporaneous frozen cone MMD recovery exceeds the dose-matched random-subspace control in the median and in at least four of five models.
2. `LEX-MATCH` succeeds for at least four models and contemporaneous cone has greater MMD recovery in at least four of those five.
3. Symmetric full-subspace removal does not systematically outperform the frozen cone on reused VAL2 evidence (secondary stratum only).
4. Failed calibrations, failed lexical matches, preservation losses, and negative models remain visible.

Otherwise the relevant claim remains mixed or is rejected. This suite cannot establish the original universal eight-model or external-generalization claim.
