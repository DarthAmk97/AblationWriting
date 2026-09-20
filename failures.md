# failures.md — OOMs, numerical failures, incompatibilities, negative findings (never silently erase)

## 2026-09-07 — L1 gated
- `meta-llama/Llama-3.2-1B-Instruct` config.json 403 GatedRepoError with token hf_oWb...cbR. User not in authorized list. Action: log as PRE-TEST exclusion candidate; attempt access request; smoke uses O1 instead. Do not silently drop — requires explicit log before TEST per Sec 2.

## 2026-09-07 — Qwen3.5 / Gemma-4 multimodal
- Qwen3.5-0.8B/2B `Qwen3_5ForConditionalGeneration`, Gemma-4 `Gemma4ForConditionalGeneration` are vision-language with hybrid Gated DeltaNet/MoE. H2 hook must target language backbone residual stream, text-only inputs, vision encoder untouched. If hook incompatible, log exact layer/tensor mismatch. Not excluded yet.

## 2026-09-07 — /workspace HF cache incomplete
- `/workspace/.hf_home/hub/*` 0 blobs, only refs/snapshots. Syncthing peer 136.0.41.217:22067 still syncing. Mitigation: isolated cache /root/AblationWriting/.hf_cache. Do not read/write /workspace.

## 2026-09-07 — /venv/main torch missing
- Base image has no torch. Installing torch cu128 + transformers stack in background. If kappa (intervention throughput) >1.20, profile/vectorize per Sec 25.

## Template for future
- run_id, model, config, split, error, VRAM, wall time, status, artifact path, resolution

## 2026-09-07 02:00 UTC — JMQ endpoint 403 (PRE-TEST, non-blocking)
- Tried OPENCODE_BASE_URL=https://api.opencode.ai/v1 with model muse-spark-1.3-contributor, key sk-2q7C...9me → HTTP 403 Forbidden after 3 retries (judge_jmq.py --test).
- Action: JMQ deferred until frozen H2; distributional + preservation proceed. Must resolve endpoint or substitute BEFORE opening TEST per Sec 12.5, record change. Candidate: opencode CLI auth vs custom base URL — need user clarification on opencode-go provider URL.
- Raw: request_id logged, no judgments yet. Status: JMQ_PENDING.

## 2026-09-07 02:15 UTC — JMQ RESOLVED (PRE-TEST)
- Root cause: wrong endpoint (tried api.opencode.ai/v1/chat/completions) + missing User-Agent (Cloudflare 1010) + model id missing -free suffix.
- Correct per https://opencode.ai/docs/zen/: POST https://opencode.ai/zen/v1/responses, model muse-spark-1.3-contributor-free, Responses API {model,instructions,input}, headers Authorization Bearer + User-Agent Mozilla/5.0 + Accept json.
- Verified both local and VAST (request resp_6a9e1cb5...). Judge outputs JSON winner A/B/tie + reason. Separate calls per dimension, randomized order. Raw + request IDs saved.
- Update judge_jmq.py fixed. Status: JMQ_READY.

## 2026-09-07 02:45 UTC — Hook KV-cache bug: rem=0 pos=0 (PRE-TEST, fixed, smoke restarted)
- Symptom: 5 configs R=0.000 rem/token=0.0000 pos=0.000, L2 identical to baseline. Cause: ConeAblationHook checked hs.shape[1] <= prefill_len -> return. With use_cache, decode steps are T==1 <= prefill_len, so never intervened.
- Fix h2_lib.py: T==1 cached decode -> intervene all; T>prefill_len -> tail; else prefill untouched. Use gen_start for writeback. Also fixed native mapping to Winv@V (sqrt Sigma) in smoke_h2.py.
- Action: killed PID 3244, cleared runs.jsonl (5 null runs discarded as buggy, not negative evidence), restarted PID 3405 smoke v3. Previous nulls must NOT enter claims.
- Lesson: always log rem/token + pos_frac (now added) to distinguish true null from dead hook.

## 2026-09-08 03:50 UTC — S1 noisy-floor R artifact (PRE-TEST, corrected, no rerun needed)
- Symptom: S1 R values wild (O1 +0.45 with L2 worsened; L1 -5.6 and +4.2). Cause: human floor from 16-vs-16 halves is unstable and can exceed baseline distance, flipping signs and exploding magnitudes.
- Fix: frozen floor VAL1 256 halves L2-1 0.010442, L2-2 0.006575, L2-3 0.005331, JSD 0.118547 in artifacts/metrics/val1_floor.json. S1 code now uses frozen floor; finished S1 configs re-ranked by RAW L2 via correct_s1.py (rem>0, pathology gate) not noisy R.
- Running S1 procs keep old code (hot-patch impossible); their raw L2/rem/path stay valid, only their logged R + top12 file get corrected post-hoc. No GPU time wasted.

## 2026-09-08 05:00 UTC — S1 W-BEST empty-hooks bug (PRE-TEST, fixed, 10 configs rerunning)
- Symptom: O1 W-BEST x5 rem=0 exactly; L1 W-BEST x5 rem>0 but single-layer. Cause: stage_s1 loaded geometry only for W-A/W-B layers; W-BEST [8-11] layers absent from geo dict so hooks_t came out empty (O1) or partial via layer-8 overlap (L1). Diagnosis probe (mu shift 0.10, native top-1 cos 0.99, all variants posrate 0.6+) proved geometry+protect healthy — pure plumbing.
- Fix: load union of all tested windows; added --only subset flag. Removed 10 invalid rows (backup runs.jsonl.bak-wbest-bug-*), rerunning W-BEST r2/4/8/16/32 packed O1 PID62781 + L1 PID62782 (~15 min). S1 top12 rewritten by correct_s1.py after.

## 2026-09-08 12:00 UTC — S3 dispatcher branch missing (PRE-TEST, fixed, ~30 min partial idle)
- Symptom: L1 S3 job exited instantly with stage-queued message; LF12 S1 finished meanwhile; card ran 1/3 full ~30 min.
- Cause: stage_s3 written but never wired into run_stage (no s3 branch). My oversight; compile check does not catch missing dispatch.
- Fix: branch added, uploaded, L1 S3 relaunched PID64114 + LF12 S2 PID64115 packed with S3 S1 (3-way, GPU 100%). Lesson: dispatcher needs an integration smoke (launch --help dry-run per stage) before queueing.

## 2026-09-08 12:35 UTC — L1 S3 suspected OOM in 3-pack (PRE-TEST, mitigated)
- Symptom: L1 S3 died mid VAL2-baseline (512 x cap1024) with zero traceback; S3 S1 + LF12 S2 companions alive. dmesg unavailable in container.
- Cause (suspected): 1024-token KV-cache baseline (134MB+/seq + 2.5GB model) packed 3-way with two 256-cap gens exceeded 24GB during peak overlap. No traceback = SIGKILL pattern.
- Fix: VAL2-1024 freeze jobs are SOLO-or-one-light-companion class (rule added). Relaunched L1 S3 PID64342 2-way with S3 S1 only (LF12 S2 waits for S3 S1 slot). Extend box OOM scan to full_*.log in driver v2.

## 2026-09-08 13:10 UTC — CORRECTION: L1 S3 never OOMed; duplicate-launch my error (PRE-TEST)
- Correction to 12:35 entry: original L1 S3 (PID64114) was ALIVE the whole time (silent 512x1024 baseline, no log output during generation — normal). I misread an empty ps section and launched duplicate PID64342 into the same log.
- Fixed by killing the duplicate (28 min in, still in baseline, zero runs logged — nothing duplicated in runs.jsonl). Original continues with clean log ownership.
- Retraction: no evidence of any OOM to date. The 1024-cap escort rule stays as precaution, downgraded from incident to hypothesis. Real lesson: verify death via ps ELAPSED+TIME growth across two ticks before declaring; silent logs during long baselines are EXPECTED.

## 2026-09-10 ~20:25 UTC — preservation died 3x ~12 min (cause: evidence destroyed twice, then fixed blind)
- Pattern: L1 preserve launched 19:46, 19:58, 20:11 — first two vanished with zero trace. Root meta-cause: driver launch() used truncate (>) so each relaunch erased the crash traceback. Fixed to append-only with LAUNCH separators.
- Suspected code cause: load_dataset trust_remote_code=False on script-backed sets (MMLU-Pro/GSM8K) — patched to True for attempt #4+. Attempt #3 (old code) left running as the confirming witness; its traceback, if any, is now preserved.
- Rule: launcher logs append, always; no diagnosis without a traceback.

## 2026-09-10 ~21:00 UTC — preserve killer found: seed_base vs seed0 kwarg mismatch (PRE-TEST, fixed)
- Traceback (preserved thanks to append-logging): gen_texts() got unexpected keyword seed_base at IFEval-ablated call. My wrapper names it seed0; five call sites used the callee convention. Baseline call passed positionally, so death landed minutes in, identically x3.
- trust_remote_code patch stands as harmless insurance (unproven either way).
- Collateral fix in same pass: inner delegation lines briefly flipped the wrong way by replaceAll; corrected before upload (generate_batch/generate_with_hooks_form take seed_base).
- Rule added: new stage scripts get a 60-second import+signature smoke (py_compile is not enough; kwarg mismatches only fire at runtime).

## 2026-09-10 — S3 VAL2 contradicts S2 (+0.38 -> -0.05): selection-bias shrinkage, freeze REVERTED (PRE-TEST)
- S2 top (96 VAL1 prompts, the selection set) does not survive VAL2-512: exact frozen-row recompute R_all -0.054, R_clean -0.055.
- This is the funnel working as designed (VAL2 exists to kill selection winners), not a bug. Freeze rule amended R>0 requirement (TEST unopened: PRE-TEST legal).
- S3 joins LiquidAI as weak/no-freeze. Refusal stratification on same rows: clean==all, contamination ruled out everywhere.

## 2026-09-17 — TEST higher-order recovery used the wrong floor (POST-HOC reporting fix; raw distances unchanged)
- Symptom: TEST R L2-3 values were enormous for four models, and R L2-2 disagreed with the frozen-floor definition. `test_headline.py` computed one TEST unigram human floor, reused it for orders 2/3, and clamped higher-order denominators at `1e-12`.
- Scope: raw L2-1/2/3 distances, JSD, generations, MMD, preservation, and JMQ are unaffected. Only normalized TEST recovery and products that copied it were wrong.
- Fix: downstream products now recompute all three TEST recoveries from raw distances with the order-matched frozen VAL2 floors in `artifacts/metrics/val2_floor.json`. `test_headline.py` now uses those floors and upserts existing run records rather than leaving stale append-only rows.
- Corrected TEST R L2-1/L2-2/L2-3: L1 `+.065/-.041/+.052`; O1 `-.000/+.091/+.478`; Q08 `+.249/+.097/+.258`; Q20 `-.036/-.097/-.156`; G2 `-.009/-.071/-.086`.
- The verified `dump_mirror` remains an immutable snapshot containing the historical values; corrected local combined artifacts and this audit entry are authoritative for reporting.

## 2026-09-17 — Large SCP transfer closed by remote host; MMD moved through private HF dump
- Direct parallel and sequential SCP of the ten embedding parquets was unreliable and left partial local files.
- Resolution: generated a 20-file SHA-256 manifest remotely, uploaded embeddings plus metric/done JSONs to private repo `amkkk/AblationWriting-H2-dump` at commit `9da4909fc4c1ded93b929e336f14aa421ab4931f`, downloaded into a separate local staging directory, verified all 20 hashes (626,543,960 bytes), and only then promoted them.
- Result: 10/10 local embedding parquets have expected row counts (VAL2 1,536; TEST 6,000 per model), and 5/5 metric/done pairs pass hash validation. Partial SCP files were overwritten only by verified HF copies.

## 2026-09-17 — H2-MRSC v1 blocked on archived bit-replay; superseded by contemporaneous-anchor v2
- Symptom: v1 reconstruction gate required exact reproduction of archived sampled text; L1 mismatched 8/8 unhooked baselines and 7/8 cone outputs on the current runtime (driver `595.71.05` vs original `580.173.02`). No control generations were produced; the worker exited cleanly and the v1 plan remained plan-only.
- Diagnosis: baseline mismatch precedes any hook logic, so this is a runtime-identity failure, not hook evidence. Seeded BF16 sampling is not guaranteed bit-identical across drivers/backends.
- Fix: v2 epoch `H2-MRSC-400-v2-contemporaneous-anchors` generates ANCHOR-BASE, ANCHOR-CONE, PC1-DOSE, and RAND-RANK-DOSE together in one runtime and compares only contemporaneous anchors on TEST. Archived replay is retained as a recorded environment diagnostic, and a new same-runtime old/new code-equivalence gate must pass before calibration. Archived VAL2 FULL-SYM stays in a separate secondary stratum.
- v1 L1 plan archived remotely as `controls_plan.v1-blocked-<timestamp>.json`; v2 published to HF commit `362e5ff4b49056b67e56e4124d6115abd9770105` and launched alone as PID 5555.
- v2 then failed dose calibration on L1: per-layer PC1 needed alpha 25.15 at layer 8 (cap 4.0), failing closed with 64/64 calibration prompts complete and no generations written. Diagnosis: a single direction cannot reproduce a 4-direction cone's per-layer removal distribution under the one-sided clamp. Resolution: v3 matches total summed dose with one alpha per arm (same value every layer), approved by user; v1 and v2 epochs stay blocked.
- v3 (total-dose) implemented, self-tested, preflighted 5/5 locally, published to HF commit `375a32edc81ea4d54229b7a2599a30035d5b1d6e` (remote byte-match verified). VAST DNS was down for HF hydration, so the 9 small text files (~200KB) went by direct SCP with all 6 code/doc hashes verified equal post-transfer. v2 L1 plan, reconstruction marker, and calibration progress archived remotely; v3 launched alone as PID 6921.
- 2026-09-17 ~07:2x UTC: v3 L1 worker died at model download — VAST egress broken (DNS resolves, HTTPS `Network is unreachable`). Transient infrastructure, not a plan failure; v3 plan file persists for resume. No relaunch until egress recovers.
- 2026-09-17 08:55 UTC: egress recovered (HTTPS 200); v3 relaunched alone as PID 7388 after 5/5 preflights. Monitoring L1 download/calibration.
- 2026-09-17 ~15:42 UTC: v4 (PC1 dropped, RAND-only) implemented, self-tested, preflighted 5/5, published to HF `b73159ac` (remote byte-match verified), hydrated via HF snapshot, v3 L1 state archived remotely, launched alone as PID 8327. Monitoring L1 equivalence/calibration/generation.
- 2026-09-17 ~21:0x UTC: v4 died merging CPU LEX rows after completing all 1200 L1 TEST generations — `_merge_fixed_rows` cannot merge not-yet-present rows (deterministic code defect, texts safe in progress file). Fix is v5 with a dedicated validated-append merge path plus regression test; v4 L1 progress/anchors adopted after strict hash verification, gates regenerated. No generations lost.
- 2026-09-17 ~22:0x UTC: v5 implemented, self-tested (incl. new merge regression tests), preflighted 5/5, published to HF `b58cf663` (remote byte-match verified), hydrated via HF snapshot, v4 L1 plan/markers archived with progress+anchors kept for adoption, launched alone as PID 9719. Monitoring L1 adoption → equivalence → calibration.
- 2026-09-18 ~06:5x UTC: O1 v5 generation COMPLETE (done `06b3d834...`); pulled 7 finals via staged SCP sync (7/7 byte-exact, load_generation 4048 rows, done-marker self-consistent), promoted locally, stage emptied. Worker on Q08.
- 2026-09-18 ~20:5x UTC: Q08 v5 generation COMPLETE (done `4b5db611...`); pulled 8 finals via staged SCP sync (8/8 byte-exact, 4048 rows, exact arms, v5 suite), promoted + validated locally, stage emptied. Worker on Q20.
- 2026-09-18 ~21:5x UTC: Q20 v5 generation COMPLETE (done `205b645f...`); pulled 8 finals via staged SCP sync (8/8 byte-exact, 4048 rows, exact arms, v5 suite), promoted + validated locally, stage emptied. Worker on G2. Remote disk at 100% (77M free) during G2 load: 14G embedder (keep) + 9.6G G2 weights (in use) leave no safe frees; generation-phase writes are MB-scale and should fit, embed phase re-evaluated later.
- 2026-09-17 ~21:5x UTC: L1 control generation COMPLETE under v5 (adoption + fixed lex merge worked; done marker sha `673e9e43...`). Worker advanced to O1 (dose calibration underway). Disk at 6.9G free — below the 8GiB stage gate but the running stage already passed its check; watching.
- 2026-09-17 ~19:30 UTC: O1+Q08 packing experiment staged (`src/launch_controls_pack.sh`, remote `bash -n` clean). Same v4 epoch/code/seeds, two processes sharing one GPU with independent RNG (outputs identical to serial by construction). Shared offline prefetch cache, per-model logs, 16GiB VRAM + 6GiB disk gates, OOM/failure kills survivor and falls back to serial resume. Q20/G2/embedder stay solo-class. Swap window: after L1 done-marker, before O1 progress.
- 2026-09-17 ~09:1x UTC: v3 failed closed on L1 too — total-dose PC1 alpha 16.83 > 4.0 cap, with equivalence gate passed and 64/64 calibration complete. The nested top direction carries only ~1/17 of the frozen cone's positive-part removal norm: per-layer matching was not the problem, PC1 is simply weak under the one-sided clamp. RAND total-dose value was not reached (PC1 raises first). No generations written; v3 epoch blocked pending a scope decision (drop PC1 vs new matching statistic vs report infeasible).
