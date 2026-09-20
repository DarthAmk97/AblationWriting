# H2 Protocol — Lab Notebook (chronological, PRE-TEST vs POST-HOC)
# Project: The Geometry of Machine Writing: Training-Free Selective Ablation of Post-Training Writing Collapse
# Hypothesis H2: low-rank multi-directional concept cone mediates AI-writing phenotype
# Location: /root/AblationWriting ONLY (isolated from /workspace syncthing copy)
# Instance: RTX 3090 24GB, CUDA 12.8, driver 580.173.02

## 2026-09-07 01:44 UTC — PRE-TEST — Instance orientation
- Verified SSH via vast_ed25519_20260907, created /root/AblationWriting.
- workspace_is_volume=false → nothing survives recycle; must push valuable work to HF Hub.
- /workspace/.hf_home blobs incomplete (0 blobs, only refs) → syncthing copy in progress from 136.0.41.217:22067. Isolated HF cache at /root/AblationWriting/.hf_cache.
- GPU idle, 564G free, /venv/main has no torch → launching isolated install.

## 2026-09-07 02:46 UTC — PRE-TEST — Model validation
- All 8 panel repos exist. SHAs recorded in artifacts/model_validation.json.
- Q08/Q20 (Qwen3.5) + G2/G4 (Gemma-4) are multimodal ForConditionalGeneration (vision+language, hybrid). Intervention will target language backbone residual stream only; text-only prompts. Not excluded — log as architectural caveat.
- L1 Llama-3.2-1B-Instruct gated 403 with provided HF token. Smoke switches from Q08+L1 to Q08+O1. L1 exclusion pending access request; must log before TEST per Sec 2.
- O1 (OLMo2 16L/2048d), LF12 (LFM2 16L/2048d), S3 (SmolLM3 36L/2048d) text-only, ideal for smoke.
- GEN sha a0f143c1..., PAIRED sha dacfc1bc..., EMBED sha aa3b43a4... confirmed.
- Rank-1 retained only as appendix negative control; discovery starts at H2.

## 2026-09-07 — PRE-TEST — Install
- torch cu128 + transformers/datasets/accelerate install in background, log logs/install_nohup.log
- Qwen3.5 needs transformers>=4.55 / main; will upgrade if AutoModel fails.

## TODO PRE-TEST
- [x] Inspect GEN schema, freeze splits (CAL 2048 / VAL1 256 / VAL2 512 / TEST 2000 / TEST-JMQ 400 / INTERP 128 / external 1000 / storyscope 100)
- [x] Freeze sampler temp 0.8 top-p 0.95 top-k 50, seed per prompt, cap 1024
- [x] Freeze eval tokenizer Qwen3.5-0.8B revision; select each MMD bandwidth from VAL2 human+baseline only
- [x] Smoke Q08+O1 512 CAL 128 VAL ranks {2,4,8} 2 windows alpha {0.5,1.0} rho 0.75
- [x] Record kappa (throughput multiplier), update compute_budget.md from 100-prompt smoke
- [x] Freeze H2 config per model before TEST; never touch TEST during selection
- [x] JMQ via opencode-go muse-spark-1.3-contributor; save raw judgments + request IDs
- [x] Preservation gates: IFEval 95%, capability 97%, refusal 95%, pathology +2pp
- [x] H3 deferred entirely; partial work marked awaiting-H3

## 2026-09-09 — Discussion record (paper detail, user dialogue)
- R% defined for methods: (d_base-human - d_abl-human)/(d_base-human - d_human-human); phenotype = AI-voice distributional habits (tics, scaffolding, headers, preambles), NOT quality (JMQ), depth (StoryScope), safety (gates).
- Qwen3.5-2B keeps full sweep despite flat smoke: anti-cherry-picking (panel scored 6/8, missing != negative); Qwen3.5-0.8B precedent (smoke flat -5.5% to full late +20.6%); H4-grade negatives only from full sweeps. VAL2-freeze skips ONLY on uniformly-negative full evidence (LiquidAI/LFM2-1.2B rule: 62/62 <=0).
- Llama-3.2-1B frozen late-rank-4 (4/4 finalists gated-pass, cone>full 4/4); SmolLM3-3B S2 best +37.7% rank-32; OLMo weak (+1.5%, VAL2 deciding); LiquidAI weak certified, VAL2 skipped.
- Four additions dropped for timeline (multi-pass, winsorization, KL-rollback, bias check); OBLITERATUS comparison mapped (mirror/forbidden/parked).
- Ordering principle: GPU chain first on VAST, local chain dump-fed; figure-data incremental; StoryScope extraction local.
- Llama-3.2-1B license accepted by user (was gated 403); JMQ endpoint resolved to Zen responses + UA header.

## 2026-09-09 — Phenotype-vs-tokens note (paper-bound, user dialogue)
- H2 ablation operates on tracked human-AI distance metrics; n-gram shifts are byproduct + readout, never the target.
- Byproduct evidence: SmolLM3-3B moves L2-1/2/3 + JSD together (drivers: commas, articles, line breaks = surface voice); Llama-3.2-1B moves unigrams+JSD with bigrams flat (dissociation a word-ban cannot produce).
- Explicit word-ban is a DIFFERENT claim (output cosmetics; protocol: fooling surface stats establishes nothing). Adopted as CONTROL arm: top-k contributor suppression at matched shift on VAL2 frozen models.
- Length audit: abl length ~= base << human (225 vs 734 VAL1; 481 vs 900 VAL2) — no length confound in R, but length gap itself is an unablated phenotype dimension, recorded.
- Qwen3.5-2B JSD/L2 divergence (JSD improves while L2 worsens) held for MMD adjudication, not averaged away.

## 2026-09-17 — POST-TEST — MMD completed, transferred through HF, and integrated
- Pinned embedder/revision: `nvidia/llama-embed-nemotron-8b` at `aa3b43a495a9b280d1bdb716da37c54bb495d630`; BF16 inference, attention-mask mean pooling in FP32, L2 normalization, max length 4096.
- Ten embedding parquets completed on the reconstructed RTX 3090 instance. Each model has 1,536 VAL2 rows and 6,000 TEST rows (human, baseline, ablated per prompt).
- TEST MMD recovery: L1 `+.108`, O1 `+.214`, Q08 `+.022`, Q20 `+.240`, G2 `+.134`. VAL2 recovery: `+.167/+.295/+.024/+.421/+.163` in the same order.
- MMD resolves the earlier JSD/L2 disagreement only in the narrow sense that global embedding movement is positive for Q20; it does not make token metrics wrong. Q08 shows the reverse pattern: strong token recovery with almost no embedding recovery. The affected features are model-dependent.
- Large SCP was unreliable. The authoritative transfer used private HF dump commit `9da4909fc4c1ded93b929e336f14aa421ab4931f`, a separate local staging directory, and a 20-file SHA-256 manifest before promotion.
- A post-hoc audit found TEST higher-order R had reused a unigram floor. Reporting now recomputes all TEST L2 recoveries with frozen order-matched VAL2 floors; raw distances and all generated text remain unchanged. See `failures.md`.

## 2026-09-17 — POST-TEST — H2-MRSC-400 controls frozen
- This mechanistic suite was specified after TEST/MMD/JMQ were known and is labeled post-hoc. It narrows inference to the five frozen headline models; it cannot restore the original universal eight-model or external-generalization criterion.
- New GPU generation is limited to two TEST-JMQ-400 arms: the first nested frozen basis direction and a matched-rank keyed random subspace. Both are dose-matched per layer on 64 fixed VAL1 prompts; alpha above 4 fails rather than clips.
- Baseline, frozen cone, and symmetric full-subspace outputs are reused. A deterministic unigram-deletion comparator is calibrated only on frozen-config VAL1-96 and applied unchanged to TEST-JMQ.
- Launch budget is 28–30 nominal GPU-hours, 36 hard-envelope. Before any new control output, the reconstructed model/hook must exactly reproduce archived baseline and cone hashes for eight fixed prompts.
- Canonical specification: `CONTROL_PROTOCOL.md`; implementation: `src/h2_controls.py`, `src/control_embed.py`, `src/control_stats.py`, `src/launch_controls.sh`.
