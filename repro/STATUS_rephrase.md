# Rephrase status: before and after

## Model-card rephrase (done)
Before: staff-written v3 cards. After: Q20-cone rewrote 10 sections, 6 accepted
(L1a, O1a, O1i, Q08i, Q20a, G2i), 2 rejected (meaning-flip, spelled-out digits),
1 weakened-out, 1 verbatim copy. Live on all five repos as v4. Jobs, outputs,
and verifier committed under rephrase_jobs.json, rephrase_outs.json.

## Manuscript rephrase (in progress, 0 of 39 generated)
Before: staff-written sections (this manuscript). After: pending.
Pipeline moved from VAST to local RTX 5070 after VAST egress flapped three
times mid-download. Local: CUDA torch 2.11.0+cu128 verified, Q20 base pinned
revision downloaded, 5 basis files fetched, hooks built by repo code,
sequential run with per-job checkpoints to rephrase_paper_outs.json.
Verification per paragraph (numbers, cites, refs) before any replacement.

## Appendix panels (done)
Before: two giant longtables with 900-char cells, caption colliding with the
header, page-break gaps swallowing rows. After: four HIP-style panels with
score headers, 200-char cells, small type, verified by screenshot
(repro/shot_p19.png): two panels per page, compact, readable.