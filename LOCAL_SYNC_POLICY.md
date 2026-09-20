# Local-ahead dump synchronization policy

Local JMQ judgments, statistics, figure data, paper edits, and recovery code are newer than the staged HF dump.

While `.local_ahead.json` exists:

1. Never run `hf download amkkk/AblationWriting-H2-dump --local-dir dump_mirror`.
2. Download a newer dump into a new sibling such as `dump_incoming_YYYYMMDDTHHMMSSZ`.
3. Validate the incoming `dump_manifest.json`, every declared file size, and source hashes where available.
4. Compare incoming files with the authoritative local paths listed in `.local_ahead.json`.
5. Do not copy incoming versions of those paths over the project root. Local wins until it has been copied to VAST, staged to HF, downloaded back, and hash-verified.
6. Preserve the current `dump_mirror` until the incoming snapshot and all local-ahead artifacts are verified.
7. Remove `.local_ahead.json` only after the refreshed HF dump contains byte-identical copies of every listed local artifact.

`dump_mirror` is a remote snapshot, not the writable source of truth. The project-root files are authoritative while local is ahead.
