# Results

`octvla_results.tar.xz` holds every rollout and diagnostic this project scored:
the `eval/` and `diagnostics/` directories of `$OCTVLA_OUTPUT_ROOT` as of
2026-09-30, 731 JSON files (25 MB unpacked). Every number in
`docs/research_questions.md` is re-derived from them. The cluster copies
are gone; the checkpoints and the full per-episode videos were not kept.
`docs/videos/` holds the selected clips.

```bash
mkdir -p /tmp/octvla && tar -C /tmp/octvla -xJf results/octvla_results.tar.xz
PYTHONPATH=src python scripts/collect_final_results.py --prefix GF /tmp/octvla/eval
PYTHONPATH=src python scripts/paired_compare.py /tmp/octvla/eval GF-rgb_cont GF-kv_adaln
```

Both scripts need only the standard library.

- **One file per rollout**, named `<PREFIX>-<arm>-s<training seed>-<job>.json`:
  - `job` is `seen`, `heldout`, `novel`, `count` (2 and 4 objects) or `four`.
  - Prefixes are listed in `docs/rollout_videos.md` and `research_questions.md`.
- **Contents:** each file records the checkpoint, the identity pin and one entry
  per episode, with its per-object outcomes.
- **Subdirectories:** `vshift/` (visual shift), `noise/<level>/` (entity noise),
  `exec5/` and `exec8/` (GR00T execution horizon).
- **Kept but not counted:**
  - The `superseded_*/` and `void_*/` rollouts. They are kept for provenance and
    the collector never reads them.
  - Four-object episodes inside `*-count.json`, which ran 600 steps instead of
    800 (`research_questions.md` §1). The collector sets them aside; the
    `*-four.json` files replace them.
- **Video paths** inside the episode records point at the cluster staging area,
  which no longer exists.
