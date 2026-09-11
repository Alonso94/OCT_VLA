# Slurm data collection

Collection runs in the RoboTwin Python environment on one GPU per scene seed.
Each array task writes only `canonical/<profile>/seed_<seed>/`, avoiding shared
writer state. A separate dependent job exports all successful canonical clips
into paired RGB and object-token LeRobot datasets.

Profiles are `two_object`, `three_object` and `four_object`; they share one
geometry and differ only in object count. Run one feasibility seed of the
profile you are collecting first. It creates canonical data and, with
`--finalize`, LeRobot videos for review:

```bash
export SLURM_ACCOUNT=YOUR_ACCOUNT
export SLURM_PARTITION=YOUR_A100_PARTITION
export OCTVLA_REPO=/path/to/OCT_VLA
export OCTVLA_ROBOTWIN_ROOT=/path/to/RoboTwin
export OCTVLA_ROBOTWIN_PYTHON=/path/to/RoboTwin/.venv-robotwin/bin/python
export OCTVLA_POLICY_PYTHON=/path/to/policy-venv/bin/python
export OCTVLA_COLLECTION_ROOT=/scratch/$USER/octvla-collection
export OCTVLA_DATASET_ROOT=/scratch/$USER/octvla-datasets
./slurm/submit_shelf_restock_collection.sh three_object a100 1000 --finalize
```

After reviewing the resulting video, submit a seed range, for example
`1000-1009`. A discarded seed is recorded in that seed directory and contributes
no clip; expect to lose a fraction of seeds this way (a five-seed three-object
sample yielded three) and reserve more seeds than the split table needs rather
than retrying a failed one. Do not overwrite a finalized dataset root. The
finalizer must use the LeRobot policy environment, while collection must use
RoboTwin's Python 3.10 environment.

## Full collection run

Reserved seed blocks and the split each belongs to are in
[dataset_protocol.md](dataset_protocol.md#reserved-seed-ranges). Submit one
job per (profile, split) block. The chunks below are roughly twice each
split's target, which covers the discard rate seen so far; if a split comes up
short, extend into the unused tail of its own reserved block rather than
re-running a discarded seed.

```bash
# three_object -- train (target 30), IID validation (10), final test (20)
./slurm/submit_shelf_restock_collection.sh three_object a100 100-159
./slurm/submit_shelf_restock_collection.sh three_object a100 200-219
./slurm/submit_shelf_restock_collection.sh three_object a100 250-289

# two_object -- count-shift validation (10), final test (20)
./slurm/submit_shelf_restock_collection.sh two_object a100 400-419
./slurm/submit_shelf_restock_collection.sh two_object a100 450-489

# four_object -- count-shift validation (10), final test (20)
./slurm/submit_shelf_restock_collection.sh four_object a100 600-619
./slurm/submit_shelf_restock_collection.sh four_object a100 650-689
```

Pass `--finalize` on the *last* submission for a profile, not on every one: the
finalizer exports a whole profile's canonical clips in one pass, and it refuses
to overwrite an existing dataset root. Splitting a profile's clips into train,
validation and test happens afterwards, from the seed recorded for each episode
in `octvla_episode_manifest.json` -- collection itself is split-agnostic.

Check yield per block before finalizing:

```bash
grep -l '"status": "ok"' $OCTVLA_COLLECTION_ROOT/three_object/seed_*/collection_report.json | wc -l
```
