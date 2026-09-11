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
