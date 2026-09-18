# Where the training datasets live, and what is in them

Every training job reads a LeRobot v3 dataset directory. This is where those
directories are, which files describe them, and which fields matter.

## Roots

Three roots, one per control-space generation. They are separate because the
earlier sweeps are still referenced by results and must not be overwritten. Each
is selected by sourcing an environment overlay before submitting.

| overlay | `OCTVLA_DATASET_ROOT` | holds |
| --- | --- | --- |
| `~/octvla/env.sh` | `~/octvla/datasets` | EE-delta (`cartesian`) |
| `~/octvla/env-jointspace.sh` | `$HPCVAULT/octvla-datasets-jointspace` | absolute joint |
| `~/octvla/env-jointdelta.sh` | `$HPCVAULT/octvla-datasets-jointdelta` | joint delta, and every ACT cell |

`$HPCVAULT` is `/home/vault/g107ea/g107ea12`. The canonical recordings the
datasets are exported *from* live at `$OCTVLA_COLLECTION_ROOT`
(`$HPCVAULT/octvla-collection-joints/three_object` for the current corpus) —
those are the raw clips, not a training input.

## Files in a dataset directory

```
<root>/<dataset>/
  meta/info.json          <- schema, fps, control space, feature shapes
  meta/stats.json         <- per-column mean/std/min/max/q01/q99 (the normalisation)
  meta/episodes/...       <- per-episode index and length
  meta/tasks.parquet      <- the instruction strings
  data/                   <- observation.state, action, indices (parquet)
  videos/                 <- encoded camera streams (absent for privileged)
  split_manifest.json     <- train/val boundary and the eval_split it implies
  octvla_episode_manifest.json  <- which canonical clip each episode came from
```

### `meta/info.json` — the schema

The two fields this project added, both read back at evaluation time:

- **`control_space`** — `cartesian`, `cartesian_absolute`, `joint` or
  `joint_delta`. Absolute and incremental joint actions have identical column
  layouts, so without this field executing one as the other is silently wrong;
  evaluation refuses a joint-shaped dataset that lacks it.
- **`state_encoding`** — `position` or `position_velocity`. The latter doubles
  `observation.state` by appending the backward difference.

Everything else is LeRobot's: `fps` (15), `robot_type`, `total_episodes`,
`total_frames`, and `features` with each column's dtype and shape.

### `meta/stats.json` — the normalisation

This is what makes losses comparable or not. ACT normalises STATE/ACTION with
MEAN_STD, pi0.5 with QUANTILES (`2·(x−q01)/(q99−q01)−1`). **A normalised loss
cannot be compared across control spaces**: absolute joint actions have roughly
13× the scale of increments, so the same physical error reads as a smaller
number. Convert to radians using this file before comparing.

### `split_manifest.json` — the split

`eval_split` is derived, not chosen. LeRobot holds out the last
`ceil(n · eval_split)` episodes positionally, so the exporter writes train
episodes first and computes the fraction that makes that positional rule select
exactly the reserved validation seeds. Training reads it from here rather than
restating it.

## Current datasets

All on the r75 corpus: **75 train runs / 225 atomic clips, 26 val runs / 78
clips, 42 777 frames, 15 fps**, `eval_split` 0.2557755775577558.

Under `$HPCVAULT/octvla-datasets-jointdelta/`:

| dataset | control_space | state | action | cameras |
| --- | --- | --- | --- | --- |
| `three_object_rgb_r75` | joint_delta | 16 | 16 | 3 |
| `three_object_rgb_r75_abs` | joint | 16 | 16 | 3 |
| `three_object_rgb_r75_velo` | joint_delta | **32** | 16 | 3 |
| `three_object_rgb_r75_eedelta` | cartesian | 16 | **14** | 3 |
| `three_object_rgb_r75_eeabs` | cartesian_absolute | 16 | 16 | 3 |
| `three_object_privileged_r75` | joint_delta | 16 | 16 | none |
| `three_object_privileged_r75_abs` | joint | 16 | 16 | none |
| `three_object_privileged_r75_velo` | joint_delta | **32** | 16 | none |

Privileged variants carry `observation.environment_state` (120-d = 8 objects ×
15-d tokens) and no cameras. Object-conditioned variants additionally carry
`observation.object_tokens` (8×15), `observation.object_token_mask` (8) and
`observation.object_token_rank` (8).

Column layouts:

- **joint** (16): `left_arm.j0…j6`, `left_arm.gripper`, then the same for right.
- **joint + velocity** (32): the above, then each column again suffixed `.vel`.
- **EE pose** (16): `left_eef.{x,y,z,qx,qy,qz,qw,gripper}`, then right.
- **EE increment** (14): 3 translation + 3 rotation + gripper, per arm.

## Rebuilding one

```bash
source ~/octvla/env-jointdelta.sh
sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION --gres=gpu:a40:1 \
  --export=ALL,DATASET_NAME=three_object_rgb_r75_abs,CONTROL_SPACE=joint,STATE_ENCODING=position,PRIVILEGED=0 \
  slurm/build_shelf_restock_splits.sbatch
```

`PRIVILEGED=1` drops the cameras and adds `environment_state` (minutes instead
of about an hour, since nothing is video-encoded). The job refuses to overwrite
an existing directory.

## Training against one

```bash
source ~/octvla/env-jointdelta.sh
sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION --gres=gpu:a40:1 \
  --export=ALL,ACT_DATASET=three_object_rgb_r75_abs,ACT_CONDITIONING=rgb,\
TRAIN_STEPS=40000,BATCH_SIZE=8,CHUNK_SIZE=50,N_ACTION_STEPS=50,TRAIN_SEED=1000,SAVE_FREQ=2000,EVAL_STEPS=1000 \
  slurm/train_act_shelf_restock.sbatch
```

The run writes `$OCTVLA_OUTPUT_ROOT/act_<dataset>_s<seed>/`, with
`train_config.json` and `config.json` under each checkpoint recording exactly
what was used. `scripts/select_best_checkpoint.py` then points
`checkpoints/best` at the lowest-validation-loss step, and evaluation prefers it
over `last`.

Evaluation must be pointed at the same dataset via `EVAL_DATASET`, because the
action width, the control space and the state encoding all come from it.
