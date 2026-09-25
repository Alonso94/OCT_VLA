# Reproducibility: environments, cluster, storage, pipeline

How to run everything, from collecting data to publishing it. The data
contracts that each step enforces are in `docs/data_protocol.md`. Paths are
repo-relative unless they start with `$`.

## 1. Environments

The simulator and the policy **cannot share an interpreter**. The simulator
needs NumPy 1.x and Torch 2.4; the policy needs Torch 2.11 and LeRobot. So
evaluation runs a simulator server and a policy client, connected by a socket
(`src/oct_vla/serve/`). Only `protocol.py` is imported on both sides, and it
uses the standard library only.

| | `$OCTVLA_ROBOTWIN_PYTHON` | `$OCTVLA_POLICY_PYTHON` |
| --- | --- | --- |
| path | `~/octvla/RoboTwin/.venv-robotwin/bin/python` | `~/octvla/policy-venv/bin/python` |
| Python | 3.10.20 | 3.12.13 |
| key packages | SAPIEN 3.0.0b1, Torch 2.4.1+cu121, NumPy 1.26.4, cuRobo `d64c4b0` (v0.7.8), mplib 0.2.1 (patched), setuptools 69.5.1 | LeRobot 0.6.2 @ `6adf515` (from git, not PyPI), Torch 2.11.0+cu128, transformers 5.5.4, PEFT 0.20.0 |
| runs | collection, the eval server, bridge and replay checks | export, views, training, merge, init check, eval client, diagnostics, tests |

RoboTwin is at revision `cffb78057724c6899fbae7b71ec4cfb249fbc225`. Installation
steps and the two upstream patches are in `docs/reproducibility.md`. The core
package (`src/oct_vla`) has no runtime dependencies (`dependencies = []`) and
runs on Python 3.10–3.12.

**Environment files** live outside the repo, in `~/octvla/`. Source one before
anything else:

```bash
source ~/octvla/env-leftmost.sh   # sources env.sh, then points the three roots at the current corpus
```

| variable | value under `env-leftmost.sh` |
| --- | --- |
| `OCTVLA_REPO` | `$HOME/OCT_VLA`. Override it to run from another checkout or worktree |
| `OCTVLA_ROBOTWIN_ROOT` | `$HOME/octvla/RoboTwin` (its assets are symlinked from `$HPCVAULT/RoboTwin-assets`) |
| `OCTVLA_COLLECTION_ROOT` | `$HPCVAULT/octvla-collection-leftmost` (canonical clips) |
| `OCTVLA_DATASET_ROOT` | `$HPCVAULT/octvla-datasets-leftmost` (LeRobot datasets) |
| `OCTVLA_OUTPUT_ROOT` | `$HPCVAULT/octvla-outputs-leftmost` (runs, `eval/`, `diagnostics/`) |
| `HF_HOME` | `$HPCVAULT/huggingface`. Jobs run with `HF_HUB_OFFLINE=1`, so cache base models here first |
| `OCTVLA_TORCHCODEC_LIBS` | `~/octvla/ffmpeg-soname`. Policy-side jobs only (see §2) |
| `SLURM_ACCOUNT`, `SLURM_PARTITION` | `g107ea`, `a40` |

`env.sh`, `env-jointspace.sh` and `env-jointdelta.sh` point at older corpora.
Results from those corpora are not comparable with current ones.

**Base models** to cache in `$HF_HOME`: `lerobot/pi05_base`, `lerobot/smolvla_base`,
`nvidia/GR00T-N1.7-3B` (named by config, not by path), and `lerobot/VLA-JEPA-Pretrain`.
pi0.5 also loads the licence-gated tokenizer `google/paligemma-3b-pt-224`. Your
Hugging Face account must have accepted that licence, and the token must be in
`$HF_HOME/token`.

**transformers is a silent trap.** PEFT does not cap the transformers version.
If a newer release gets installed, pi0.5's vision tower fails to load with only
a warning, and training still produces a falling loss. Install transformers
after PEFT, then check each training log:
`grep -c "Could not load state dict" <log>` must print 0.

**Tests.** Run them with the policy interpreter. A bare `python` skips every
policy test and still reports success.

```bash
PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" -m pytest tests/ -q
```

`PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" -B -m oct_vla doctor --robotwin-root … --robotwin-python … --policy-python …`
reports which dependencies it finds. It does not check that a GPU run works.

## 2. Cluster: NHR@FAU Alex

| | |
| --- | --- |
| partitions | `a40` (default, 44 nodes × 8 A40), `a100` (37 nodes), `rtxpro6k` (11), `a100mig` (1) |
| wall time | 24 h maximum on every partition |
| job shape | one node per job at most |
| GPU | `--gres=gpu:<type>:1` is **mandatory on every job**, CPU-only work included |
| memory | never pass `--mem`; GPU jobs are rejected with it (memory comes with the GPU) |
| login nodes | `alex1`, `alex2`. **No GPU**, so anything that imports SAPIEN or cuRobo needs a batch job |
| `/tmp` | separate on each login node (alex1 and alex2 differ). Keep durable work on the vault or in git |
| `$TMPDIR` in a job | node-local, discarded when the job ends (the smoke test uses it) |
| array size | `MaxArraySize` 10000, but reserved seeds stay below 1000 for portability |

**torchcodec and `LD_LIBRARY_PATH`.** The cluster has no system FFmpeg, so
torchcodec cannot load and LeRobot silently falls back to PyAV. Decoding then
drops from 21.6 to 204.8 ms per sample, and the dataloader becomes the
bottleneck. `$OCTVLA_TORCHCODEC_LIBS` holds plain-SONAME symlinks onto the
FFmpeg libraries that PyAV bundles.

The training, build and diagnose jobs prepend that directory to
`LD_LIBRARY_PATH`. **Never expose it to SAPIEN.** The libraries shadow SAPIEN's
own `libxcb`/`libXau`, and `import sapien.core` fails. The error surfaces only
as "Could not load RoboTwin task". For this reason `slurm/eval_shelf_restock.sbatch`
and the replay jobs run `unset LD_LIBRARY_PATH`, and the collection jobs never
set it. Do not export it from an env file.

**Vulkan.** SAPIEN prints `Failed to find Vulkan ICD file`. The warning is
harmless: SAPIEN falls back to its bundled ICD and renders on the GPU.

## 3. Storage budget

Storage is the binding constraint. To check current usage, run `shownicerquota.pl`.

| filesystem | quota | rule |
| --- | --- | --- |
| `$HOME` (`/home/hpc`) | ~105 GB soft, 500 k files | Holds source and both virtualenvs only. It allocates **32 MB per file**, so many small files (such as videos) cost far more than their size |
| `$HPCVAULT` (`/home/vault`) | ~1 TB soft (1048.6 GB), 200 k files | Holds everything else: assets, HF cache, collection, datasets, outputs, videos |
| `$WORK` | — | Not used (the group inode quota is nearly full) |

Rules the scripts already enforce. Keep them when you add a new path that saves
data.

- **Final checkpoint only.** `--save_freq` defaults to the step budget, and
  every evaluation reads `checkpoints/last`.
- **Optimiser state dropped.** After `srun` succeeds, both training sbatch files
  run `rm -rf $OUTPUT/checkpoints/*/training_state`. A run cannot be resumed,
  and none is ever resumed. An ACT run is about 0.4 GB.
- **Slim GR00T.** `slim_groot` saves only the trained head and a
  `slim_checkpoint.json` marker. The frozen backbone (6.1 GB) is rebuilt from
  the base model on load; it was measured bit-identical to the base. A stock
  save was about 49 GB per checkpoint.
- **Merged pi0.5/SmolVLA checkpoints.** These are full-weight stage-1
  checkpoints, written only for the control regime that goes on to stage 2
  (`REGIME_<backbone>`). They are saved to `<stage-1 run>/merged/`.
- **Videos on the vault.** Rollout videos go under
  `$HPCVAULT/octvla-rollout-videos/`, never into the repo or home. The repo
  keeps only `docs/rollout_videos.md`.
- **Hard-linked views.** Dataset views and sibling exports share video files by
  inode (`project_dataset_view.py`, `dedupe_dataset_videos.py`).
- **Existing outputs are not overwritten.** Every builder, view and training job
  refuses an output path that already exists. **Ask before deleting a
  checkpoint or dataset.** Some are the only provenance behind a reported
  number.

## 4. Pipeline, end to end

Run everything from the repo root, after `source ~/octvla/env-leftmost.sh`.
`SB` below stands for
`--account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION --gres=gpu:a40:1`.

**1. Collect** (RoboTwin env, one array task per seed):

```bash
./slurm/submit_shelf_restock_collection.sh three_object a40 100-159   # train block
grep -l '"status": "ok"' $OCTVLA_COLLECTION_ROOT/three_object/seed_*/collection_report.json | wc -l
# RoboTwin built-in (array index = seed):
sbatch $SB --array=0-249 --export=ALL,BUILTIN_COLLECTION_ROOT=$HPCVAULT/octvla-collection-builtin \
  slurm/collect_robotwin_builtin_array.sbatch
```

Before collecting a new asset or task at scale, collect a few seeds and inspect
them (`scripts/probe_robotwin_builtin.py`). Passing a geometry screen does not
mean the oracle can plan a grasp.

**2. Build the unified export** (policy env; it encodes video, which takes about 1 h):

```bash
sbatch $SB --export=ALL,DATASET_NAME=three_object_identity_entity,CONTROL_SPACE=unified,\
GRIPPER_ENCODING=binary_command,ENTITY_TOKENS=1,MAX_ENTITIES=16,IDENTITY_HOLDOUT=2 \
  slurm/build_shelf_restock_splits.sbatch
```

The sbatch defaults are `CONTROL_SPACE=joint_delta` and
`GRIPPER_ENCODING=measured_aperture`. **Always set both explicitly.** Other
knobs: `COLLECTION_PROFILE`, `MAX_RUNS`, `EPISODE_KIND`, and `DEDUPE_VIDEO=0`,
which turns off the hard-linking pass. The built-in task is exported with
`scripts/export_robotwin_builtin.py --collection-root … --output … --repo-id …`
(always unified, binary gripper, `021_cup` held out).

**3. Project a view** (parquet only; takes minutes):

```bash
PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" scripts/project_dataset_view.py \
  --unified "$OCTVLA_DATASET_ROOT/three_object_identity_entity" \
  --output "$OCTVLA_DATASET_ROOT/three_object_identity_eeabs" --control-space cartesian_absolute
"$OCTVLA_POLICY_PYTHON" scripts/entity_training_args.py "$OCTVLA_DATASET_ROOT/three_object_identity_eeabs"  # the gate
```

**4a. Train ACT** (stage 1, stage 2 and rollouts are queued with dependencies):

```bash
DRY_RUN=1 slurm/submit_final_experiments.sh     # print the plan first
slurm/submit_final_experiments.sh               # PREFIX=F, the eeabs view, all six arms, seeds 1000-1002
PREFIX=J FINAL_DATASET=three_object_identity_abs ARMS=rgb slurm/submit_final_experiments.sh
```

- Selection variables: `ARMS`, `SEEDS`, `PREFIX`, `FINAL_DATASET`,
  `SKIP_EVAL=1`, `AFTER_JOB`, `VIDEO_STAGE`.
- A cell whose `checkpoints/last/pretrained_model` already exists is reused. A
  run directory that exists without that checkpoint stops the submission.
- ACT settings: 40 k steps, batch 8, chunk 50, `n_action_steps=25`, 4 h limit
  (`slurm/train_act_shelf_restock.sbatch`).
- The stage-2 arms and `rgb_cont` start from the same seed's `rgb`
  checkpoint (`ACT_PRETRAINED`). `CONDITIONING=kv|kv_adaln|kv_tokens` selects
  the arm. `scratch_kv` leaves `ACT_PRETRAINED` unset.

**4b. Train VLAs** (`slurm/train_shelf_restock.sbatch`; defaults 8 k steps,
batch 16, 24 h limit):

```bash
PHASE=stage1 slurm/submit_vla_experiments.sh           # rgb on F and J, 3 backbones x 3 seeds, seen rollout
PHASE=stage2 BACKBONES=groot REGIME_groot=F \
  ARMS="rgb_cont kv kv_adaln kv_tokens" slurm/submit_vla_experiments.sh
```

- `CORPUS` selects the training data: `paired_full` (continuous runs, the
  default) or `identity` (the September atomic clips).
- `BACKBONES` defaults to `pi05 smolvla groot` in stage 1. **Stage 2 requires it
  explicitly**: it is only meaningful on a backbone whose RGB stage 1 does the
  task, which on the current corpus is GR00T alone.
- `ROLLOUTS` defaults to `seen heldout` in stage 2: held-out identity is the
  scope where conditioning beat the budget control in ACT. Add `count` to test
  object count.
- `vla_jepa` is accepted by the training sbatch but not queued by default.

In stage 2, the following steps are queued for each backbone and seed:

- **5. Merge** (pi0.5 and SmolVLA only). `scripts/merge_stage1_adapter.py`
  merges the stage-1 LoRA adapter into its base and writes `<run>/merged/`.
  It refuses to write if the adapter is zero, if the merged model predicts a
  different chunk than the adapter model, or if a reload differs. GR00T stage 1
  is full weights and loads directly.
- **6. Init check** (`INIT_CHECK=1`, `scripts/check_stage2_init.py`). This
  builds the stage-2 run exactly as training would, then checks two things on
  one real batch. First, it must reproduce stage 1's predicted chunk: within
  a relative change of 1e-2 (bf16 noise) for `rgb_cont`, `kv` and
  `kv_adaln`. `kv_tokens` is not identity at init, so it fails only above a
  relative change of 1.0. Second, one backward pass must put non-zero
  gradient into the KV layers and into the arm's own branch (AdaLN or
  in-context). The report is written to
  `run_metadata/<run>_init_check.json`. The training job depends on this job
  (`afterok`).

**7. Roll out** (both interpreters in one job; the submitters queue these):

```bash
EVAL_CHECKPOINT=$OCTVLA_OUTPUT_ROOT/<run>/checkpoints/last/pretrained_model \
EVAL_DATASET=three_object_identity_eeabs EVAL_SEEDS=800-819 EVAL_PROFILES=three_object \
EVAL_MAX_STEPS=600 EVAL_N_ACTION_STEPS=25 EVAL_MODEL_IDS=1,2,3,4 EVAL_TAG=F-kv-s1000-seen \
  sbatch $SB --time=04:00:00 --export=ALL slurm/eval_shelf_restock.sbatch
```

- For the count rollout, use `EVAL_PROFILES=two_object,four_object` with
  `EVAL_STEPS_PER_OBJECT=200`.
- `EVAL_VIDEO_DIR` records every episode.
- The job waits on the server's ready file and kills the server on exit. The
  port is derived from the job id.

**8. Collect results:**

```bash
PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" scripts/collect_final_results.py "$OCTVLA_OUTPUT_ROOT/eval" \
  --prefix F --output "$OCTVLA_OUTPUT_ROOT/diagnostics/final_matrix_F.json"
```

**9. Pick videos** (by rule: the median training seed, then the most typical
episode; exits non-zero if a row has none):

```bash
PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" scripts/select_rollout_videos.py "$OCTVLA_OUTPUT_ROOT/eval" \
  --dest "$HPCVAULT/octvla-rollout-videos/report" --table docs/rollout_videos.md
```

**10. Publish** a unified export. It is always private; ask before pushing.

```bash
PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" scripts/publish_dataset.py "$OCTVLA_DATASET_ROOT/<unified>" \
  --repo-id 3liyounes/<name> --title "…" --dry-run     # drop --dry-run to upload via `hf upload --private`
```

**Floor check, not a ranking.** `slurm/diagnose_action_head.sbatch`
(`DIAG_DATASET`, `DIAG_CHECKPOINT`, `DIAG_OUTPUT`) compares a checkpoint's
offline error on the validation split with two reference predictors: a
constant and repeating the previous action. Never use it,
or validation loss, to rank cells or choose checkpoints.

### Run naming and checkpoint layout

| backbone | run directory under `$OCTVLA_OUTPUT_ROOT` |
| --- | --- |
| ACT `rgb` | `act_<dataset>_s<seed>` |
| ACT `rgb_cont` | `act_<dataset>_s<seed>_cont` |
| ACT `kv` | `control_act_<dataset>_s<seed>` |
| ACT `kv_adaln` / `kv_tokens` / `scratch_kv` | `control_act_<dataset>_s<seed>_{adaln,incontext,scratch}` (suffixes kept from before the rename, so existing checkpoints are still found) |
| VLA stage 1 | `{backbone}_rgb_s<seed>_{id_eeabs,id_abs}` |
| VLA stage 2 | `{backbone}_{arm}_s<seed>_<tag>_s2_<arm>`; `rgb_cont` is `{backbone}_rgb_s<seed>_<tag>_s2_rgb_cont` |

In general a VLA run is named `{backbone}_{arm}_s{seed}_{TRAIN_TAG}`, and
`TRAIN_TAG` is required for any run with `STAGE1_CHECKPOINT` set. The
`id_` prefix keeps these runs apart from earlier runs on older corpora.

```
<run>/checkpoints/<step>/pretrained_model/   config.json, train_config.json, model.safetensors
                                             (LoRA runs: adapter files), pre/post-processor files
<run>/checkpoints/last -> <step>
<run>/merged/                                pi0.5/SmolVLA stage 1: merged weights + merge_provenance.json
run_metadata/<run>.json                      arm, stage, stage-1 source, seed, steps, dataset, job id
run_metadata/<run>_init_check.json           init-check report
eval/<prefix>-<arm>-s<seed>-<job>.json       one rollout; the source of truth
diagnostics/*.json                           collector outputs
slurm_logs/<cell>-<jobid>.out
```

## 5. Measured costs

These numbers are quoted from the documents named in the last column. None was
re-measured for this file. Re-measure if the policy, batch size or dataset
changes.

| item | cost | source |
| --- | --- | --- |
| ACT training, 40 k steps, A40 | ≈ 1 GPU-h per cell | `research_questions.md` §2.1; `experiment_matrix.md` |
| ACT training on an RTX 4090 | ~40 min, 1.69 GB peak VRAM | `handoff_local_4090.md` |
| ACT rollout, 20 scenes × 600 steps | ≈ 0.5 GPU-h; ~1.7 min per episode on a 4090 | `experiment_matrix.md`; `handoff_local_4090.md` |
| pi0.5 LoRA, 8 k steps, batch 16, A40 | ≈ 9.3 GPU-h per cell; 4.13 s/step; 15.6 GB; ~13 min startup | `experiment_matrix.md`; `cluster_envelope.md` |
| VLA cells in general | 3–11 GPU-h per cell (no separate SmolVLA or GR00T figure is documented) | `research_questions.md` §2.1 |
| GR00T training | 36 GB peak VRAM (does not fit a 4090) | `handoff_local_4090.md` |
| VLA rollout | ~0.8 s per step, plus ~3 min fixed overhead; ~2.7 GPU-h per 20-scene three-object job, ~5.3 GPU-h per 2/4-object job | `slurm/submit_vla_experiments.sh`; `cluster_envelope.md` |
| unified export with video | ~1 h | `handoff_local_4090.md` |
| cuRobo build | ~18 min on one A40 | `setup_nhr_alex.md` |

Rollout job limits: 4 h for ACT and 8 h for VLAs (set in the submitters).

## 6. Smoke tests

`slurm/smoke_vla_pipeline.sbatch` runs the whole two-stage VLA path for one
backbone in a few minutes of training. Run it before queuing real VLA work, and
again after any change to that path.

```bash
BACKBONE=pi05 sbatch $SB --export=ALL slurm/smoke_vla_pipeline.sbatch   # or smolvla, groot
```

- The job works entirely in node-local `$TMPDIR`: it overrides
  `OCTVLA_OUTPUT_ROOT` and discards everything at exit.
- Only reports are copied back, to `$OCTVLA_OUTPUT_ROOT/smoke_vla/<backbone>/`
  (`summary.txt`, the init-check JSONs, merge provenance, and rollout JSONs).
- It uses `SMOKE_DATASET` (default `three_object_identity_eeabs`),
  `SMOKE_STEPS=10`, batch 2 and `SMOKE_ROLLOUT_STEPS=3`.

| step | checks |
| --- | --- |
| stage 1 (`rgb`) | the run saves; the optimiser state is removed; for GR00T, `slim_checkpoint.json` is present |
| merge (pi0.5, SmolVLA) | `merge_stage1_adapter.py` passes its functional and round-trip checks |
| init check × 4 (`rgb_cont`, `kv`, `kv_adaln`, `kv_tokens`) | stage 2 starts at stage 1; gradient reaches each branch |
| stage 2 × 4 | verified load, PEFT keeps the branch trainable, the checkpoint saves and reloads |
| rollout of `kv_adaln` and `kv_tokens` (seed 800, seen tier) | the branches run at inference through the bridge |

The job exits non-zero if any line of `summary.txt` is not `ok`.

Other gates that need a simulator (GPU node, RoboTwin env):

- `scripts/validate_eval_bridge.py` sends scripted actions through the real
  socket and server. It checks that a zero action holds, that a commanded
  displacement is achieved, that the gripper reaches its target, and that the
  sim clock runs at 15 Hz. Run it after changing IK, the step loop or frames.
- `slurm/replay_oracle_episode.sbatch` / `slurm/replay_oracle_joints.sbatch`
  (`ORACLE_EPISODE` = a `transfer_index=0` clip) replay the demonstration
  through the bridge. This measures the oracle ceiling that each regime is
  read against.
- `scripts/validate_control_act.py` runs a short train, save, reload and
  inference cycle of `control_act`, on CPU or GPU.
