# OCT-VLA dataset protocol

This protocol governs generation, validation, splitting, export, evaluation,
packaging, and release of shelf-restock data for RGB and object-conditioned
π0.5 experiments.

## Units and anti-leakage rule

One continuous oracle run clears a shelf and is cut into one atomic clip per
transfer. The atomic clip is the training unit; the source scene seed is the
split unit. Never place clips from one source seed in different train,
validation, or test sets: they share layout, asset variant, camera state, and
placement history.

The existing 27 clips from nine three-object source seeds are a **superseded**
development dataset. They were recorded before the spawn geometry was unified
across profiles and before the head camera was reframed, so they do not match
what the collection scripts now produce and must not be mixed with new data.
They remain usable only for pipeline smoke tests, never for reported results.

## Canonical recording contract

Canonical recordings are the source of truth. Each episode directory contains
`episode.json` plus gzipped head, left-wrist, and right-wrist RGB streams.
Each sample contains timestamp, RGB, both EEF poses and gripper states,
canonical 14-D action, `ObjectScene`, `TaskContext`, and phase.

Episode metadata records seed, success, sampling rate, transfer index, target,
previous neighbour, compaction state, object count, and task profile. The
validator rejects non-monotonic or irregular timestamps, inconsistent cameras,
non-finite actions, missing targets, invalid object context, and empty phases.
Failed or partial oracle runs are discarded in full. Privileged diagnostics are
debug-only and never enter policy observations or targets.

## Collection profiles

| Profile | Objects | Use | Status |
| --- | ---: | --- | --- |
| `two_object` | 2 | Count-shift (sparse) validation/evaluation | Implemented |
| `three_object` | 3 | Training and IID held-out data | Implemented |
| `four_object` | 4 | Count-shift (crowded) validation/evaluation | Implemented |

**Every profile shares one geometry and differs only in object count.** All
three use the same `DEFAULT_SPEC` and the same task class hierarchy, varying
only `object_count`. This is deliberate and load-bearing: when each profile
carried its own spawn region, the crowded profile also moved that region, so a
policy evaluated on it faced a combined count-and-position shift that could not
be attributed to either cause. The shared spawn span is therefore sized for the
largest profile — `0.47 m`, above the `0.45 m` that four objects at `0.15 m`
separation require — and the smaller counts occupy the same span more sparsely.
Neither count-shift profile may replace the three-object training profile.

The current task uses one coffee-box variant per source scene;
appearance-generalization needs a future profile with held-out assets or
variants and documented geometry.

## Split and evaluation regime

Reserve seed ranges before collection. Do not revise membership after observing
policy scores.

### Reserved seed ranges

Fixed before any collection run. A seed belongs to exactly one split, so a
scene can never move between train and test after the fact.

| Split | Profile | Reserved seeds | Successes needed |
| --- | --- | --- | ---: |
| Train | `three_object` | 100-199 | 30 |
| IID validation | `three_object` | 200-249 | 10 |
| Final offline test | `three_object` | 250-349 | 20 |
| Count-shift validation | `two_object` | 400-449 | 10 |
| Final offline test | `two_object` | 450-549 | 20 |
| Count-shift validation | `four_object` | 600-649 | 10 |
| Final offline test | `four_object` | 650-749 | 20 |

Selection rule, fixed in advance: within a block, take successful seeds in
ascending order until the split's target is met and ignore the remainder.
That is what makes a top-up safe -- extending into the unused tail of a block
cannot change which scenes are already in the split.

Seeds below 100 are not reserved: they were used by the superseded 27-clip
development dataset and by ad-hoc feasibility runs. Seeds at and above 1000 are
likewise unreserved and are the right place for feasibility runs and for
closed-loop evaluation scenes, which must never have been trained on.

### Collected so far

| Split | Seeds submitted | Evaluated | Successful | Clips |
| --- | --- | ---: | ---: | ---: |
| Train (`three_object`) | 100-159 | 58 | 30 | 90 |
| IID validation (`three_object`) | 200-219 | 20 | 11 | 33 |

Yield is 52% and 55% respectively, against the ~60% the block sizes were
provisioned for. Every successful `three_object` scene contributes exactly three
clips. Measured footprint: 172 KB per sample canonical, ~29 MB per clip average,
and ~0.7 MB per clip once exported to LeRobot -- canonical dominates by more than
an order of magnitude because it stores gzipped raw RGB rather than H.264.

Seeds 100 and 101 were never evaluated: both array tasks hit an infrastructure
`NODE_FAIL` and wrote no report. They were deliberately not re-run. The train
split already held its full target of 30 successful scenes, and the selection
rule takes successes in ascending order -- so a late success at seed 100 or 101
would have displaced a higher-numbered scene already in the split. The
consequence for reproducibility is that re-running block 100-199 from scratch
could produce different membership; the seeds actually used are recorded per
episode in `octvla_episode_manifest.json`, which is the authoritative record.

The two- and four-object blocks are unused. Count-shift generalization is
measured closed-loop in simulation rather than from pre-collected clips, so those
profiles need scenes at evaluation time, not demonstrations.

Every reserved seed is under 1000 on purpose. `collect_shelf_restock_array.sbatch`
uses the Slurm array index as the scene seed, and Slurm's default
`MaxArraySize` is 1001, so a seed of e.g. 10000 is rejected at submission with
an invalid job array specification. Keep any future block under that ceiling,
or give the array script its own seed base.

A discarded seed is spent. Replace it by extending into the same block's
unused tail, never by re-running it -- the failure does not depend on how many
times the seed is asked.

| Set | Profile | Successful source scenes | Use |
| --- | --- | ---: | --- |
| Train | 3 objects | 30 | Fit both policies |
| IID validation | 3 objects | 10 | Checkpoint and hyperparameter selection |
| Count-shift validation | 2 and 4 objects | 10 each | Development transfer measurement |
| Final offline test | 2, 3 and 4 objects | 20 each | Frozen-model BC comparison |
| Closed-loop simulation | Each scenario | 30–50 each | Task success and confidence intervals |

Count shift is measured in both directions — sparser (2) and denser (4) than
training — so a drop can be told apart from a monotonic sensitivity to scene
clutter.

Evaluate closed-loop policies separately on IID scenes, two- and four-object
count shift, unseen spawn pose/yaw, no-neighbour versus compaction clips,
held-out appearance/geometry when available, and combined shifts. For object policies,
report simulator-GT tokens and runtime-estimator tokens separately.

## Slurm collection

Collection runs with RoboTwin Python 3.10 and a CUDA-visible GPU. One Slurm
array task owns one seed and writes only under
`$OCTVLA_COLLECTION_ROOT/<profile>/seed_<seed>/`. This prevents concurrent
writers. A discarded seed is recorded but produces no usable data; replace it
with a fresh reserved seed rather than retrying it as a new observation.

After collection, run exactly one finalizer in the LeRobot policy environment.
It creates consistent parquet indices, video files, metadata, and statistics.
Use [slurm_collection.md](slurm_collection.md) for commands. First submit one
seed of the profile you are about to collect and review its exported video
before any range submission.

## LeRobot export contract

LeRobot is a derived portable format; export never edits canonical recordings.
A finalized dataset contains `data/`, `meta/`, `videos/`, and
`octvla_episode_manifest.json`.

| Feature | Shape | Meaning |
| --- | --- | --- |
| `observation.state` | `[16]` | Left/right EEF xyz, xyzw, and gripper state |
| `action` | `[14]` | Canonical dual-arm action |
| `observation.images.*` | `[H,W,3]` video | Head and wrist RGB |
| `task` | string | Task instruction |

The object-conditioned export comes from the identical canonical sources and
adds `observation.object_tokens` with shape `[8,15]` plus a `[8]` mask. Each
token contains workcell position, xyzw orientation, own-frame size, visibility,
confidence, and target/previous/other role. Tokens are ordered target, previous
neighbour, then remaining episode-local IDs. Asset and simulator identifiers
are excluded.

Check that RGB and object exports have identical shared state/action/timestamp
rows before paired training. `octvla_episode_manifest.json` maps LeRobot index
to canonical source name, seed, sample count, metadata, and `episode.json`
SHA-256. It is the split-audit record.

Verified on the current export: all of `timestamp`, `frame_index`,
`episode_index`, `index`, `task_index`, `observation.state[16]` and
`action[14]` are identical across the paired datasets, so the only difference
between the two training arms is the presence of object tokens.

### Split layout inside the dataset

LeRobot's offline validation splits **one** dataset positionally --
`make_train_eval_datasets` holds out the last `ceil(n * eval_split)` episodes per
task and never looks at seeds. `scripts/build_shelf_restock_splits.py` therefore
exports train episodes first, then validation, each in ascending seed order, and
derives the fraction that makes that positional rule select exactly the reserved
validation block. The value lands in `split_manifest.json` next to the dataset,
and the training job reads it from there rather than restating it.

For the current export that is 123 episodes, 90 train / 33 validation, with the
boundary between episode 89 (seed 159, last train) and episode 90 (seed 202,
first validation). The exporter fails rather than proceeding if the boundary does
not fall on that division -- silently training on validation clips is the failure
this arrangement exists to prevent.

For split experiments, normalization statistics must be calculated from train
episodes only. Never normalize using validation or test scenes.

## Provenance and versioning

Each release records its semantic version, source seed lists and split mapping,
task profile, collection command/reports, accepted and discarded seed counts,
OCT-VLA revision, RoboTwin/cuRobo/SAPIEN/PyTorch versions, GPU/driver, asset
checksums, geometry, camera configuration, action/token schemas, and SHA-256
manifests. Changing recordings, geometry, profile, exporter, token schema, or
splits creates a new version; never overwrite a release.

## Packaging and use

Freeze and publish three artifacts: canonical source archive, RGB LeRobot
dataset, and object-conditioned LeRobot dataset from the same sources. RGB and
object schemas use separate Hugging Face repositories. The publisher uploads
data, metadata, and videos, writes `dataset_manifest.json`, and creates an
immutable Hub tag.

Publish with `scripts/hub_dataset.py publish <dataset> <org/repo> --release
v1.0.0 --private`. Retrieve on any machine with `scripts/hub_dataset.py
download <org/repo> --release v1.0.0 --output <directory>`. The downloaded
directory is directly usable as LeRobot `--dataset.root`; no regeneration is
needed. See [dataset_distribution.md](dataset_distribution.md).

Train RGB and object-conditioned policies on the same source-scene split. Keep
training, validation, offline test, and simulation results separate. Do not
tune on final test results, and report scene counts, discarded seeds, atomic
clips, per-cell task success, and confidence intervals.
