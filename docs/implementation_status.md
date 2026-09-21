# Object-centric implementation and validation

This is an implementation record, not a manipulation-performance report. No
large training sweep has been launched for these changes.

## Ground-truth geometry

`entity_v2` contains 17 values per entity: workcell position (3), the first two
rotation-matrix columns (6), size (3), gripper aperture (1), and a fixed four-way
type one-hot (movable object, left gripper, right gripper, support). It excludes
object IDs, slot embeddings, target roles, and oracle phase labels. The type
codes are orthogonal; the geometric values remain continuous. A separate mask
marks padding. These are not pretrained visual embeddings.

Export and online evaluation call the same builder. Disk values remain raw;
training-only position/size statistics are carried in the policy configuration
and registered checkpoint buffers. Feature normalization is identity before the
policy's entity embedding, avoiding accidental double normalization. Existing
15-value legacy representations and injection modes remain available.

Collection accepts `--episode-kind full_run`. This preserves the original
recorder stream across transfer boundaries. It does not stitch atomic clips
or synthesize missing boundary frames from existing recordings.

Export entity datasets with `scripts/export_lerobot_dataset.py --entity-tokens
--entity-training-sources ...`, specifying the training recordings explicitly.
For a train/validation dataset ready for the launchers, use
`scripts/build_shelf_restock_splits.py --entity-tokens --episode-kind full_run`
with the existing path/repository arguments. It fits statistics automatically
from its reserved training partition after applying `--max-runs`; validation
never contributes. The split manifest records the fitting sources. Cluster
collection accepts `EPISODE_KIND=full_run` and writes into a separate
`full_run/<profile>/seed_<seed>` subtree.

Training launchers validate the exported schema and stored normalization:

- VLA launcher: `TRAIN_VARIANT=object OBJECT_REPRESENTATION=entity_v2`.
- ACT launcher: `ACT_POLICY_TYPE=control_act`.

Both still require their existing dataset, output, environment and Slurm
settings. Collect new full-run recordings before claiming a continuous-run
experiment. Statistics must be fitted exclusively to the training partition.

## Layerwise attention and ACT

The added branch reuses each host layer's projected query and output projection,
with zero-initialized object keys and values. The host output bias is applied
once. Empty and padded entities contribute zero. The ACT integration attaches
to every decoder cross-attention layer; the native ACT network remains the
zero-initialization reference.

CPU tests cover numerical dual-attention equivalence, zero initialization,
gradient progression, permutation/padding invariance, conditioning sensitivity,
and actual ACT checkpoint round trips.

On 2026-09-21, Slurm job **4285960** completed on an A40 in **49 seconds** with
exit code 0. `scripts/validate_control_act.py` ran six synthetic RGB+entity
training steps, followed by inference and checkpoint reload. Loss changed from
0.935524 to 0.746196; the maximum action change under altered entity inputs was
0.0122813. Reload parity passed. These numbers establish executable CUDA
training and conditioning; they do not estimate task success or generalization.
Artifacts are local under `.validation/act-cuda-20260921/` and are git-ignored.

## Validation gates

A dependency-light suite passed **539 tests, with 9 skipped** during integration.
Optional-policy tests require the separate installed PyTorch/LeRobot environment;
a skip in the lightweight environment is not evidence that a model integration
works. Subsequent additions require their own rerun before final sign-off.

Pretrained-model attention hooks, PEFT persistence, JEPA repairs, visual mask
representations, and the upstream baseline are under separate review. In
particular, adapter wrappers can copy modules after attention hooks are created;
checks must establish gradients and save/reload behavior on the active modules,
not only validate a list of adapter target names.

Full model loading, closed-loop simulator pilots, measured throughput manifests,
and staged scientific evaluation remain prerequisites to a sweep. A synthetic
ACT check does not satisfy those gates for the other model families.
