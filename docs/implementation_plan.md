# Implementation plan: bounded experiment tooling

This is the guardrail layer before any cluster launch. It records the proposed
experiment cells, checks the 100 GPU-hour envelope, and ranks already-completed
rollout checkpoints. It does not submit Slurm jobs, mutate checkpoint
directories, or change the paired evaluation aggregator.

## Budget envelope

The plan is split into five stages with fixed GPU-hour ceilings:

| stage | GPU-hour ceiling |
| --- | ---: |
| stage_1 | 10 |
| stage_2 | 15 |
| stage_3 | 20 |
| stage_4 | 25 |
| stage_5 | 30 |
| **total** | **100** |

`scripts/plan_experiments.py` refuses to build a manifest for a stage unless a
measured throughput record is supplied for that stage. The tooling deliberately
has no fallback estimate, because the cluster envelope is one of the things the
experiment is meant to measure. A throughput record must name its source, such
as a Slurm accounting row or a completed pilot run.

## Manifest identity

Each manifest cell records the fields that define the scientific comparison:

- stage
- backbone
- representation
- training seed
- action schema
- profile
- split
- object-token schema, when applicable

The cell hash is computed from those identity fields using canonical JSON and
SHA-256. The full manifest hash covers the stage ceilings, measured throughput,
budget table and all cells, so changing a backbone, representation, seed or
action schema produces a different manifest.

All cells are bounded to the `three_object` development split. Test and
count-shift scoring are separate evaluation questions and must not be mixed into
development-stage checkpoint choice.

Example:

```bash
$OCTVLA_POLICY_PYTHON scripts/plan_experiments.py \
  --cells /path/to/cells.json \
  --throughput /path/to/measured_throughput.json \
  --output /path/to/manifest.json
```

## Rollout checkpoint ranking

`scripts/select_rollout_checkpoint.py` ranks rollout evaluation JSONs using only
three-object development reports with identical rollout seed sets. Reports are
rejected if they name a test split, count-shift scope, any profile other than
`three_object`, or a seed set that does not match the other candidates.

The ranking tuple is:

1. successes, descending
2. transfers completed, descending
3. infeasible episodes, ascending

This ranking is only for choosing which development checkpoint should advance
to the next stage. It is not a replacement for `scripts/aggregate_eval.py`; the
paired evaluation tooling remains unchanged and should still be used for final
comparisons.

Example:

```bash
$OCTVLA_POLICY_PYTHON scripts/select_rollout_checkpoint.py \
  outputs/*/eval_three_object_dev.json \
  --output outputs/rollout_checkpoint_selection.json
```
