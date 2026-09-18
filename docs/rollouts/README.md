# Rollout videos

One closed-loop episode per trained cell, recorded through the same bridge and
servo loop evaluation uses. Each file is 40 s at 15 Hz — head camera, left wrist,
right wrist, side by side — with the cell name and live counters burned in.

Every cell is the **same scene seed (800)** so the grid is comparable. The one
exception is noted below.

Recorded with `slurm/submit_rollout_videos.sh`; the recorder is
`src/oct_vla/serve/video.py`, wired into `scripts/eval_shelf_restock_policy.py`
behind `--video-dir`. The frames are free: the server sends all three cameras on
every step whatever the policy reads, so the vision-free privileged checkpoints
record as well as the RGB ones.

## What to look for

The counters cannot separate *reached and missed* from *never approached*, and
that distinction is the open question in `control_space_comparison.md`. The wrist
views are what settle it — they show whether the gripper closed on anything.

`infeas` counts steps whose command was kinematically infeasible and was held
instead. Only the EE-delta cells can produce it; joint space cannot, by
construction, and absolute EE produced none.

## The grid

pi0.5 arms are seed 1000 at 8000 steps, evaluated from `checkpoints/last`.
ACT cells are seed 1000 at 40000 steps on the r75 corpus, evaluated from
`checkpoints/best` (lowest validation loss, not the final step).

| policy | observation | action space | transfers | lifted | infeas | video |
| --- | --- | --- | ---: | ---: | ---: | --- |
| pi0.5 | RGB | EE delta | 0 | 0 | 12 | [video](pi05_rgb__ee_delta-seed800.mp4) |
| pi0.5 | ControlVLA | EE delta | 0 | 0 | 606 | [video](pi05_controlvla__ee_delta-seed800.mp4) |
| pi0.5 | ControlVLA role-stripped | EE delta | 0 | 0 | 394 | [video](pi05_controlvla_rolestripped__ee_delta-seed800.mp4) |
| pi0.5 | RGB | absolute joint | 0 | 0 | 0 | [video](pi05_rgb__absolute_joint-seed800.mp4) |
| pi0.5 | ControlVLA | absolute joint | 0 | 0 | 0 | [video](pi05_controlvla__absolute_joint-seed800.mp4) |
| pi0.5 | ControlVLA role-stripped | absolute joint | 0 | 0 | 0 | [video](pi05_controlvla_rolestripped__absolute_joint-seed800.mp4) |
| pi0.5 | RGB | joint delta | 0 | 0 | 0 | [video](pi05_rgb__joint_delta-seed800.mp4) |
| pi0.5 | ControlVLA | joint delta | 0 | 0 | 0 | [video](pi05_controlvla__joint_delta-seed800.mp4) |
| pi0.5 | ControlVLA role-stripped | joint delta | 0 | 0 | 0 | [video](pi05_controlvla_rolestripped__joint_delta-seed800.mp4) |
| ACT | RGB | EE delta | 0 | 0 | 102 | [video](act_rgb__ee_delta-seed800.mp4) |
| ACT | RGB | absolute EE | 0 | 0 | 0 | [video](act_rgb__absolute_ee-seed800.mp4) |
| ACT | RGB | joint delta | 0 | 0 | 0 | [video](act_rgb__joint_delta-seed800.mp4) |
| ACT | RGB | joint delta + velocity | 0 | 0 | 0 | [video](act_rgb__joint_delta__velocity-seed800.mp4) |
| ACT | RGB | absolute joint | 0 | 0 | 0 | [video](act_rgb__absolute_joint-seed800.mp4) |
| ACT | privileged | joint delta | 0 | 0 | 0 | [video](act_privileged__joint_delta-seed800.mp4) |
| ACT | privileged | joint delta + velocity | 0 | 0 | 0 | [video](act_privileged__joint_delta__velocity-seed800.mp4) |
| **ACT** | **privileged** | **absolute joint** | **1** | **1** | 0 | [video](act_privileged__absolute_joint-seed800.mp4) |

### The other successful rollout

`ACT | RGB | absolute joint` completes no transfer on seed 800 but does on
others. Seed 806 is recorded as well — 1 transfer, 3 objects lifted, its best
episode of the twenty — so that both of the project's successful cells can be
watched:

| policy | observation | action space | seed | transfers | lifted | video |
| --- | --- | --- | ---: | ---: | ---: | --- |
| ACT | RGB | absolute joint | 806 | 1 | 3 | [video](act_rgb__absolute_joint-seed806.mp4) |

Seed 800 is not representative of the absolute-joint cells — across 20 seeds they
lift in 13 and transfer in 1–2. A single episode is an illustration, not a
measurement; the rates are in `control_space_comparison.md` section 5.

## Caveats

- **pi0.5 EE-delta cells are void as evidence.** They ran against the evaluation
  harness described in `control_space_comparison.md` section 7, which has since
  been shown broken — the high `infeas` counts are that bug, not a property of
  task-space control. They are here for completeness; the valid EE numbers are
  the ACT EE cells, recorded on the fixed bridge.
- **pi0.5 cells are scored from `checkpoints/last`**, because no validation-loss
  history was kept for those runs. The ACT cells use `checkpoints/best`.
- **One episode per cell.** Nothing here establishes a rate.
