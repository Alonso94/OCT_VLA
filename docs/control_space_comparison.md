# Control spaces for shelf restocking: EE deltas, absolute joints, joint deltas

Everything needed to write this up: what was measured, under which conditions,
and which numbers are not yet in. Cells marked **TBD** are pending runs, not
omissions.

For a report-ready synthesis of all of this, see `docs/findings.md`; this
document is the detail behind it.

**Start at section 5.** It holds the only controlled comparison in this
document — one policy, one corpus, one seed, one evaluation protocol, varying
nothing but the action encoding — and it is the first result in this project
that is not zero. Sections 2-4 describe the three earlier pi0.5 sweeps, which
varied several things at once and, for two of them, ran against an evaluation
harness since shown to be broken.

The headline: **absolute joint targets beat joint increments decisively**, and
the offline metrics predicted the opposite. Anything in sections 2-3 arguing
from encoding conditioning should be read against section 6.

---

## 1. The three control spaces

| | **EE delta** (`cartesian`) | **Absolute joint** (`joint`) | **Joint delta** (`joint_delta`) |
| --- | --- | --- | --- |
| `observation.state` | 16-d EEF pose (pos + quat + gripper, both arms) | 16-d joint configuration | 16-d joint configuration |
| `action` | 14-d per-step EE increment (m, rad) | 16-d absolute joint target | 16-d joint increment, gripper absolute |
| Execution | increment → IK → joint command | joint command | measured joints + increment → joint command |
| Inverse kinematics | **yes, every step, both arms** | none | none |
| Can a command be infeasible? | **yes** | no | no |
| Integrator | yes (Cartesian reference) | none | none (anchored to measured) |

`action` is fixed at export time — LeRobot's rename map is observation-only, so
the control space cannot be selected afterwards. Each dataset records its own
`control_space` in `meta/info.json`, and evaluation refuses to run without it:
absolute and incremental joint actions have identical column layouts, and
executing one as the other is silently wrong.

---

## 2. Action encoding conditioning

The single most explanatory measurement in this comparison. A policy must
resolve one control step against the normalisation scale of its action column.

pi0.5 normalises STATE and ACTION with **QUANTILES**, not mean/std:
`2·(x − q01)/(q99 − q01) − 1`, mapping the 1st–99th percentile range onto
[−1, 1]. Visual features are IDENTITY. The relevant scale is therefore the
q01–q99 range, and quoting a standard deviation understates the problem: the
joint-position range (1.147 rad) is far wider than 2σ (0.614 rad), because
joint values are broadly spread and not Gaussian.

| | EE delta | Absolute joint | Joint delta |
| --- | ---: | ---: | ---: |
| Mean motion per step | 11.32 mm | 0.00907 rad | 0.00907 rad |
| q01–q99 range of the action column | — | 1.14740 rad | **0.16278 rad** |
| **One step, in normalised units ([−1, 1])** | — (delta-encoded) | **0.0158** | **0.1114** |
| Resolution vs absolute | — | 1× | **7.0×** |

So one control step moves the absolute-joint action by 1.6 % of the model's
entire output range. The measured prediction error of 0.033 rad is 0.058 in the
same units — **3.7× the step itself**, which matches the 3.6× excess motion
observed directly.

Measured on 19 005 frames of the joint corpus. Reconstruction check for the
delta encoding: `state[i] + action[i] − state[i+1]` has max error **1.49e-08**,
and the gripper stays absolute (range 0.535–0.833), not differenced.

### What the policy actually predicts

Measured on *training* frames, so this is not a generalisation effect.

| | EE delta | Absolute joint | Joint delta |
| --- | ---: | ---: | ---: |
| Oracle motion per step | 11.32 mm | 0.00825 rad | 0.00825 rad |
| Policy motion commanded | 9.91 mm | **0.02969 rad** (3.6× too much) | 0.00802 rad |
| Mean prediction error | 0.0047 (mixed units) | **0.03291 rad** (4× the step) | **0.01452 rad** (1.76× the step) |

For absolute joints the prediction error **exceeds the motion being predicted**.
That is the mechanism behind the zero closed-loop result, and it is a property
of the encoding, not of model capacity or data volume.

Joint deltas fixed the *magnitude* problem completely — 0.00802 rad commanded
against an oracle 0.00825 — and did not cross 1.0 on the ratio. Both rows were
measured with `policy.reset()` on each frame, so they are best-case single-step
predictions on frames the model was fit on.

**This section's argument turned out not to predict control.** Joint deltas win
every conditioning measure here and lose the closed-loop comparison in section 5
by a wide margin. Section 6 explains why: what matters is not how large the
error is relative to the step, but whether the error is a *position* error that
the servo absorbs or a *direction* error that redirects the arm.

---

## 3. Training statistics

pi0.5, LoRA r=16 α=32, bf16, gradient checkpointing, batch 16, 8000 steps,
96 train / 39 val episodes. Three arms × three seeds.

### EE delta (`cartesian`)

| arm | seed | train | eval |
| --- | ---: | ---: | ---: |
| rgb | 1000 / 1001 / 1002 | 0.049 / 0.048 / 0.047 | 0.0764 / 0.0746 / 0.0766 |
| object_full | 1000 / 1001 / 1002 | 0.038 / 0.039 / 0.037 | 0.0746 / 0.0750 / 0.0714 |
| object_role_stripped | 1000 / 1001 / 1002 | 0.040 / 0.040 / 0.038 | 0.0944 / 0.0799 / 0.0904 |

### Absolute joint (`joint`)

| arm | seed | train | eval |
| --- | ---: | ---: | ---: |
| rgb | 1000 / 1001 / 1002 | 0.042 / 0.043 / 0.041 | 0.1318 / 0.1311 / 0.1397 |
| object_full | 1000 / 1001 / 1002 | 0.030 / 0.031 / 0.029 | 0.1514 / 0.1547 / 0.1502 |
| object_role_stripped | 1000 / 1001 / 1002 | 0.033 / 0.034 / 0.032 | 0.1310 / 0.1253 / 0.1348 |

### Joint delta (`joint_delta`)

Job array 4266187, 9 cells, same recipe (LoRA r=16, batch 16, 8000 steps).
Train / eval loss per seed:

- rgb: 0.044/0.045/0.043 → 0.0693/0.0679/0.0716
- object_full: 0.035/0.036/0.033 → **0.0605/0.0632/0.0611**
- object_role_stripped: 0.036/0.037/0.035 → 0.0666/0.0630/0.0662

The eval-loss curves are flat after about step 4000, so this is not an
overfitting story.

### Summary

| | EE delta | Absolute joint | Joint delta |
| --- | ---: | ---: | ---: |
| Train loss (best arm) | 0.037 | **0.029** | 0.033 |
| Eval loss (best arm) | 0.0714 | 0.1253 | **0.0605** |
| Train→eval gap | 0.03 | 0.09–0.12 | **0.025** |

Absolute joints fit the training set *better* and generalise *worse*. The gap
is the conditioning problem in section 2 showing up as a loss: the target is
hard to resolve, so the model memorises rather than generalising.

Note also that `object_full` has the lowest training loss in both spaces —
consistent with the role one-hot carrying real predictive signal, which is
exactly the leakage arm C exists to detect.

---

## 4. Closed-loop evaluation

Each cell: 3 count profiles × 30 scene seeds = 90 episodes, 600-step limit.
Success requires **every object on the upper shelf**.

| | EE delta | Absolute joint | Joint delta |
| --- | ---: | ---: | ---: |
| Harness state | **broken** (see §5) | verified | verified |
| Success, all arms | 1 / 1080 | 0 / 1080 | **0 / 480** |
| Mean transfers | 0.00 | 0.00 | **0.00** |
| Episodes ending `unreachable_pose` | **1064 (98.5 %)** | 0 | 0 |
| Episodes reaching the step limit | 15 | **1080 (100 %)** | **480 (100 %)** |
| Infeasible commands per episode | 29–441 | **0** | **0** |

The joint-delta sweep is 480 episodes rather than 1080 because arm C was
trained but never evaluated, and because the four-object profile did not
complete for every cell. Every episode that did run scored zero, on the best
offline loss of the three spaces — the clearest statement in this document that
held-out loss was not tracking control.

The EE-delta row is **not a usable result.** It was produced before the harness
was fixed and measures the harness, not the policies.

The absolute-joint row *is* usable: every episode ran to completion with zero
infeasible commands, so the bridge executed exactly what the policy asked for.

### Oracle replay ceiling

A perfect policy cannot exceed the rate at which the demonstrations themselves
replay through the bridge. This bounds every number above.

| | EE delta | Absolute joint | Joint delta |
| --- | ---: | ---: | ---: |
| Before harness fixes | **0 / 6** | — | — |
| After harness fixes | 6 / 6 | **19 / 20 (~95 %)** | **5 / 6** |
| Worst joint tracking error | ~9.4 mm (Cartesian) | 0.0065–0.0137 rad | 0.0139–0.0414 rad |

Absolute and delta encodings replay equally well (5/6 on the same six seeds,
failing on the same one), so the delta encoding costs nothing in executability.
Its tracking error is slightly larger, as expected: an increment applied to the
measured configuration inherits that measurement's error, where an absolute
target does not.

The one persistent failure (seed 112) is deterministic across repeats: replayed
joint targets do not reproduce the oracle's *execution dynamics*, and a
marginal grasp drops.

---

## 5. The controlled comparison (ACT, r75 corpus, 2026-09-18)

The three sweeps above each changed the control space *and* the corpus *and*
the harness. This one changes nothing but the action encoding and the
observation: one backbone (ACT), one training seed (1000), one corpus (75 train
runs / 225 atomic clips, 26 val runs / 78 clips), one recipe (40 000 steps,
batch 8, chunk 50, save every 2000, best checkpoint by validation loss), one
evaluation protocol (20 held-out scene seeds, `three_object`, 600 steps).

Both conditionings are run for every cell: `rgb` (three cameras) and
`privileged` (ground-truth object tokens, no cameras).

### 5.1 Offline prediction

Validation L1 through `predict_action_chunk` — the inference path, *not* the
training log's `eval_loss`, which conditions the VAE encoder on the ground-truth
action chunk and is therefore optimistic.

**Normalised losses are not comparable across control spaces.** Absolute joint
actions have roughly 13× the MEAN_STD scale of increments, so the same physical
error reads as a smaller normalised number. The `rad err` column converts k=0
back to radians on the arm joints and is the only cross-encoding unit here.

| cell | cond | constant | persistence | model | k=0 | k=49 | vs constant | rad err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| joint_delta, position | priv | 0.466 | 0.125 | 0.279 | 0.254 | 0.330 | 40 % | 0.0058 |
| joint_delta, position | rgb | 0.466 | 0.125 | 0.226 | 0.220 | 0.234 | 52 % | 0.0050 |
| joint_delta, +velocity | priv | 0.466 | 0.125 | 0.233 | 0.193 | 0.271 | 50 % | 0.0044 |
| joint_delta, +velocity | rgb | 0.466 | 0.125 | **0.188** | **0.153** | 0.199 | **60 %** | **0.0035** |
| absolute joint | priv | 0.839 | 0.029 | 0.163 | 0.089 | 0.205 | 81 % | 0.0278 |
| absolute joint | rgb | 0.839 | 0.029 | 0.138 | 0.075 | 0.166 | 84 % | 0.0234 |

Reference predictors, computed from the parquet without a model: **constant** is
the training mean of each action dimension, **persistence** repeats the previous
action. They bracket the problem — a model that does not beat the constant has
learned nothing, and a persistence score far below the model's says the target is
determined mostly by its own history.

### 5.2 Closed-loop

Same 20 seeds for every cell. `transfers >= 1` is the atomic bar: one object
moved to the upper shelf, which is the unit each training clip contains.

| cell | cond | task | transfers >= 1 | mean transfers | lifted >= 1 | mean lifted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| joint_delta, position | priv | 0 | 0 | 0.00 | 0/20 | 0.00 |
| joint_delta, position | rgb | 0 | 0 | 0.00 | 0/20 | 0.00 |
| joint_delta, +velocity | priv | 0 | 0 | 0.00 | 0/20 | 0.00 |
| joint_delta, +velocity | rgb | 0 | 0 | 0.00 | 2/20 | 0.10 |
| **absolute joint** | priv | 0 | **1** | 0.05 | **13/20** | **0.85** |
| **absolute joint** | rgb | 0 | **2** | 0.10 | **13/20** | **0.95** |

Every cell ran all 600 steps with zero infeasible commands, so nothing here is a
harness failure.

This is the first non-zero result in the project. Against 1 600+ episodes of
uniform zeros across the three earlier sweeps, absolute joint targets move the
lift rate from 0.00 to 0.85–0.95 objects per episode and put 13 of 20 episodes
in contact with the task.

It is a signal, not a working baseline: 1–2 transfers in 20 is 5–10 %, against
an oracle replay ceiling near 95 %.

### 5.3 The velocity cell

`state_encoding="position_velocity"` appends the backward difference of the
measured configuration to `observation.state`, doubling it to 32-d. The
motivation was measured: with `joint_delta` the action is essentially a
velocity, ACT hard-caps `n_obs_steps` at 1, and a static position snapshot does
not determine velocity — persistence scores 0.125 against the model's 0.284, and
a ridge regression from the *complete* privileged observation scores 0.563,
worse than a constant.

It worked offline and did not transfer. Next-step error fell 24–30 % (k=0
0.254→0.193 privileged, 0.220→0.153 rgb), the largest offline gain of any change
tried, and closed-loop it produced 2 lifts and 0 transfers.

Worth recording that it is **not** a copycat failure. A policy that simply
repeated the previous action — now visible in its observation — would score
0.125; it scores 0.153, worse. It is blending motion with scene information
rather than latching onto history. The blend just does not help control.

### 5.4 Execution horizon

`n_action_steps` is inference-only for a chunked policy, so the same checkpoints
were re-scored at four horizons without retraining. Same 20 seeds throughout.

| cell | n=50 | n=25 | n=5 | n=1 | n=1 + ensembling |
| --- | ---: | ---: | ---: | ---: | ---: |
| privileged, lift >= 1 | 13 | **14** | 8 | **0** | 2 |
| privileged, mean lifted | 0.85 | 0.85 | 0.40 | **0.00** | 0.10 |
| rgb, lift >= 1 | 13 | **14** | 4 | 1 | 9 |
| rgb, mean lifted | 0.95 | **1.20** | 0.20 | 0.05 | 0.45 |

Shortening the horizon makes the policy **worse**, monotonically, and fully
closed-loop execution removes the behaviour entirely on the privileged cell.
Temporal ensembling (`temporal_ensemble_coeff=0.01`, the original ACT setting)
recovers part of what `n=1` loses -- rgb goes 1 lift to 9 -- and still does not
reach `n=25`.

This was the leading hypothesis for the closed-loop failure and it is wrong. The
mechanism is visible in 5.1: the offset curve is nearly flat, so there is no
accurate near term to exploit by re-querying. At `n=1` the policy executes
`action[0]` of a freshly predicted chunk and discards the other 49, and
consecutive predictions from slightly different observations disagree, so the arm
jitters instead of committing to a motion. Executing a whole chunk gives a
temporally coherent trajectory even when each action is imperfect -- which is
what action chunking is for.

`n=25` is adopted: mildly better than 50 on both cells, and 2x chunk overlap is
the ratio LeRobot's own configuration docstring describes and the one FlashVLA
ships with. The margin over 50 is inside the +-0.2 Wilson interval at n=20, so it
is a tie-break, not a result.

---

## 6. Why absolute targets win despite a larger error

Absolute joints have 5–7× the k=0 error in radians (0.023–0.028 against
0.0035–0.0058) and win the closed-loop comparison outright. The conditioning
argument in section 2 predicts the reverse, so it is wrong about something.

The error *type* matters more than its size:

- An **absolute target** is a position the servo drives toward. A 0.023 rad
  error puts the arm about 1.2 cm from where the demonstration was, and it stays
  1.2 cm off — the error does not build.
- A **delta** is a direction. The oracle's true step is 0.0088 rad and the best
  model's error is 0.0035 rad, so roughly 40 % of each commanded step points the
  wrong way. Because the increment is added to the *measured* configuration, the
  controller re-anchors every step and nothing integrates in the servo — but the
  trajectory random-walks, and the arm wanders instead of progressing.

So the right question was never "how many sigma is one step" but "does a
prediction error redirect the arm or merely displace it". Section 2's framing
optimises the first and section 5 measures the second.

### Offline metrics have not predicted control, four times running

| change | offline effect | closed-loop effect |
| --- | --- | --- |
| EE delta → absolute joint (pi0.5) | eval loss 0.0714 → 0.1253, worse | 0 → 0 |
| absolute joint → joint delta (pi0.5) | eval loss 0.1253 → 0.0605, best of three | 0 → 0 |
| 45 → 75 training runs (ACT) | eval loss 0.460 → 0.322, −30 % | 0 → 0 |
| position → +velocity state (ACT) | k=0 −24…−30 % | 0 → 0 |
| **joint delta → absolute joint (ACT)** | **rad err 5–7× worse** | **0 → 0.85–0.95 lift** |

The only change that moved control is the one that looked worst offline. Treat
held-out action loss as a debugging tool for "did the model learn anything",
not as a proxy for task success.

### What the lift/transfer gap says next

13 of 20 episodes lift an object and 1–2 complete a transfer. That gap — reaches
and grasps, does not place or release — points at the gripper. Its command is
the measured aperture thresholded at 0.822522, which makes "open" a band 0.0108
wide at the top of the range. Measured error against that margin on the winning
cells:

| | left | right |
| --- | ---: | ---: |
| absolute, privileged | 0.97× | 1.30× |
| absolute, rgb | 0.55× | 1.48× |

The right gripper — the arm doing the work — is above 1.0 in both, so its
open/close decision is still unreliable.

---

## 7. Infrastructure caveats — read before quoting any EE-delta number

The first sweep returned 0/1080 with `mean_transfers` of exactly 0.00 for every
arm. That uniformity was the tell. Five defects in the evaluation bridge, found
by replaying the oracle's own demonstrations through it:

1. **IK used the collision-aware solver**, whose world contains the target
   object — a valid grasp was rejected *because* the gripper was touching what
   it was meant to grasp.
2. **The gripper action carried the measured aperture**, which stalls at 0.6–0.8
   while grasping. Fed back as a drive target it releases the object. This alone
   explains `mean_transfers = 0.00`.
3. **Commands were issued once and the scene ticked 16–17 times**, leaving stale
   drive targets and passive forces.
4. **The Cartesian reference was rebuilt from the measured pose each step**,
   discarding actuator residual and re-accruing it as lag.
5. **Success tested whether the lower shelf was clear**, which a policy can
   satisfy by sweeping objects onto the floor. One did, scoring a success with
   zero transfers.

Fixed; oracle replay went 0/6 → 6/6. Two further corrections affect
interpretation rather than execution:

- **Role-stripped ordering was unstable.** Sorting on each frame's positions
  re-permuted tokens whenever objects crossed. The arm fit *worse than RGB*
  (eval 0.088–0.094 vs 0.073–0.076). It now uses a per-episode ordering fixed
  from the first frame, and is no longer the worst arm (0.125–0.135 vs
  object_full's 0.150–0.155).
- **Collection is non-deterministic.** Re-collecting the same seeds produced
  different trajectories in 74 of 120 episodes. Scene seeds still partition the
  splits, but "same seed ⇒ same demonstration" does not hold here.

---

## 8. What can and cannot be claimed

**Supported by the data:**

- **Absolute joint targets beat joint increments closed-loop**, in a controlled
  comparison varying only the action encoding: 13/20 episodes lift an object
  against 0-2/20, and 1-2/20 complete a transfer against 0/20. Both
  conditionings agree.
- Removing IK from the loop eliminates command infeasibility entirely — 0 in
  1080 episodes, against 29–441 per episode in the Cartesian space.
- The evaluation harness, not the policies, produced the first sweep's result.
- Held-out action loss does not predict closed-loop success here. Five encoding
  or data changes are tabulated in section 6; the four that improved offline
  loss all scored zero, and the one that moved control looked 5–7× worse.
- Joint increments are better conditioned on every offline measure — smaller
  normalised step, smaller error in radians, best pi0.5 eval loss of the three
  spaces. That advantage did not convert.

**Not yet supported:**

- Anything about object-centric conditioning. The Stage 2 gate is still
  unanswered: the pi0.5 arms all scored zero, and the ACT cells vary the action
  encoding rather than the conditioning. `rgb` slightly outscores `privileged`
  in the ACT cells (2 transfers against 1, 0.95 lift against 0.85), which is
  within noise at n=20 and is in any case the opposite of an object-state
  advantage.
- That absolute joint targets *solve* the task. 5–10 % transfers against a ~95 %
  oracle ceiling is a signal that the pipeline can work, not a baseline.
- **Anything comparing EE control to joint control.** The only EE-delta sweep
  ran against the broken harness (section 7) and is void. A valid comparison
  needs EE re-run on the fixed bridge, and needs an absolute EE variant —
  otherwise "EE vs joint" is confounded with "delta vs absolute", which section
  5 shows is the larger effect. See section 10.

**Closed off:** data volume as the primary explanation. Going from 45 to 75
training runs cut ACT's validation loss 30 % (0.460 → 0.322) and left
closed-loop at zero; the encoding change moved it with no extra data.

---

## 9. Provenance

| | |
| --- | --- |
| EE-delta sweep | job 4251265, 2026-09-15, `$HPCVAULT/octvla-outputs` |
| Absolute-joint sweep | job 4261431, 2026-09-16, `$HPCVAULT/octvla-outputs-jointspace` |
| Joint-delta sweep | job 4266187, 2026-09-17, `$HPCVAULT/octvla-outputs-jointdelta` |
| ACT control-space cells | jobs 4268798-4268805, 2026-09-18, same root |
| ACT cell evaluation | jobs 4269720-4269725, 20 seeds each, `eval/cells/` |
| Offline diagnostics | jobs 4269708-4269713, `diagnostics/` |
| Joint corpus (pi0.5 sweeps) | 135 episodes, 45 seeds, 19 005 frames |
| Joint corpus (ACT cells, r75) | 303 episodes, 101 runs, 42 777 frames |
| Cluster envelope | `docs/cluster_envelope.md` |
| Experiment design | `docs/experiment_sweep.md` |

---

## 10. Defects that postdate the cells above

Four were found after these numbers were measured, so they are properties of the
runs recorded here, not of the current code:

- The **target was not observable** to the RGB and role-stripped arms: the
  selector chose the lowest `track_id`. Fixed by selecting the leftmost object;
  the corpus is being re-collected, and every object-conditioning number above is
  superseded by that.
- The **gripper action carried the measured aperture**, leaving a decision margin
  smaller than the policies' own error on both arms (1.28x left, 3.48x right).
- **pi0.5 fitted padded action targets** -- 17.4% of all targets, 40.2% on the
  shortest clips -- where ACT masked them. Every pi0.5 row above trained on them.
- **Evaluation never recorded placements**, so the previous-neighbour role
  differed from training from the third transfer onward.

See `docs/findings.md` section 5 for the measurements behind each.

---

## 11. Comparing EE control to joint control

No valid comparison exists yet. The only EE-delta sweep ran against the broken
harness (section 7) and is void, so every EE number above this line is
unusable — and it was also a different backbone, a different corpus and a
different evaluation protocol from the joint runs.

### The confound to avoid

Section 5 shows that **delta versus absolute is a larger effect than anything
else measured** — it is the difference between 0/20 and 13/20 episodes lifting.
The existing EE implementation is delta-only. So a comparison of EE-delta
against joint-absolute would measure the delta/absolute axis and report it as
an EE/joint result.

The comparison therefore has to be a 2×2, with the representation and the
target type varied independently:

| | **delta** | **absolute** |
| --- | --- | --- |
| **joint** | `joint_delta` — have | `joint` — have |
| **end effector** | `cartesian` — implemented, needs an r75 run | `cartesian_absolute` — **needs implementing** |

`cartesian_absolute` emits the next frame's EE pose directly (position, unit
quaternion, gripper, per arm) and solves IK to it. It is simpler than the delta
path, which has to maintain a leashed reference and integrate: an absolute pose
needs no reference at all.

### Holding everything else fixed

Same as section 5, and for the same reason — every earlier cross-space
comparison in this document varied several things at once:

- ACT, training seed 1000, 40 000 steps, batch 8, chunk 50
- the r75 corpus: 75 train runs / 225 clips, 26 val runs / 78 clips
- best checkpoint by validation loss, saved every 2000 steps
- 20 held-out scene seeds, `three_object`, 600 steps, atomic scoring
- both conditionings (`rgb`, `privileged`) for every cell

### What will not be equalised, and why that is honest

- **IK is intrinsic to EE control.** The EE cells can emit infeasible commands;
  the joint cells cannot, by construction. That is a real property of choosing
  to act in task space, not a confound to remove. `infeasible_steps` is recorded
  per episode so the cost is visible rather than hidden inside the success rate.
- **The action widths differ**: 14-d for EE delta, 16-d for EE absolute and both
  joint spaces. Nothing can be done about this and it is not a fairness problem.
- **Quaternion regression is unconstrained.** An L1-trained head has no reason to
  emit a unit quaternion, so the absolute-EE path must normalise before solving
  IK. This is a genuine disadvantage of absolute pose targets and belongs in the
  result, not in a preprocessing step that hides it.
- **Oracle replay ceilings are close but not identical**: 6/6 for EE delta,
  19/20 for absolute joint, 5/6 for joint delta. Quote each cell against its own
  ceiling rather than against 100 %.

### Reading the result

The primary metric is the atomic one — `transfers >= 1` out of 20 — with
`mean lifted` as the secondary, since section 5 showed lift moves first and the
lift/transfer gap localises the next blocker. Paired over identical scene seeds,
so `scripts/aggregate_eval.py`'s McNemar test applies.

At n=20 the 95 % Wilson interval is roughly ±0.2, so only a large difference
will be readable. Treat this as a screen for "does task space cost or buy
anything large", not as a precise effect size.
