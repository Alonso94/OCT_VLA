# Control spaces for shelf restocking: EE deltas, absolute joints, joint deltas

Everything needed to write this up: what was measured, under which conditions,
and which numbers are not yet in. Cells marked **TBD** are pending runs, not
omissions.

Read the infrastructure section before the results. Two of the three sweeps ran
against an evaluation harness that has since been shown broken, and their
closed-loop numbers mean nothing without that caveat.

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

| | EE delta | Absolute joint | Joint delta |
| --- | ---: | ---: | ---: |
| Mean motion per step | 11.32 mm | 0.00897 rad | 0.00900 rad |
| Std of the action column | — | 0.30064 rad | **0.02279 rad** |
| **One step, in normalised units** | — (delta-encoded) | **0.0298 σ** | **0.3949 σ** |
| Resolution vs absolute | — | 1× | **13×** |

Measured on 19 005 frames of the joint corpus. Reconstruction check for the
delta encoding: `state[i] + action[i] − state[i+1]` has max error **1.49e-08**,
and the gripper stays absolute (range 0.535–0.833), not differenced.

### What the policy actually predicts

Measured on *training* frames, so this is not a generalisation effect.

| | EE delta | Absolute joint | Joint delta |
| --- | ---: | ---: | ---: |
| Oracle motion per step | 11.32 mm | 0.00825 rad | TBD |
| Policy motion commanded | 9.91 mm | **0.02969 rad** (3.6× too much) | TBD |
| Mean prediction error | 0.0047 (mixed units) | **0.03291 rad** (4× the step) | TBD |

For absolute joints the prediction error **exceeds the motion being predicted**.
That is the mechanism behind the zero closed-loop result, and it is a property
of the encoding, not of model capacity or data volume.

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

**TBD** — datasets built and verified; training not yet run.

### Summary

| | EE delta | Absolute joint | Joint delta |
| --- | ---: | ---: | ---: |
| Train loss (best arm) | 0.037 | **0.029** | TBD |
| Eval loss (best arm) | **0.0714** | 0.1253 | TBD |
| Train→eval gap | **0.03** | 0.09–0.12 | TBD |

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
| Success, all arms | 1 / 1080 | 0 / 1080 | TBD |
| Mean transfers | 0.00 | 0.00 | TBD |
| Episodes ending `unreachable_pose` | **1064 (98.5 %)** | 0 | TBD |
| Episodes reaching the step limit | 15 | **1080 (100 %)** | TBD |
| Infeasible commands per episode | 29–441 | **0** | TBD |

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
| After harness fixes | 6 / 6 | **19 / 20 (~95 %)** | TBD |
| Worst joint tracking error | ~9.4 mm (Cartesian) | 0.0065–0.0137 rad | TBD |

The one persistent failure (seed 112) is deterministic across repeats: replayed
joint targets do not reproduce the oracle's *execution dynamics*, and a
marginal grasp drops.

---

## 5. Infrastructure caveats — read before quoting any EE-delta number

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

## 6. What can and cannot be claimed

**Supported by the data:**

- Absolute joint targets are badly conditioned for this task at this control
  rate, quantitatively: one step is 0.03 σ, and the measured prediction error is
  4× the motion being predicted.
- Removing IK from the loop eliminates command infeasibility entirely — 0 in
  1080 episodes, against 29–441 per episode in the Cartesian space.
- The evaluation harness, not the policies, produced the first sweep's result.

**Not yet supported:**

- Anything about object-centric conditioning. All arms score zero in every
  completed sweep, so the Stage 2 gate (does B beat A? does C survive?) is
  unanswered. The offline losses hint that the role channel carries signal, but
  that is not the closed-loop claim.
- That joint deltas fix the task. The conditioning argument predicts a large
  improvement in prediction error; whether that converts into transfers is TBD.

**Open alternative explanation:** 96 training episodes may simply be too few.
A 20 k-step checkpoint (2.5× training) drove infeasible commands to ~0 yet still
transferred nothing, including on a seed it was trained on. If joint deltas do
not move the prediction error below the per-step motion, the next variable to
change is data volume, not the action space.

---

## 7. Provenance

| | |
| --- | --- |
| EE-delta sweep | job 4251265, 2026-09-15, `$HPCVAULT/octvla-outputs` |
| Absolute-joint sweep | job 4261431, 2026-09-16, `$HPCVAULT/octvla-outputs-jointspace` |
| Joint-delta sweep | TBD, `$HPCVAULT/octvla-outputs-jointdelta` |
| Joint corpus | 135 episodes, 45 seeds, 19 005 frames, all successful |
| Cluster envelope | `docs/cluster_envelope.md` |
| Experiment design | `docs/experiment_sweep.md` |
