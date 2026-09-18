# The experiments this project needs to run

What is still unanswered, which run answers it, and what it costs. Ordered by
information per GPU-hour, not by tidiness of the grid.

All of it runs on the **leftmost-target corpus** (109 runs: 84 train / 252 clips,
25 validation / 75 clips), exported once as a unified dataset with
`GRIPPER_ENCODING=binary_command` and projected into per-encoding views. Every
number already in `docs/findings.md` predates the target-observability fix and is
superseded by these.

Cost basis, measured on this cluster: **pi0.5-class ≈ 9.3 GPU-h** per 8 k-step
cell, **ACT-class ≈ 1 GPU-h** per 40 k-step cell, **evaluation ≈ 0.5 GPU-h** per
20-seed cell at 600 steps.

---

## Stage A — does the corpus fix change anything? (~14 GPU-h)

The single most important question, and the cheapest. Two of three observation
arms were previously trained on an ill-posed mapping; ACT absolute-joint is the
only configuration that has ever moved an object. Re-run it on the corrected
corpus before anything else.

| # | backbone | observation | action | why | cost |
| --- | --- | --- | --- | --- | ---: |
| A1 | ACT | rgb | absolute joint | The incumbent. Establishes whether the leftmost target and the binary gripper move it off 13/20 lift, 1–2 transfer. | 1.5 |
| A2 | ACT | privileged | absolute joint | Same, vision-free upper bound. | 1.0 |
| A3 | ACT | rgb | joint delta | Confirms the delta/absolute gap survives the corpus change. | 1.5 |
| A4 | ACT | rgb + objects | absolute joint | **The project's question**, now well-posed: object tokens as `environment_state`, no plugin. | 1.5 |
| A5 | ACT | rgb | absolute EE | The task-space arm, on the fixed bridge. | 1.5 |
| A6 | ACT | rgb | EE delta | Completes the 2×2 of {task, joint} × {delta, absolute}. | 1.5 |

**Gate:** if A1 does not improve on 13/20 lift, the gripper and target fixes did
not matter and the next lever is elsewhere. If A4 does not beat A1, object
conditioning does not help *this* policy, and the VLA arms are unlikely to rescue
it.

---

## Stage B — backbone comparison, RGB only (~35 GPU-h)

Held fixed: absolute-joint action, `n_action_steps=25`, seed 1000, the same
corpus. Varies only the backbone. No object conditioning yet, per the plan to run
these unconditioned first.

| # | backbone | params | cost |
| --- | --- | ---: | ---: |
| B1 | SmolVLA | 0.45 B | 3 |
| B2 | GR00T N1.7 | 3.14 B | 11 |
| B3 | VLA-JEPA | 2.77 B | 11 |
| B4 | pi0.5 | 3.62 B | 10 |

ACT from Stage A is the fifth arm and needs no rerun.

**Gate:** a backbone that does not beat the constant predictor offline
(`scripts/diagnose_action_head.py`, threshold 0.471) does not earn a rollout.
That check is minutes and saves half a GPU-hour each.

**What this settles:** ACT at 80 M currently beats pi0.5 at 3.6 B. If none of
B1–B4 beats ACT, scale and pretraining are not the axis, and the report can say
so with four independent points rather than one.

---

## Stage C — object conditioning across backbones (~50 GPU-h)

Only for backbones that clear Stage B. Each runs two arms.

| # | arm | what it isolates |
| --- | --- | --- |
| C1 | object_full | Does the scene state help at all? |
| C2 | object_role_stripped | Does it help *without* being told which object is the target? |

Run per surviving backbone (~11 GPU-h each for a VLA pair, ~3 for ACT).

**The control needs replacing.** The existing shuffled-token control is a
**no-op**: the object encoder is cross-attention over a set with no positional
embedding, so permuting whole token rows cannot change the output. A valid
control must swap the *role* dimensions between objects, zero them, or use
another scene's tokens. Without that, C1-vs-C2 is the only usable contrast.

---

## Stage D — generalisation (~12 GPU-h)

Only for a configuration that clears Stage A or B with non-zero transfers.
Scoring anything at zero across object counts measures nothing.

| # | run | axis |
| --- | --- | --- |
| D1 | best cell on `two_object` | fewer objects than trained |
| D2 | best cell on `four_object` | more objects than trained |

---

## Stage E — open questions, not yet scheduled

Each needs a decision before it is worth GPU time.

| # | question | blocked on |
| --- | --- | --- |
| E1 | Does training on **full-run** episodes fix the second-object failure? | A data re-export concatenating the three atomic clips per run. The likeliest explanation for "grasps the first object, fails the second" — the policy has never seen a target switch. |
| E2 | Does a **shorter chunk** help now that padding is masked? | Cheap to test at chunk 25 against 50; 17.4 % of targets were padding at chunk 50. |
| E3 | Does FastWAM's **RoboTwin pretraining** help? | A GPU larger than a 4090 — 6 B needs ~26.5 GB. The best domain match available. |
| E4 | Can a **reward model** rank checkpoints better than validation loss? | Motivated by held-out loss failing to predict closed-loop success five times running, but second-order while policies still cannot grasp. |
| E5 | More data. | 109 runs now. 45→75 runs cut validation loss 30 % and left closed-loop at zero, so this is not the top lever. |
| E6 | **X-VLA** or **MolmoAct2**, the two backbones considered and dropped. | X-VLA's released weights (`2toINF/X-VLA-Pt`) are a transformers `AutoModel` repo carrying `model_type` rather than LeRobot's `type`, so its implementation and its checkpoint do not meet without a conversion. MolmoAct2 is 5.44 B → ~23.9 GB, no headroom on a 24 GB card, and its `validate_features` injects `observation.state` with shape (0,) — it may not read proprioception at all. |

---

## Sequencing and total

| stage | GPU-h | gate to the next |
| --- | ---: | --- |
| A | ~14 | A1 improves, or stop and re-diagnose |
| B | ~35 | any backbone beats the constant offline |
| C | ~50 | a backbone clears Stage B |
| D | ~12 | something is non-zero |
| **total** | **~110** | |

Roughly a week of wall-clock at current queue depth, and about half of it is
contingent on Stage A. Every stage writes one rollout video per cell via
`slurm/submit_rollout_videos.sh`, because the counters cannot separate "reached
and missed" from "never approached".

## Standing conventions

These are settled by measurement and should not be re-litigated per run:

- **`n_action_steps=25`.** Shortening the execution horizon makes things
  monotonically worse; `n=1` removed the behaviour entirely on the privileged
  cell. 25 is 2× chunk overlap.
- **Absolute joint targets** unless the experiment is specifically about the
  action encoding. It is the only encoding that has produced a non-zero result.
- **`checkpoints/best`**, not `last` — runs pass their validation minimum well
  before the step budget ends.
- **Atomic scoring** (`transfers >= 1`) as the headline, with task success as the
  stretch metric. Training clips are single transfers.
- **Report against each cell's own oracle ceiling** (19/20 absolute joint, 5/6
  joint delta, 5/6 absolute EE, 6/6 EE delta), not against 100 %.
