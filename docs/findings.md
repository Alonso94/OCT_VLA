# Shelf restocking: what has been measured, 2026-09-17 to 2026-09-18

Written to be quotable. Every number here was measured on this cluster against
the code at the commit named in §7; nothing is estimated, and where a result
overturns something this project previously believed, that is stated rather than
quietly corrected.

The short version: **the project produced its first non-zero closed-loop result**,
and the change that produced it is the one that looked worst by every offline
measure. Four separate hypotheses about the failure were tested and three were
refuted by their own experiments. A defect was then found that invalidates the
headline object-conditioning comparison as originally designed, and the corpus is
being re-collected to fix it.

---

## 1. Setup

| | |
| --- | --- |
| Task | Dual-arm Panda moves 3 objects from a lower to an upper shelf (RoboTwin/SAPIEN) |
| Control rate | 15 Hz; simulator inner loop 250 Hz |
| Corpus | 101 complete runs: 75 train / 225 atomic clips, 26 validation / 78 clips, 42 777 frames |
| Clip length | bimodal: 61–75 frames (first transfer), 170–191 (later transfers, compaction) |
| Backbones | ACT (40 k steps, batch 8, chunk 50, seed 1000) and pi0.5 (LoRA r=16, 8 k steps) |
| Evaluation | 20 held-out scene seeds, `three_object`, 600 steps |

Three levels of credit are reported, because a single binary hid everything:
**lift** (an object raised clear of the lower shelf), **transfer** (an object
restocked), **task** (all three restocked).

**Oracle replay ceilings**, policy-free, through the same socket and servo loop:
19/20 absolute joint, 5/6 joint delta, 5/6 absolute end-effector, 6/6 EE delta.
A perfect policy tops out near 95 %; quote each cell against its own ceiling.

---

## 2. The headline result

One backbone, one corpus, one seed, one protocol, varying only the action
encoding. 20 seeds per cell.

| observation | action encoding | transfers ≥1 | lifted ≥1 | mean lifted |
| --- | --- | ---: | ---: | ---: |
| privileged | joint delta | 0 | 0/20 | 0.00 |
| privileged | joint delta + velocity | 0 | 0/20 | 0.00 |
| **privileged** | **absolute joint** | **1** | **13/20** | **0.85** |
| rgb | joint delta | 0 | 0/20 | 0.00 |
| rgb | joint delta + velocity | 0 | 2/20 | 0.10 |
| rgb | EE delta | 0 | 0/20 | 0.00 |
| rgb | absolute EE | 0 | 0/20 | 0.00 |
| **rgb** | **absolute joint** | **2** | **13/20** | **0.95** |

Against roughly 1 600 prior episodes of uniform zeros, absolute joint targets
move the lift rate from 0.00 to 0.85–0.95 objects per episode and put 13 of 20
episodes in contact with the task. Every cell ran all 600 steps with zero
infeasible commands, so none of this is a harness failure.

It is a signal, not a working baseline: 5–10 % transfers against a ~95 % ceiling.

---

## 3. Offline metrics did not predict control

All losses below are validation L1 through `predict_action_chunk` — the inference
path, not the training log's `eval_loss`, which conditions the VAE encoder on the
ground-truth chunk and is therefore optimistic.

**Normalised losses are not comparable across control spaces.** Absolute joint
actions carry ~13× the MEAN_STD scale of increments, so the same physical error
reads as a smaller number. The radian column is the only cross-encoding unit.

| cell | constant | persistence | model | k=0 | k=49 | rad err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| joint delta, priv | 0.466 | 0.125 | 0.279 | 0.254 | 0.330 | 0.0058 |
| joint delta, rgb | 0.466 | 0.125 | 0.226 | 0.220 | 0.234 | 0.0050 |
| joint delta + velocity, priv | 0.466 | 0.125 | 0.233 | 0.193 | 0.271 | 0.0044 |
| joint delta + velocity, rgb | 0.466 | 0.125 | **0.188** | **0.153** | 0.199 | **0.0035** |
| absolute joint, priv | 0.839 | 0.029 | 0.163 | 0.089 | 0.205 | 0.0278 |
| absolute joint, rgb | 0.839 | 0.029 | 0.138 | 0.075 | 0.166 | 0.0234 |

Reference predictors, computed from the parquet with no model: **constant** is
the training mean of each dimension; **persistence** repeats the previous action.

The change that moved control has **5–7× the prediction error in radians** of the
one that did not. Five changes, tabulated:

| change | offline effect | closed-loop effect |
| --- | --- | --- |
| EE delta → absolute joint (pi0.5) | eval loss 0.0714 → 0.1253, worse | 0 → 0 |
| absolute joint → joint delta (pi0.5) | 0.1253 → 0.0605, best of three | 0 → 0 |
| 45 → 75 training runs (ACT) | 0.460 → 0.322, −30 % | 0 → 0 |
| position → +velocity state (ACT) | k=0 −24…−30 % | 0 → 0 |
| **joint delta → absolute joint (ACT)** | **rad err 5–7× worse** | **0 → 0.85–0.95 lift** |

**Interpretation.** Error *type* matters more than error size. An absolute target
is a position the servo absorbs: a 0.023 rad error leaves the arm ~1.2 cm from
where the demonstration was, and it stays there. A delta is a direction: the true
step is 0.0088 rad and the best model's error is 0.0035, so roughly 40 % of every
commanded step points the wrong way and the trajectory random-walks. The
conditioning argument that motivated joint deltas optimises the wrong quantity.

Treat held-out action loss as a check on "did the model learn anything", not as a
proxy for task success.

---

## 4. Four hypotheses tested

### 4.1 Open-loop execution — refuted

`n_action_steps` is inference-only, so existing checkpoints were re-scored at
four horizons with no retraining.

| cell | n=50 | n=25 | n=5 | n=1 | n=1 + ensembling |
| --- | ---: | ---: | ---: | ---: | ---: |
| privileged, lift ≥1 | 13 | **14** | 8 | **0** | 2 |
| rgb, lift ≥1 | 13 | **14** | 4 | 1 | 9 |
| rgb, mean lifted | 0.95 | **1.20** | 0.20 | 0.05 | 0.45 |

Shortening the horizon makes the policy monotonically worse; fully closed-loop
removes the behaviour entirely on the privileged cell. Temporal ensembling
(`coeff=0.01`, the original ACT setting) recovers part of it and still does not
reach `n=25`.

The mechanism is in §3: the offset curve is flat (k=0 0.089, k=49 0.205), so
there is no accurate near term to exploit. At `n=1` the policy executes
`action[0]` of a fresh chunk and discards 49; consecutive predictions disagree and
the arm jitters rather than committing. **`n=25` adopted** — 2× overlap, the ratio
LeRobot's configuration docstring describes.

### 4.2 Data volume — refuted

45 → 75 training runs cut ACT's validation loss 30 % (0.460 → 0.322) and left
closed-loop at zero. The encoding change then moved it with no extra data.

### 4.3 Action history in the observation — refuted closed-loop

Persistence (0.126) beats every trained model, and a ridge regression from the
*complete* privileged observation scores 0.563, worse than a constant. With
`joint_delta` the action is essentially a velocity and ACT sees one frame
(`n_obs_steps` is capped at 1), so a static snapshot does not determine it.
Appending the measured joint velocity cut next-step error 24–30 % — the largest
offline gain of any change tried — and produced 2 lifts and 0 transfers.

Not a copycat failure: a policy that simply repeated the visible previous action
would score 0.125; it scores 0.153, worse. It blends motion with scene
information; the blend does not help control.

### 4.4 Absolute-joint conditioning — real but not binding

One step is 0.0158 of the normalised output range against a prediction error of
~0.058, i.e. the error is ~3.7× the motion. Correct, and absolute joint is
nonetheless the only encoding that has produced anything non-zero.

---

## 5. Defects found and fixed

Each was confirmed against the code before being fixed.

| # | Defect | Evidence | Status |
| --- | --- | --- | --- |
| 1 | **Target not observable.** The selector chose the lowest `track_id`, an internal identifier absent from images and dropped by role stripping. | Leftmost object differs from lowest track id in ~50 % of scenes | Fixed: selector is leftmost-by-x; instruction names the rule; **corpus re-collecting** |
| 2 | **Gripper decision near-chance.** The action carried the measured aperture, making "open" a band 0.0108 wide holding 37–73 % of frames. | error ÷ margin **1.28×** left, **3.48×** right | Fixed: binary command at export, decode at 0.5, round-trip identical to the legacy decode |
| 3 | **pi0.5 fits padded actions.** Its loss ends at `losses.mean()`; ACT masks `action_is_pad`. | **17.4 %** of all targets, **40.2 %** on the shortest clips | Fixed: `masked_pi05` + masked `control_pi05` |
| 4 | **Evaluation never recorded placements.** `_placed_order` stayed empty, so `_previous_neighbor` returned the lexicographically smallest neighbour where training got the most recently placed. | Diverges from the 3rd transfer on | Fixed: server mirrors collection |
| 5 | **Two ACTION features.** A cartesian export from a joints-carrying corpus wrote `action.joint_position`, which LeRobot types as a second ACTION feature. | `{'action': (16,), 'action.joint_position': (16,)}` | Fixed: side channel opt-in |
| 6 | **Split extension ignored.** Seeds 350–399 were documented as train but absent from `SPLIT_BLOCKS`. | 32 → 75 train runs once honoured | Fixed |
| 7 | **Checkpoint selection.** Runs were scored at `last`, past their validation minimum. | privileged ACT +5.2 % worse at 40 k than 20 k | Fixed: `checkpoints/best` |

### Known-invalid, not fixed

- **The shuffled-token control is a no-op.** The object encoder is cross-attention
  over a set with no positional embedding, so permuting whole token rows cannot
  change the output. The control measured nothing. A valid control must swap
  roles between objects, zero the role dimensions, or use another scene's tokens.
- **The original EE-delta sweep is void** — it ran against a harness with five
  since-fixed defects (98.5 % of episodes ended `unreachable_pose`). The valid EE
  numbers are the ACT cells in §2.

---

## 6. Infrastructure built

- **Unified dataset.** One directory carries the canonical absolute-joint pair
  under the names LeRobot expects and every alternative under `alt.`, which
  LeRobot's feature typing skips. `scripts/project_dataset_view.py` derives any
  single-encoding view, hard-linking the video. All five views reproduce their
  source columns exactly (0.0e+00); stats move with their column, since a view
  holding increments under absolute-joint statistics would normalise by a scale
  5.9× too large.
- **Video deduplication.** Camera streams are byte-identical across variants;
  hard-linking took five RGB variants from 842 MB to 400 MB, and a new variant now
  costs ~7 MB instead of 168 MB.
- **Rollout videos.** One 40 s mp4 per cell (head + both wrists, captioned) in
  `docs/rollouts/`, which is what distinguishes "reached and missed" from "never
  approached".
- **Offline diagnostic.** `scripts/diagnose_action_head.py` reports per-offset and
  per-dimension validation L1 against constant and persistence baselines, in
  minutes and with no simulator.
- **FlashVLA.** `z-lab/flashvla-pi05-robotwin` (pi0.5 finetuned on RoboTwin 2.0)
  loads against our LeRobot 0.6.2 despite its `==0.5.1` pin: install `--no-deps`
  plus one line supplying `torch` to `transformers.configuration_utils`, which
  draccus needs to resolve a forward reference. Checkpoint matches the class
  exactly, 813/813 tensors.

---

## 7. What can and cannot be claimed

**Supported**

- Absolute joint targets beat joint increments closed-loop, in a comparison
  varying only the action encoding, with both conditionings agreeing.
- Long-horizon chunk execution is load-bearing; per-step re-querying is worse.
- Removing IK eliminates command infeasibility entirely (0 in 1 080 episodes
  against 29–441 per episode in Cartesian space).
- Held-out action loss has not predicted closed-loop success in five changes.
- Data volume is not the primary constraint at this scale.

**Not supported**

- **Anything about object-centric conditioning.** This is the project's headline
  question and it remains unanswered. Two of three observation arms were trained
  on an ill-posed mapping (§5.1), and the shuffled control was a no-op, so neither
  the comparison nor its control was valid. The re-collection addresses the first;
  the second needs a different control.
- That absolute joint targets *solve* the task. 5–10 % transfers against a ~95 %
  ceiling.
- Any EE-versus-joint conclusion at parity. The 2×2 (task/joint × delta/absolute)
  exists in code but only the joint half and the RGB EE cells have run.

**Open**

The lift/transfer gap — 13 of 20 episodes lift, 1–2 transfer — localises the
remaining failure to grasp-and-place rather than approach. The gripper fix (§5.2)
targets exactly that and has not yet been trained through.

---

## 8. Provenance

| | |
| --- | --- |
| Code | commit `e6da3f6`, branch `main` |
| ACT control-space cells | jobs 4268798–4268805, 2026-09-18 |
| Cell evaluation | jobs 4269720–4269725, 20 seeds each |
| Execution-horizon sweep | jobs 4270447–4270452, 4270671–4270672 |
| Offline diagnostics | jobs 4269708–4269713 |
| Unified datasets | jobs 4270561–4270562 |
| Leftmost re-collection | jobs 4270833–4270835, 200 array cells |
| Roots | `$HPCVAULT/octvla-{datasets,outputs}-jointdelta`, `-leftmost` |
| Detail | `docs/control_space_comparison.md`, `docs/dataset_reference.md`, `docs/rollouts/README.md` |
