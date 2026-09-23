# Research questions, the options tested, and where each answer stands

A living document: updated each time a batch of rollouts is collected. The
update log at the bottom says what changed and when. Every number here is
re-derived from `$OCTVLA_OUTPUT_ROOT/eval/*.json`, never copied from an earlier
report. This file supersedes the results sections of `report_2026_09.md` and
`project_status_2026_09_22.md` wherever they disagree.

**Last updated: 2026-09-23.**

## The four questions

| # | question | ACT | VLAs (pi0.5, SmolVLA, GR00T) |
| --- | --- | --- | --- |
| Q1 | Is object conditioning useful? | **preliminary yes**, see §4.1 | not answerable yet |
| Q2 | Which conditioning mechanism is best: layerwise, AdaLN, or in-context? | **preliminary**, see §4.2 | not answerable yet |
| Q3 | Does conditioning help compositional generalisation (train on 3 objects, test on 2, 3, 4)? | rollouts queued | not answerable yet |
| Q4 | Which control regime is best: absolute joint, joint delta, absolute EE, EE delta? | single-seed only, old corpus | single-seed and all zero |

Status meanings:
- **preliminary**: three training seeds, but with a known confound or an
  unfinished control.
- **answered**: three or more seeds, confounds controlled, and the paired test
  stated.

Nothing is marked answered yet.

---

## 1. The task and the evaluation protocol

**Task.** Dual-arm Franka Panda shelf restock in RoboTwin/SAPIEN. Move every
object from the lower shelf to the upper shelf, in order. Training profile:
`three_object`.

**Corpus.** The identity-holdout export `three_object_identity_*`:
- 222 training clips from 74 runs, and 63 IID validation clips.
- The asset `113_coffee-box` ships seven meshes (`model_id` 0–6). Four are
  trained on, two are held out of the dataset entirely, and one appears in no
  collected run.

| tier | model_ids | role |
| --- | --- | --- |
| seen | 1, 2, 3, 4 | in the training set |
| held-out | 0, 6 | collected, then excluded from the dataset |
| novel | 5 | in zero collected runs; the oracle cannot plan it |

In this corpus, geometry is identity. A mesh differs from the others only in
its size, so "held-out identity" means "an object size the policy never saw".

**Rollouts.**
- Scene seeds 800–819, outside every seed block reserved for data.
- `n_action_steps=25`.
- Three objects run for 600 steps.
- The object-count rollouts use 200 steps per object: 400 for 2 objects and
  800 for 4. The time per object stays equal to the 3-object runs, so a
  difference is not just a difference in time.

**Scoring.** Every episode reports four atomic measures:
- success: the whole shelf restocked;
- ≥1 transfer;
- ≥1 lift;
- mean transfers.

Infeasible IK steps are recorded separately.

**Statistics.**
- A cell is only reported across **three or more training seeds**, as the mean
  with the min–max range. One seed once turned 6/20 into 0/20 and 0/20.
- An arm is compared with `rgb` by **paired exact McNemar**. Pairs are matched
  on (training seed, scene seed): each conditioned arm fine-tunes from the same
  seed's `rgb` checkpoint and faces the same scenes.
- Many contrasts are tested, so p < 0.05 is read as a lead, not a finding.

**Proof of tier.**
- Each reset sends the mesh pin. The simulator reports which meshes it spawned
  and refuses a scene outside the pin.
- `scripts/collect_final_results.py` rejects any rollout that does not record
  both the pin and the spawned meshes.
- This exists because the first run of the matrix exported the pin after the
  simulator had started, so all 44 tier rollouts scored the same unpinned scenes.

**Reporting rule.**
- The identity tiers are shown separately only if some arm's held-out or novel
  score moves the same way as its seen score on every training seed. Otherwise
  they are pooled into one row, and the table says so.
- Object count is always shown.

**Videos.** There is one representative episode per reported row in
`rollouts/final/`, chosen by rule: the median training seed, then that seed's
most typical episode. See `rollouts/README.md`.

---

## 2. The options, how each is implemented, and why

### 2.1 Backbones

| backbone | params | how it is trained | where |
| --- | ---: | --- | --- |
| ACT | 0.08 B | from scratch, 40 k steps, batch 8, lr 1e-5, chunk 50 | `slurm/train_act_shelf_restock.sbatch` |
| SmolVLA | 0.45 B | LoRA on `lerobot/smolvla_base` | `slurm/train_shelf_restock.sbatch`, `BACKBONE=smolvla` |
| pi0.5 | 3.62 B | LoRA on `lerobot/pi05_base`, bf16 | same, `BACKBONE=pi05` |
| GR00T N1.7 | 3.16 B | full fine-tune, no PEFT | same, `BACKBONE=groot` |

ACT is the working substrate because it is the only backbone that has completed
a transfer at any rate, and it trains in about 1 GPU-hour. The VLAs cost 3–11
GPU-hours per cell.

### 2.2 What the object conditioning sees

The input is an **entity set** (`entity_v2`, `src/oct_vla/data/entity_tokens.py`):
up to 16 entities, each a 17-column geometric token, with a mask for empty slots.
The entities are the objects plus the support surfaces (the shelf decks).
Tokens are normalised with statistics fitted on the **training split only**. An
earlier version fitted them over training and validation together.

It is a set, not a flattened vector, and that choice is deliberate. The legacy
arm flattened 8 × 15 tokens into `environment_state`: 88 of its 120 columns had
zero variance in a three-object scene, and permuting the objects could not
change its output.

### 2.3 The conditioning mechanisms

All of these are **stage-2** arms. They fine-tune from the same seed's `rgb`
checkpoint and start as exactly that policy: every added path is
zero-initialised, and all 160 ACT tensors carry over (verified). This follows
ControlVLA (arXiv:2506.16211), which pretrains first and adds object
conditioning second. Its "without pretraining" ablation fails, and our
`scratch` arm reproduces that ablation deliberately.

| arm | what it adds to `rgb` | code |
| --- | --- | --- |
| `rgb` | nothing: stock ACT, images and proprioception | LeRobot `act` |
| `entity` | **layerwise** (ControlVLA as published): at every host cross-attention layer, `softmax(QKᵀ)V + softmax(QK_zᵀ)V_z` over the unpooled entity set, with `V_z` zero-initialised and the host's own dropout | `policies/layerwise_attention.py`, hooked in `control_act/modeling_control_act.py` |
| `adaln` | `entity` plus **AdaLN-Zero** (LPWM, arXiv:2603.04553): a scene vector modulates all 11 encoder and decoder blocks | same files, `object_adaln=true` |
| `incontext` | `entity` plus **in-context entity tokens** (LPWM): the tokens are appended to the encoder sequence and discarded after it | same files, `object_incontext=true` |
| `scratch` | `entity` conditioning trained from scratch, with no stage 1 | `ACT_PRETRAINED` unset |
| `rgb_cont` | nothing; `rgb` continued for the 40 k steps stage 2 adds (the **budget control**) | `ACT_POLICY_TYPE=act` with `ACT_PRETRAINED` |

**Why `adaln` and `incontext` exist.** LeRobot ACT has **one** decoder layer. So
`entity`, which is "layerwise", really adds one attention term at one seam. The
four encoder layers, where images and proprioception are fused, never see the
objects. Each LPWM arm adds exactly one encoder-side path, so comparing it
against `entity` isolates that one addition.

**AdaLN on a post-norm network.** ACT is post-norm, `x = LN(x + f(x))`. DiT's
AdaLN-Zero would multiply `f(x)` by α = 0 at the start, which deletes the
pretrained sublayer and destroys stage 1. We use an identity-centred form,
`x = LN(x + (1+α)·f(x))·(1+γ) + β`, with α = β = γ = 0 at step 0. That is
exactly the stage-1 policy. It is tested bit-for-bit (`rtol=atol=0`) in train
and eval mode, for both norm orders
(`tests/policies/test_control_act_encoder_branches.py`).
- Sites: 4 encoder layers × (self-attention, FFN), plus 1 decoder layer ×
  (self-attention, cross-attention, FFN), for 11 in all. The VAE encoder is
  excluded, even though it is also an `ACTEncoder`.
- The scene vector comes from its own entity embedding, then attention pooling
  with a learned query, then SiLU, then a zero-initialised Linear.
- An empty scene gives zero modulation.

**In-context tokens are not identity at initialisation.** The entity keys enter
every encoder softmax. A learned logit gate `b`, initialised to −4, holds them to
about e⁻⁴/S of the attention mass. The gradient into `b` stays alive; at −20 it
would be dead. The tokens are stripped before the decoder, so its memory and
positional embedding are exactly the host's.

**Legacy mechanisms,** kept so old checkpoints load:
- `controlvla`: a single seam at the action embedding.
- `pooled`: one broadcast bias from a pooled scene vector. This is strictly
  weaker, and it is what "ControlVLA" meant here before the published method
  was implemented.

**On the VLAs:**
- `layerwise`, `controlvla` and `pooled` are implemented for all three
  (`policies/layerwise_backbones.py`).
- `adaln` and `incontext` exist **only for ACT** so far.
- No VLA can yet run stage 2 from its own stage-1 checkpoint. pi0.5 and SmolVLA
  save LoRA adapters, and nothing loads an adapter into the conditioned policy.

### 2.4 Control regimes

The unified export carries every action encoding. Each regime is a *view*
projected from it (`src/oct_vla/data/control_views.py`,
`scripts/project_dataset_view.py`), with the videos hard-linked. So every
regime trains on the same episodes, frames and splits.

| regime | action | how the simulator executes it (`serve/server.py`) |
| --- | --- | --- |
| absolute joint | 16-d joint targets, 7 joints plus gripper per arm | commanded directly; cannot be infeasible |
| joint delta | 16-d joint increments | added to the measured joints |
| absolute EE | 16-d pose (position, quaternion, gripper) per arm | IK per step; an infeasible target holds the arm and is counted |
| EE delta | 14-d canonical increment | integrated into a command reference, with a 5 cm leash against wind-up, then IK |

Every regime uses a `binary_command` gripper, thresholded at export. Executing
the measured aperture as a command released grasped objects, and that is why
an early sweep scored exactly zero transfers.

---

## 3. Known confounds and open controls

1. **Training budget.** Each stage-2 arm trains 40 k steps *on top of* `rgb`'s
   40 k. So every conditioned arm has seen twice the gradient steps, and part
   of any gain could be extra training rather than objects. **The control is
   `rgb` continued for 40 k more steps**, called `rgb_cont`. It uses the same
   fresh optimiser, and every arm is paired against it as well as against
   `rgb`. It is submitted. The training script used to ignore `ACT_PRETRAINED`
   for plain `act`, so a continued run would have silently retrained from
   scratch; that is fixed in `69c620b`. Until it runs, Q1 and Q2 are
   preliminary. `scratch` gets the same 40 k as `rgb`, so the
   `scratch` comparison is not confounded.
2. **Validation loss.** Validation loss is not used to rank cells or to pick
   checkpoints. It has contradicted closed-loop success nine times. Every cell
   is scored from `checkpoints/last`.
3. **Oracle ceiling.** Scores are read against each cell's own oracle ceiling,
   not against 100 %.

---

## 4. Current results

### 4.1 Q1: is object conditioning useful? (ACT, absolute EE)

**Source.** The first run of the matrix, 3 seeds × 20 scenes, three objects.
These rollouts were *unpinned*, so each scene drew from all seven meshes. They
are valid as three-object rollouts on mixed identities, and **void as tier
results**. The files are in `eval/void_unpinned_2026_09_22/`. Pinned replacements
are queued.

Each cell shows successes / episodes with ≥1 transfer / mean transfers.

| arm | s1000 | s1001 | s1002 | success, 60 episodes | ≥1 transfer, 60 episodes |
| --- | --- | --- | --- | ---: | ---: |
| rgb | 0 / 10 / 0.60 | 0 / 3 / 0.15 | 0 / 1 / 0.05 | 0 | 14 |
| scratch | 0 / 5 / 0.25 | 0 / 5 / 0.25 | 0 / 5 / 0.25 | 0 | 15 |
| entity | 1 / 9 / 0.65 | 0 / 4 / 0.20 | 1 / 7 / 0.50 | 2 | 20 |
| adaln | 5 / 11 / 1.05 | 1 / 7 / 0.50 | 2 / 4 / 0.40 | **8** | 22 |
| incontext | 3 / 12 / 0.95 | 0 / 7 / 0.40 | 2 / 6 / 0.50 | 5 | **25** |

Paired against `rgb` over 60 matched episodes. b = episodes only `rgb` won;
c = episodes only the arm won.

| arm | success rgb → arm | p | ≥1 transfer rgb → arm | p |
| --- | --- | ---: | --- | ---: |
| entity | 0 → 2 (b 0, c 2) | 0.50 | 14 → 20 (b 7, c 13) | 0.26 |
| adaln | 0 → 8 (b 0, c 8) | **0.008** | 14 → 22 (b 8, c 16) | 0.15 |
| incontext | 0 → 5 (b 0, c 5) | 0.063 | 14 → 25 (b 4, c 15) | **0.019** |
| scratch | 0 → 0 | 1.00 | 14 → 15 (b 8, c 9) | 1.00 |

**Reading.**
- For the first time in this project, conditioned arms beat `rgb` on every seed
  by mean transfers. The two encoder-side arms also clear p < 0.05 on one
  measure each.
- **Conditioning without stage 1 does nothing.** `scratch` equals `rgb`,
  matching ControlVLA's ablation.
- **Not yet an answer.** The training-budget confound (§3.1) is uncontrolled.
- The earlier single-seed results pointed the same way but proved nothing: C2
  (old corpus, pooled objects) scored 2/20 success against A5's 6/20.

### 4.2 Q2: which mechanism? (ACT)

| contrast | success | p | ≥1 transfer | p |
| --- | --- | ---: | --- | ---: |
| adaln vs entity | 2 → 8 (b 1, c 7) | 0.070 | 20 → 22 | 0.83 |
| incontext vs entity | 2 → 5 (b 2, c 5) | 0.45 | 20 → 25 | 0.38 |

**Reading.**
- Both encoder-side arms beat layerwise alone numerically. Neither separates
  from it at n = 60.
- `adaln` leads on success; `incontext` leads on reaching a transfer.
- A ranking between them is not supported.
- **VLAs:** there is no valid comparison. The one VLA mechanism comparison
  (GR00T F0/F1/F2: none / `controlvla` / `pooled`) scored 0/20 success in every
  cell, because its own baseline was at zero.

### 4.3 Q3: compositional generalisation (2 / 3 / 4 objects)

**Pinned, three-seed rollouts are queued.** The only existing data is
single-seed, on the old corpus, at a fixed 600 steps whatever the object count:

| cell (seed 1000) | 2 objects | 3 objects | 4 objects |
| --- | --- | --- | --- |
| A5: rgb, absolute EE | 0 success, 2 transfer, 0.10 | 6, 7, 1.00 | 0, 4, 0.20 |
| C2: rgb + objects (pooled) | 1, 2, 0.15 | 2, 10, 0.85 | 0, 9, 0.45 |
| C3: objects only, no cameras | 0, 0, 0.00 | 0, 1, 0.05 | 0, 0, 0.00 |

Each cell shows success / ≥1 transfer / mean transfers, out of 20.

Success collapses off the trained count for every cell. The one hint is C2's 9/20
transfers at four objects against A5's 4/20, and it is single-seed.

### 4.4 Q4: control regime

**ACT, RGB, old leftmost corpus.** Seed 1000 only, except absolute EE, which has
three seeds.

| regime | cell | success | ≥1 transfer | ≥1 lift | mean transfers | infeasible steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| absolute joint | A1 | 0/20 | 3/20 | 15/20 | 0.15 | 0 |
| joint delta | A3 | 0/20 | 1/20 | 2/20 | 0.05 | 0 |
| absolute EE | A5 / R1 / R2 (s1000 / 1001 / 1002) | 6 / 0 / 0 | 7 / 1 / 1 | 17 / 7 / 4 | 1.00 / 0.05 / 0.05 | 137 / 138 / 335 |
| EE delta | A6 | 0/20 | 0/20 | 0/20 | 0.00 | 67 |

Absolute EE is the only regime with any success, and it failed to replicate on
two of three seeds. Across its three seeds it averages 0.37 mean transfers,
against absolute joint's single-seed 0.15. **No regime ranking survives.** The
delta regimes are weakest on every measure.

On the identity corpus, `rgb` absolute EE scores 0.60 / 0.15 / 0.05 mean
transfers across seeds (§4.1). That is the same regime, and it shows the same
spread.

**VLAs, RGB, seed 1000, 6–8 k steps.**

| backbone | regime | success | ≥1 transfer | ≥1 lift |
| --- | --- | ---: | ---: | ---: |
| SmolVLA | absolute joint | 0/20 | 0/20 | 5/20 |
| GR00T | absolute joint | 0/20 | 1/20 | 12/20 |
| pi0.5 | absolute joint | 0/20 | 0/20 | 5/20 |
| GR00T | absolute EE | 0/20 | 2/20 | 12/20 |

No VLA has succeeded in any regime or with any mechanism. No regime comparison
is possible from rates at zero.

---

## 5. What answers each question, and what is running

Matrix prefixes: `F` absolute EE, `J` absolute joint, `D` joint delta, `X` EE
delta. Each cell is 3 seeds × 4 pinned rollouts: three-object seen, held-out and
novel, plus a `count` job at 2 and 4 objects.

| # | runs | status |
| --- | --- | --- |
| Q1, Q2, Q3 (ACT) | `F`: rgb, rgb_cont, entity, adaln, incontext, scratch | **submitted.** rgb_cont trains first; the other five reuse existing checkpoints |
| Q4 (ACT) | `J`, `D`, `X`: rgb × 3 seeds, on views projected from the same unified identity export | **submitted**, after the projection job |
| Q1 × Q4 (ACT) | the best Stage A conditioned arm, plus rgb_cont, on `J`, `D`, `X` | waits for the Stage A winner |
| Q4 (VLAs) | SmolVLA, pi0.5, GR00T × {absolute joint, absolute EE} × 3 seeds, RGB. The delta regimes are dropped: they have been the weakest for every backbone | code first (below), then after Stage A |
| Q1, Q2 (VLAs) | on each VLA's better regime: layerwise, plus ACT's best encoder-side arm, × 3 seeds | code first (below) |

VLA prerequisites, none written yet:
- an adapter-aware stage-2 loader: merge the stage-1 LoRA, load it into
  `control_*`, and add a fresh adapter, tested to be exactly stage 1 at step 0;
- GR00T's action padding (16 → 132 columns) and missing `action_is_pad` mask;
- GR00T's broken `checkpoints/last` link;
- a port of ACT's best encoder-side arm;
- a VLA path in `submit_final_experiments.sh`.

The trimmed VLA plan is about 250 GPU-h.

## Update log

- **2026-09-23.** Submitted `rgb_cont` (the budget control) and `rgb` on the
  three other control regimes, 12 training cells and 48 rollouts. The VLA
  scope is set to the trimmed plan.
- **2026-09-23.** Document created.
  - Q1/Q2 are preliminary, from unpinned three-seed rollouts.
  - Found and fixed the identity-tier bug; pinned tier and count rollouts are
    queued (60 jobs).
  - Identified the training-budget confound, whose control (`rgb_cont`) is not
    yet run.
