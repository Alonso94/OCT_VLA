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
| Q1 | Is object conditioning useful? | **preliminary yes**, most of all on held-out objects (§4.1). Waits on the `rgb_cont` budget control | pipeline ready, not run |
| Q2 | Which conditioning mechanism is best: layerwise, AdaLN, or in-context? | **preliminary: depends on the object.** In-context on seen objects, AdaLN on held-out ones (§4.2) | pipeline ready, not run |
| Q3 | Does conditioning help compositional generalisation (train on 3 objects, test on 2, 3, 4)? | **preliminary no.** Every arm collapses at 2 and 4 objects, and the encoder-side arms are worse at 4 (§4.3) | pipeline ready, not run |
| Q4 | Which control regime is best: absolute joint, joint delta, absolute EE, EE delta? | three-seed runs on the identity corpus submitted | stage 1 ready to submit |

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

**Videos.** There is one representative episode per reported row, chosen by
rule: the median training seed, then that seed's most typical episode. The
table is `rollouts/final/table.md`; the videos are on the vault
(`rollouts/README.md` says why).

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

**On the VLAs** (`policies/vla_branches.py`, `control_*/modeling_*.py`):
- `layerwise`, `controlvla` and `pooled` are implemented for all three
  (`policies/layerwise_backbones.py`).
- **AdaLN.** pi0.5's action expert and GR00T's DiT already modulate every
  block's adaptive norm from a condition vector, the flow-matching timestep. The
  pooled scene is added to that vector through a zero-initialised projection
  (`SceneVector`), so the scene conditions every block through the pretrained
  modulation layers, as DiT conditions on a class label. It is exact identity
  at init. SmolVLA's expert has no adaptive norm, so it gets a zero-initialised
  FiLM, `out·(1+γ)+β`, on every expert RMSNorm.
- **In-context.** Entity tokens are prepended to the action-model sequence and
  read back out by position.
  - pi0.5 and SmolVLA: in the suffix, as their own attention block. Actions see
    them; they never see the noisy actions; padded slots are masked out.
  - GR00T's DiT takes no attention mask, so only real entities are inserted.
    That requires an equal count per row of a batch, which is checked and holds
    for each object-count profile.
  - As for ACT, this is not identity at init.
- **Stage 2 from stage 1.**
  - pi0.5 and SmolVLA stage 1 is a LoRA adapter. `scripts/merge_stage1_adapter.py`
    merges it into its base. It refuses a zero adapter, a merge that changes the
    predicted chunk, or a checkpoint that reloads differently.
  - GR00T stage 1 is full weights and loads directly.
  - Every such load is checked tensor by tensor against its file
    (`policies/stage_loading.py`). pi0.5's own loader returns an unloaded model
    on failure, with only a warning.
  - `INIT_CHECK=1` builds a stage-2 run exactly as training does and proves it
    starts at stage 1 and that gradient reaches its branch. Every stage-2
    training is queued behind that check.

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

**Source.**
- 5 arms × 3 training seeds × 20 scenes, with every tier pinned and verified
  (`diagnostics/final_matrix_F.json`).
- The tiers are **informative**: every arm drops from seen to held-out on every
  seed, and `rgb`, `adaln` and `incontext` from seen to novel too. So the tiers
  are reported separately.
- **Caveat that still applies:** the budget control `rgb_cont` is still
  training (§3.1), so every "conditioned beats `rgb`" below may partly be
  extra training.

Mean transfers per episode, with the min–max across the 3 seeds:

| arm | seen | held-out | novel |
| --- | --- | --- | --- |
| rgb | 0.62 [0.45–0.90] | 0.03 [0.00–0.10] | 0.28 [0.15–0.55] |
| scratch | 0.45 [0.30–0.55] | 0.05 [0.00–0.10] | 0.30 [0.05–0.60] |
| entity | 0.83 [0.55–1.35] | 0.22 [0.00–0.35] | 0.25 [0.00–0.75] |
| adaln | 0.93 [0.75–1.10] | **0.55 [0.35–0.80]** | 0.23 [0.10–0.35] |
| incontext | **1.25 [1.05–1.40]** | 0.13 [0.05–0.20] | 0.25 [0.00–0.75] |

Task success per 20 episodes (mean [min–max]):

| arm | seen | held-out | novel |
| --- | --- | --- | --- |
| rgb | 0.7 [0–1] | 0 | 0 |
| entity | 2.0 [0–5] | 0.3 [0–1] | 0.3 [0–1] |
| adaln | 3.3 [2–5] | **2.0 [1–3]** | 0 |
| incontext | **4.7 [2–8]** | 0 | 0.7 [0–2] |
| scratch | 0.3 [0–1] | 0 | 0 |

Paired against `rgb` over 60 matched episodes per tier (exact McNemar; b/c =
episodes only the baseline / only the arm won):

| tier | arm | success | p | ≥1 transfer | p |
| --- | --- | --- | ---: | --- | ---: |
| seen | adaln | 0.03 → 0.17 (2/10) | 0.039 | 0.45 → 0.48 (15/17) | 0.86 |
| seen | incontext | 0.03 → 0.23 (1/13) | **0.002** | 0.45 → 0.65 (6/18) | 0.023 |
| held-out | entity | 0.00 → 0.02 (0/1) | 1.00 | 0.03 → 0.18 (1/10) | 0.012 |
| held-out | adaln | 0.00 → 0.10 (0/6) | 0.031 | 0.03 → 0.33 (1/19) | **<0.001** |
| held-out | incontext | 0.00 → 0.00 | 1.00 | 0.03 → 0.13 (1/7) | 0.070 |
| novel | any | no contrast below p = 0.2 | | | |

**Reading.**
- **Held-out objects are where conditioning matters most.** `rgb` all but
  stops transferring an object whose size it never saw: 0.03 mean transfers,
  down from 0.62. AdaLN keeps 0.55, with success on every seed. It is the
  largest effect in the project so far.
- On seen objects the conditioned arms help too, mostly in-context.
- **Conditioning without stage 1 does nothing** (`scratch` ≈ `rgb` in every
  tier), matching ControlVLA's ablation.
- **Novel** (mesh 5, the one the oracle could never plan) is not helped by any
  arm.

### 4.2 Q2: which mechanism? (ACT)

| tier | contrast | success | p | ≥1 transfer | p |
| --- | --- | --- | ---: | --- | ---: |
| seen | entity → incontext | 0.10 → 0.23 (3/11) | 0.057 | 0.52 → 0.65 | 0.19 |
| seen | adaln → incontext | 0.17 → 0.23 (6/10) | 0.45 | 0.48 → 0.65 (8/18) | 0.076 |
| held-out | entity → adaln | 0.02 → 0.10 (1/6) | 0.13 | 0.18 → 0.33 (6/15) | 0.078 |
| held-out | adaln → incontext | 0.10 → 0.00 (6/0) | **0.031** | 0.33 → 0.13 (17/5) | **0.017** |

**Reading.** The two encoder-side arms trade places by tier:
- **In-context is best on seen objects.**
- **AdaLN is best on held-out ones**, and beats in-context there on both
  measures.

One reading, not yet tested: in-context gives the encoder per-object tokens,
which is enough to memorise the four trained sizes. AdaLN gives a pooled scene
summary that modulates every block, and that transfers to sizes it never saw.

**For the VLAs:** no valid comparison exists yet. The one earlier VLA mechanism
comparison (GR00T F0/F1/F2) scored 0/20 success in every cell, because its own
baseline was at zero.

### 4.3 Q3: compositional generalisation (2 / 3 / 4 objects)

Trained on three objects; seen identities; 200 steps per object. Mean
transfers per episode [min–max across seeds]:

| arm | 2 objects | 3 objects | 4 objects |
| --- | --- | --- | --- |
| rgb | 0.20 [0.15–0.25] | 0.62 [0.45–0.90] | 0.40 [0.30–0.50] |
| entity | 0.13 [0.10–0.20] | 0.83 [0.55–1.35] | **0.42 [0.35–0.45]** |
| adaln | 0.13 [0.05–0.25] | 0.93 [0.75–1.10] | 0.20 [0.15–0.25] |
| incontext | 0.18 [0.10–0.25] | **1.25 [1.05–1.40]** | 0.28 [0.25–0.35] |
| scratch | 0.12 [0.05–0.20] | 0.45 [0.30–0.55] | 0.32 [0.05–0.85] |

Success is 0 at four objects for every arm and seed. At two objects it is at
most 0.3/20.

**Reading.**
- **No arm generalises off the trained count,** and conditioning does not change
  that.
- At four objects the encoder-side arms are *worse* than `entity`: ≥1 transfer
  0.40 → 0.18 for both, p = 0.019 (AdaLN) and p = 0.015 (in-context).
- **Two objects is harder than three for every arm.** The trained count is the
  best case, both above and below it.

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
delta. VLA matrices put the backbone first: `P` pi0.5, `S` SmolVLA, `G` GR00T
(so `PF` is pi0.5 on absolute EE).

| # | runs | status |
| --- | --- | --- |
| Q1, Q2, Q3 (ACT) | `F`: rgb, entity, adaln, incontext, scratch | **done**: 60 pinned rollouts (§4.1–4.3) |
| confound | `F`: rgb_cont × 3 seeds, 4 rollouts each | submitted |
| Q4 (ACT) | `J`, `D`, `X`: rgb × 3 seeds, 4 rollouts each | submitted |
| Q1 × Q4 (ACT) | the best conditioned arm(s) on `J`, `D`, `X` | waits for Q4 |
| Q4 (VLAs) | `slurm/submit_vla_experiments.sh PHASE=stage1`: 3 backbones × {F, J} × 3 seeds, seen rollout only | ready; end-to-end smoke tests running |
| Q1–Q3 (VLAs) | `PHASE=stage2`: rgb_cont + entity + the encoder arm(s) on each VLA's better regime | ready after stage 1 |

**Storage rules**, since the vault has 1 TB for everything:
- Every run keeps only its final checkpoint.
- Every run drops its optimiser state on success.
- GR00T saves only its trained head (`slim_groot`). Its frozen backbone was
  measured bit-identical to the base model. A full save was about 49 GB on disk
  per checkpoint.
- Merged pi0.5/SmolVLA checkpoints are made only for the regime that goes to
  stage 2.
- Rollout videos stay on the vault: home allocates 32 MB per file.

## Update log

- **2026-09-23 (afternoon).** Pinned Stage A results for Q1–Q3.
  - Tiers are informative. In-context is best on seen objects, AdaLN on
    held-out ones. No arm generalises across object count.
  - The VLA two-stage pipeline is written and tested: verified loads, the
    merge, init checks, AdaLN/in-context ports, slim GR00T checkpoints, and
    storage rules. It is being smoke-tested end to end.
  - Freed ~380 GB of intermediate checkpoints and optimiser state. Every
    checkpoint a result names was kept.

- **2026-09-23.** Submitted `rgb_cont` (the budget control) and `rgb` on the
  three other control regimes, 12 training cells and 48 rollouts. The VLA
  scope is set to the trimmed plan.
- **2026-09-23.** Document created.
  - Q1/Q2 are preliminary, from unpinned three-seed rollouts.
  - Found and fixed the identity-tier bug; pinned tier and count rollouts are
    queued (60 jobs).
  - Identified the training-budget confound, whose control (`rgb_cont`) is not
    yet run.
