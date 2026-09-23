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

## 1. Protocol and arms, in brief

The full protocol is `data_protocol.md`, and the method is `method.md`.

- **Task.** Dual-arm shelf restock, trained on `three_object`. The identity
  holdout has three tiers:
  - seen: meshes 1, 2, 3, 4;
  - held-out: meshes 0 and 6, collected but excluded from the dataset;
  - novel: mesh 5, in no collected run.

  In this corpus a mesh differs only in size, so "held-out identity" means "an
  object size never seen".
- **Rollouts.** Scene seeds 800–819; `n_action_steps=25`.
  - Three objects run for 600 steps; 2 and 4 objects get 200 steps per object.
  - Every reset pins the tier, and the simulator verifies it.
- **Statistics.**
  - A cell is reported across ≥ 3 training seeds, as the mean with the min–max
    range.
  - Arms are compared by paired exact McNemar on matched (training seed, scene)
    episodes.
  - Tiers are reported separately only if some arm shifts on every seed.
  - Many contrasts are tested, so p < 0.05 is read as a lead.
- **Arms.** All conditioned arms fine-tune from the same seed's `rgb`
  checkpoint.

| arm | what it is |
| --- | --- |
| `rgb` | stage 1: stock policy, images + proprioception |
| `rgb_cont` | `rgb` continued for the steps stage 2 adds (the budget control) |
| `kv` | + ControlVLA's zero-init KV term at every hooked attention layer |
| `kv_adaln` | `kv` + a pooled scene vector modulating every block (LPWM-inspired) |
| `kv_tokens` | `kv` + the entities as tokens in the host's sequence (LPWM-inspired) |
| `scratch_kv` | `kv` with no stage 1 (ControlVLA's failing ablation) |

Before 2026-09-23 these were called `kv`, `kv_adaln`, `kv_tokens` and
`scratch_kv`. Result files carry the old names, and the collector reads both.

## 3. Known confounds and open controls

1. **Training budget.** Each stage-2 arm trains 40 k steps *on top of* `rgb`'s
   40 k. So every conditioned arm has seen twice the gradient steps, and part
   of any gain could be extra training rather than objects. **The control is
   `rgb` continued for 40 k more steps**, called `rgb_cont`. It uses the same
   fresh optimiser, and every arm is paired against it as well as against
   `rgb`. It is submitted. The training script used to ignore `ACT_PRETRAINED`
   for plain `act`, so a continued run would have silently retrained from
   scratch; that is fixed in `69c620b`. Until it runs, Q1 and Q2 are
   preliminary. `scratch_kv` gets the same 40 k as `rgb`, so the
   `scratch_kv` comparison is not confounded.
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
  seed, and `rgb`, `kv_adaln` and `kv_tokens` from seen to novel too. So the tiers
  are reported separately.
- **Caveat that still applies:** the budget control `rgb_cont` is still
  training (§3.1), so every "conditioned beats `rgb`" below may partly be
  extra training.

Mean transfers per episode, with the min–max across the 3 seeds:

| arm | seen | held-out | novel |
| --- | --- | --- | --- |
| rgb | 0.62 [0.45–0.90] | 0.03 [0.00–0.10] | 0.28 [0.15–0.55] |
| scratch_kv | 0.45 [0.30–0.55] | 0.05 [0.00–0.10] | 0.30 [0.05–0.60] |
| kv | 0.83 [0.55–1.35] | 0.22 [0.00–0.35] | 0.25 [0.00–0.75] |
| kv_adaln | 0.93 [0.75–1.10] | **0.55 [0.35–0.80]** | 0.23 [0.10–0.35] |
| kv_tokens | **1.25 [1.05–1.40]** | 0.13 [0.05–0.20] | 0.25 [0.00–0.75] |

Task success per 20 episodes (mean [min–max]):

| arm | seen | held-out | novel |
| --- | --- | --- | --- |
| rgb | 0.7 [0–1] | 0 | 0 |
| kv | 2.0 [0–5] | 0.3 [0–1] | 0.3 [0–1] |
| kv_adaln | 3.3 [2–5] | **2.0 [1–3]** | 0 |
| kv_tokens | **4.7 [2–8]** | 0 | 0.7 [0–2] |
| scratch_kv | 0.3 [0–1] | 0 | 0 |

Paired against `rgb` over 60 matched episodes per tier (exact McNemar; b/c =
episodes only the baseline / only the arm won):

| tier | arm | success | p | ≥1 transfer | p |
| --- | --- | --- | ---: | --- | ---: |
| seen | kv_adaln | 0.03 → 0.17 (2/10) | 0.039 | 0.45 → 0.48 (15/17) | 0.86 |
| seen | kv_tokens | 0.03 → 0.23 (1/13) | **0.002** | 0.45 → 0.65 (6/18) | 0.023 |
| held-out | kv | 0.00 → 0.02 (0/1) | 1.00 | 0.03 → 0.18 (1/10) | 0.012 |
| held-out | kv_adaln | 0.00 → 0.10 (0/6) | 0.031 | 0.03 → 0.33 (1/19) | **<0.001** |
| held-out | kv_tokens | 0.00 → 0.00 | 1.00 | 0.03 → 0.13 (1/7) | 0.070 |
| novel | any | no contrast below p = 0.2 | | | |

**Reading.**
- **Held-out objects are where conditioning matters most.** `rgb` all but
  stops transferring an object whose size it never saw: 0.03 mean transfers,
  down from 0.62. AdaLN keeps 0.55, with success on every seed. It is the
  largest effect in the project so far.
- On seen objects the conditioned arms help too, mostly in-context.
- **Conditioning without stage 1 does nothing** (`scratch_kv` ≈ `rgb` in every
  tier), matching ControlVLA's ablation.
- **Novel** (mesh 5, the one the oracle could never plan) is not helped by any
  arm.

### 4.2 Q2: which mechanism? (ACT)

| tier | contrast | success | p | ≥1 transfer | p |
| --- | --- | --- | ---: | --- | ---: |
| seen | kv → kv_tokens | 0.10 → 0.23 (3/11) | 0.057 | 0.52 → 0.65 | 0.19 |
| seen | kv_adaln → kv_tokens | 0.17 → 0.23 (6/10) | 0.45 | 0.48 → 0.65 (8/18) | 0.076 |
| held-out | kv → kv_adaln | 0.02 → 0.10 (1/6) | 0.13 | 0.18 → 0.33 (6/15) | 0.078 |
| held-out | kv_adaln → kv_tokens | 0.10 → 0.00 (6/0) | **0.031** | 0.33 → 0.13 (17/5) | **0.017** |

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
| kv | 0.13 [0.10–0.20] | 0.83 [0.55–1.35] | **0.42 [0.35–0.45]** |
| kv_adaln | 0.13 [0.05–0.25] | 0.93 [0.75–1.10] | 0.20 [0.15–0.25] |
| kv_tokens | 0.18 [0.10–0.25] | **1.25 [1.05–1.40]** | 0.28 [0.25–0.35] |
| scratch_kv | 0.12 [0.05–0.20] | 0.45 [0.30–0.55] | 0.32 [0.05–0.85] |

Success is 0 at four objects for every arm and seed. At two objects it is at
most 0.3/20.

**Reading.**
- **No arm generalises off the trained count,** and conditioning does not change
  that.
- At four objects the encoder-side arms are *worse* than `kv`: ≥1 transfer
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

### 4.5 Why the baseline is weak: stage-wise diagnostics (ACT, Stage A)

Before changing the conditioning again, three measurements, none needing new
rollouts.

**Stage-wise survival** (`collect_final_results.py`, STAGES section). Pooled
over 3 seeds × 20 scenes, three objects, seen identities:

| arm | lift | T1 \| lift | T2 \| T1 | T3 \| T2 | task success |
| --- | ---: | ---: | ---: | ---: | ---: |
| rgb | 0.78 | 0.57 | 0.30 | 0.25 | 0.03 |
| kv | 0.92 | 0.56 | 0.42 | 0.46 | 0.10 |
| kv_adaln | 0.90 | 0.54 | 0.59 | 0.59 | 0.17 |
| kv_tokens | 0.93 | 0.70 | 0.56 | 0.64 | 0.23 |

Reading:
- **The conditioned arms have flat per-stage survival.** For `kv_tokens` it is
  about 0.6 per stage, and the product (0.93 × 0.70 × 0.56 × 0.64 ≈ 0.23) *is*
  its task success. That is compounding local error, not a sequencing failure.
  ControlVLA's long-horizon tasks run at about 0.75–0.9 per stage.
- **`rgb` decays stage by stage** (0.57, 0.30, 0.25): a genuine sequencing
  problem, which conditioning partly repairs.
- **The weakest link in every arm is the first lift → transfer** (0.54–0.70).
  That points at grasp and release timing, not at object identity.
- **With two objects even the first transfer collapses** (T1 | lift about 0.2,
  against about 0.55 with three). That is a distribution shift from the trained
  count, not compounding.

**Atomic clips and ACT chunking.** Training cuts each run into three atomic
clips, and targets past a clip's end are masked (`action_is_pad`). With chunk
50:
- 17 % of all target positions are padding;
- 35 % of frames have less than a full 50-step target, and 17 % under half.

At rollout, 15 % of the 25-step executed blocks run past a clip's end, by 14.7
unsupervised actions on average. So the policy executes actions that had no
target in training, exactly around the switch to the next object.

**The CVAE latent collapses.** ACT's KL term falls to 0.005 by 10 k steps and
0.000 by 40 k, identically on all three seeds. The latent carries nothing.
ControlVLA attributes ACT's weakness in low data to exactly this.

### 4.6 The baseline rescue study (running)

Plain RGB ACT, one variable at a time, before any more conditioning:

| study | arm / prefix | what differs | status |
| --- | --- | --- | --- |
| continuous runs | `AF-rgb` vs `CF-rgb` | atomic clips vs full runs, recollected with the current code from the **same** 200 seeds, identical export | collection → build → view → train → rollouts queued |
| no CVAE | `F-rgb_novae` vs `F-rgb` | `use_vae=false`, same data and recipe | queued |

Both corpora are recollected because the current scene sampler draws one
random number more per object than the September collection did: same scene
distribution, but each seed's y positions differ. So the old atomic corpus is
not a like-for-like control.

**Next, only if these do not fix it** (in this order):
- ACT with two observations and a short chunk (16–24, executing 5–8);
- a gripper classification head, since lift → transfer is the weakest stage;
- a scene-geometric rather than history-based previous-neighbour rule, so the
  oracle's labels are Markov in the observation.

Object conditioning, including the 16 + 16 entity, returns after RGB ACT chains
the sequence reliably.

## 5. What answers each question, and what is running

Matrix prefixes: `F` absolute EE, `J` absolute joint, `D` joint delta, `X` EE
delta. VLA matrices put the backbone first: `P` pi0.5, `S` SmolVLA, `G` GR00T
(so `PF` is pi0.5 on absolute EE).

| # | runs | status |
| --- | --- | --- |
| Q1, Q2, Q3 (ACT) | `F`: rgb, kv, kv_adaln, kv_tokens, scratch_kv | **done**: 60 pinned rollouts (§4.1–4.3) |
| confound | `F`: rgb_cont × 3 seeds, 4 rollouts each | submitted |
| Q4 (ACT) | `J`, `D`, `X`: rgb × 3 seeds, 4 rollouts each | submitted |
| Q1 × Q4 (ACT) | the best conditioned arm(s) on `J`, `D`, `X` | waits for Q4 |
| Q4 (VLAs) | `slurm/submit_vla_experiments.sh PHASE=stage1`: 3 backbones × {F, J} × 3 seeds, seen rollout only | ready; end-to-end smoke tests running |
| Q1–Q3 (VLAs) | `PHASE=stage2`: rgb_cont + kv + the encoder arm(s) on each VLA's better regime | ready after stage 1 |

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

- **2026-09-23 (night).** Stage-wise diagnostics (§4.5). Conditioned arms fail
  by compounding local error, `rgb` by sequencing; lift → transfer is the
  weakest stage; atomic-clip padding and a collapsed CVAE confirmed. The
  baseline rescue study (§4.6) is queued.

- **2026-09-23 (evening).** Code cleanup (`cec6f52`).
  - Arms renamed `kv`, `kv_adaln`, `kv_tokens`, `scratch_kv`.
  - Method and protocol moved to `method.md` and `data_protocol.md`.
  - Stage A predictions are bit-identical before and after the cleanup.
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
