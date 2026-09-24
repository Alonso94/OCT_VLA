# Research questions, the options tested, and where each answer stands

A living document: updated each time a batch of rollouts is collected. The
update log at the bottom says what changed and when. Every number here is
re-derived from `$OCTVLA_OUTPUT_ROOT/eval/*.json`, never copied from an earlier
report.

**Last updated: 2026-09-24.**

## The four questions, and the Stage A answers

Stage A (ACT, 3 training seeds × 20 pinned scenes per tier, every arm against
the budget control) is **complete**; the VLAs (Stage B) have not been trained.
All Stage A arms learned from **atomic clips**; §4.6 shows that training on
continuous runs changes how well a policy chains transfers, which bounds what
the Stage A answers can claim (see "Scope" under each).

| # | question | ACT (Stage A) | VLAs |
| --- | --- | --- | --- |
| Q1 | Is object conditioning useful? | **Only for held-out objects, and only through AdaLN.** Against the budget control, no arm improves seen objects; `kv_adaln` keeps transferring objects of sizes it never saw (§4.1) | not trained |
| Q2 | Which mechanism: KV, AdaLN, or in-context tokens? | **AdaLN.** It is the only mechanism that beats the budget control anywhere, and it beats tokens on held-out objects (§4.2) | not trained; `kv_tokens` reworked first (§4.7) |
| Q3 | Does conditioning help compositional generalisation (train on 3, test on 2 and 4)? | **No.** No arm beats the budget control at 2 or 4 objects, and the encoder-side arms are worse than `kv` at 4. The count gap is largely a training-format effect: trained on full runs, plain RGB ACT does better on 2 objects than on 3 (§4.3, §4.6) | not trained |
| Q4 | Which control regime: absolute joint, joint delta, absolute EE, EE delta? | **Absolute, not delta.** Both delta regimes complete 0 transfers in 180 episodes each. Absolute EE and absolute joint cannot be separated (§4.4) | not trained |

Status meanings: **answered** = ≥ 3 seeds, the budget control run, the paired
test stated, on the stated scope. With about 30 paired contrasts per matrix,
p < 0.05 is read as a lead, and a claim is made only where the effect also
holds on every training seed.

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
| `rgb_novae` | `rgb` with the CVAE latent off (`use_vae=false`) |
| `rgb_short` | `rgb` with a 20-step chunk executed 8 at a time |
| `rgb_hist` | `rgb_short` over a two-frame observation history (`history_act`) |

Prefixes name the matrix: `F`, `J`, `D`, `X` are the September identity corpus
(atomic clips) under absolute EE, absolute joint, joint delta and EE delta;
`AF` and `CF` are the paired corpus of §4.6 as atomic clips and as full runs.

Before 2026-09-23 the conditioned arms were called `entity`, `adaln`,
`incontext` and `scratch`. Result files carry the old names, and the collector
reads both, preferring a re-run under the new name.

## 3. Known confounds and open controls

1. **Training budget (controlled).** Each stage-2 arm trains 40 k steps *on
   top of* `rgb`'s 40 k. The control is `rgb` continued for the same 40 k,
   `rgb_cont`, and every arm is paired against it. It matters: most of what
   the conditioned arms gained over `rgb` on seen objects, `rgb_cont` gains
   too (§4.1). `scratch_kv` gets the same 40 k as `rgb`.
2. **Validation loss.** Validation loss is not used to rank cells or to pick
   checkpoints. It has contradicted closed-loop success nine times. Every cell
   is scored from `checkpoints/last`.
3. **Oracle ceiling.** Scores are read against each cell's own oracle ceiling,
   not against 100 %.
4. **Two-object layout.** The spawn sampler reserves 0.15 m per gap in a
   0.47 m strip, so the leftmost object -- always the first target -- ranges
   over [-0.49, -0.17] with two objects, [-0.49, -0.32] with three and
   [-0.49, -0.47] with four. An independently drawn two-object scene puts the
   first grasp up to 15 cm outside anything training showed. The two-object
   profile now draws a three-object layout and drops one of the later objects
   (`layout = "nested_in_3"`), so its first target is identical per seed to the
   three-object one. The server reports the layout, and the collector drops
   two-object episodes recorded without it. Four objects stay inside the
   training support, only denser at its left edge. Every Stage A count cell
   was re-run on the nested layout; the old files are in
   `eval/superseded_independent_layout/`.
5. **Atomic training data (open, and the largest).** Every `F`/`J`/`D`/`X`
   cell trained on atomic clips. Chaining depends on that choice (§4.6), so the
   Stage A comparisons hold for atomic-clip training; whether conditioning
   helps a policy that already chains is not yet measured.

---

## 4. Current results

### 4.1 Q1: is object conditioning useful? (ACT, absolute EE, `F`)

**Source.** 7 arms × 3 training seeds × 20 scenes per tier, every tier pinned
and verified. The tiers are informative (every arm drops from seen to
held-out on every seed), so they are reported separately.

Mean transfers per episode, mean [min–max] across the 3 seeds:

| arm | seen | held-out | novel |
| --- | --- | --- | --- |
| rgb | 0.62 [0.45–0.90] | 0.03 [0.00–0.10] | 0.28 [0.15–0.55] |
| **rgb_cont** (budget control) | 0.95 [0.50–1.40] | 0.17 [0.05–0.25] | 0.30 [0.00–0.75] |
| rgb_novae | 0.83 [0.45–1.40] | 0.02 [0.00–0.05] | 0.83 [0.30–1.25] |
| kv | 0.83 [0.55–1.35] | 0.22 [0.00–0.35] | 0.25 [0.00–0.75] |
| kv_adaln | 0.93 [0.75–1.10] | **0.55 [0.35–0.80]** | 0.23 [0.10–0.35] |
| kv_tokens | **1.25 [1.05–1.40]** | 0.13 [0.05–0.20] | 0.25 [0.00–0.75] |
| scratch_kv | 0.45 [0.30–0.55] | 0.05 [0.00–0.10] | 0.30 [0.05–0.60] |

Task success per 20 episodes, mean [min–max]:

| arm | seen | held-out | novel |
| --- | --- | --- | --- |
| rgb | 0.7 [0–1] | 0 | 0 |
| rgb_cont | 3.3 [2–6] | 0 | 1.0 [0–2] |
| rgb_novae | 2.3 [1–5] | 0 | 0.7 [0–2] |
| kv | 2.0 [0–5] | 0.3 [0–1] | 0.3 [0–1] |
| kv_adaln | 3.3 [2–5] | **2.0 [1–3]** | 0 |
| kv_tokens | 4.7 [2–8] | 0 | 0.7 [0–2] |
| scratch_kv | 0.3 [0–1] | 0 | 0 |

**Paired against the budget control `rgb_cont`**, 60 matched episodes per tier
(exact McNemar; b/c = episodes only the control / only the arm won):

| tier | arm | success | p | ≥1 transfer | p |
| --- | --- | --- | ---: | --- | ---: |
| seen | kv | 0.17 → 0.10 (9/5) | 0.42 | 0.53 → 0.52 (13/12) | 1.00 |
| seen | kv_adaln | 0.17 → 0.17 (7/7) | 1.00 | 0.53 → 0.48 (12/9) | 0.66 |
| seen | kv_tokens | 0.17 → 0.23 (8/12) | 0.50 | 0.53 → 0.65 (10/17) | 0.25 |
| seen | scratch_kv | 0.17 → 0.02 (10/1) | **0.012** | 0.53 → 0.35 (21/10) | 0.071 |
| held-out | kv | 0.00 → 0.02 (0/1) | 1.00 | 0.17 → 0.18 (5/6) | 1.00 |
| held-out | kv_adaln | 0.00 → 0.10 (0/6) | **0.031** | 0.17 → 0.33 (5/15) | **0.041** |
| held-out | kv_tokens | 0.00 → 0.00 | 1.00 | 0.17 → 0.13 (9/7) | 0.80 |
| held-out | scratch_kv | 0.00 → 0.00 | 1.00 | 0.17 → 0.05 (8/1) | 0.039 |
| novel | any | no contrast below p = 0.25 | | | |

Against `rgb` alone the picture looked better: `kv_tokens` seen success
0.03 → 0.23 (p = 0.002), `kv_adaln` held-out ≥1 transfer 0.03 → 0.33
(p < 0.001). The first of these does not survive the budget control.

**Conclusion (answered, for ACT trained on atomic clips).**
- **On seen objects, conditioning adds nothing measurable over the same
  amount of extra RGB training.** `rgb_cont` recovers most of the gain every
  conditioned arm showed over `rgb`. `kv_tokens` is highest on the point
  estimate (1.25 against 0.95), but it wins on only 17 of the 27 episodes the
  two disagree on (p = 0.25).
- **On held-out object sizes, AdaLN conditioning helps.** `kv_adaln` succeeds
  on 6/60 held-out scenes against 0/60 for `rgb_cont` (p = 0.031), and its
  mean transfers are above the control's on every seed (0.35–0.80 against
  0.05–0.25). This is the one conditioning effect that survives every control,
  and the largest in the project.
- **Conditioning without stage 1 hurts:** `scratch_kv` is below `rgb_cont` on
  both measures (seen success p = 0.012), as ControlVLA's "w/o pretrain"
  ablation predicts.
- **Novel** (mesh 5, which the oracle has never collected) is helped by no
  arm; no rate there differs from the control.

**Scope.** One corpus, one regime, atomic clips. On the paired corpus a policy
trained on full runs chains transfers the atomic policies cannot (§4.6), and
whether conditioning helps *that* policy is the open question for Stage A2.

### 4.2 Q2: which mechanism? (ACT)

| tier | contrast | success | p | ≥1 transfer | p |
| --- | --- | --- | ---: | --- | ---: |
| seen | kv → kv_tokens | 0.10 → 0.23 (3/11) | 0.057 | 0.52 → 0.65 (10/18) | 0.19 |
| seen | kv_adaln → kv_tokens | 0.17 → 0.23 (6/10) | 0.45 | 0.48 → 0.65 (8/18) | 0.076 |
| held-out | kv → kv_adaln | 0.02 → 0.10 (1/6) | 0.13 | 0.18 → 0.33 (6/15) | 0.078 |
| held-out | kv_adaln → kv_tokens | 0.10 → 0.00 (6/0) | **0.031** | 0.33 → 0.13 (17/5) | **0.017** |
| four objects | kv → kv_adaln | 0 → 0 | — | 0.40 → 0.18 (20/7) | **0.019** |
| four objects | kv → kv_tokens | 0 → 0 | — | 0.40 → 0.18 (19/6) | **0.015** |

**Conclusion (answered, for ACT on atomic clips): AdaLN.**
- It is the only mechanism that beats the budget control on any tier
  (held-out, §4.1), and it beats in-context tokens there on both measures.
- In-context tokens are best on seen objects by point estimate, but not
  significantly above either `kv` or the budget control.
- Both encoder-side mechanisms (AdaLN and tokens) are *worse* than `kv` at
  four objects.

One reading, not tested: tokens give the encoder a per-object handle, enough
to fit the four trained sizes; AdaLN gives a pooled scene summary that
modulates every block, which extrapolates to sizes it never saw but mixes up
a scene with more objects than it was fitted on.

**VLAs:** no valid comparison yet; §4.7 describes the rework of the VLA
token arm that has to come first.

### 4.3 Q3: compositional generalisation (2 / 3 / 4 objects)

Trained on three objects; seen identities; 200 steps per object; the
two-object scenes on the nested layout (§3, item 4). Mean transfers per
episode, mean [min–max] across seeds:

| arm | 2 objects | 3 objects | 4 objects |
| --- | --- | --- | --- |
| rgb | 0.22 [0.20–0.25] | 0.62 [0.45–0.90] | 0.40 [0.30–0.50] |
| rgb_cont | 0.38 [0.15–0.55] | 0.95 [0.50–1.40] | 0.32 [0.15–0.45] |
| rgb_novae | 0.33 [0.15–0.50] | 0.83 [0.45–1.40] | 0.58 [0.35–1.00] |
| kv | 0.33 [0.25–0.40] | 0.83 [0.55–1.35] | 0.42 [0.35–0.45] |
| kv_adaln | 0.43 [0.20–0.80] | 0.93 [0.75–1.10] | 0.20 [0.15–0.25] |
| kv_tokens | 0.37 [0.25–0.45] | 1.25 [1.05–1.40] | 0.28 [0.25–0.35] |
| scratch_kv | 0.22 [0.15–0.30] | 0.45 [0.30–0.55] | 0.32 [0.05–0.85] |
| *AF-rgb (atomic, paired corpus)* | 0.47 [0.35–0.55] | 0.48 [0.45–0.55] | 0.42 [0.30–0.60] |
| *CF-rgb (full runs, paired corpus)* | **0.88 [0.70–1.05]** | 0.70 [0.40–0.90] | **0.58 [0.45–0.75]** |

Task success: 0/20 at four objects for every cell and seed. At two objects,
every `F` cell is at most 1.7/20; `CF-rgb` is 6.3/20 [5–8].

**Conclusion (answered, for ACT): conditioning does not help object-count
generalisation.**
- No conditioned arm beats `rgb_cont` at two or four objects (smallest p =
  0.17, `kv` at four objects), and the encoder-side arms are worse than `kv`
  at four (§4.2).
- **The collapse at two objects is mostly the training format.** An atomic
  policy scores lower on two objects than on three for every `F` arm — an
  easier scene, done worse. Trained on continuous runs, the same RGB ACT
  scores *higher* on two than on three (0.88 against 0.70) and succeeds on
  19/60 two-object scenes, against 3/60 for its atomic twin (p < 0.001, §4.6).
  The old two-object layout (independent draws) cost a little on top: `rgb`
  0.20 → 0.22 and `kv_tokens` 0.18 → 0.37 once the first target was placed as
  in training.
- **Four objects fails for everyone,** full runs included (0 of 60 four-object
  episodes succeed in `CF`). Four is inside the training support positionally (§3, item 4), so
  this is a genuine sequence-length limit: no policy has chained past three
  transfers.

### 4.4 Q4: control regime (ACT, RGB, identity corpus)

`rgb` × 3 seeds × 20 scenes per tier in each regime:

| regime | cell | seen mean transfers | seen success | held-out | novel | ≥1 lift (all tiers) |
| --- | --- | --- | --- | --- | --- | --- |
| absolute EE | `F-rgb` | 0.62 [0.45–0.90] | 0.7/20 | 0.03 | 0.28 | 139/180 |
| absolute joint | `J-rgb` | 0.35 [0.15–0.55] | 0.3/20 | 0.02 | 0.57 | 110/180 |
| joint delta | `D-rgb` | 0.00 | 0 | 0.00 | 0.00 | 24/180 |
| EE delta | `X-rgb` | 0.00 | 0 | 0.00 | 0.00 | 32/180 |

Absolute EE against absolute joint, paired: seen ≥1 transfer 19 → 27 of 60
(p = 0.20), novel 23 → 15 (p = 0.13, the other way); nothing below p = 0.1.

**Conclusion (answered, for ACT): use an absolute regime; EE and joint are
tied.**
- **The delta regimes do not work at all:** 0 transfers in 180 episodes each,
  with the arm lifting anything in only 13–18 % of episodes. An integrated
  command accumulates the policy's error, and nothing in the observation
  corrects it.
- **Absolute EE and absolute joint cannot be separated** with 3 seeds: EE is
  ahead on seen objects, joint on novel, neither significantly. Absolute EE
  stays the default because every other cell was built on it.
- The planned "best conditioned arm on each regime" cells were not run: with
  the delta regimes at zero and EE ≈ joint, they could not change the answer.

**VLAs:** no VLA has been trained on the final corpora yet. The earlier
single-seed VLA cells (old corpus, 6–8 k steps) scored 0/20 success
everywhere and cannot rank regimes.

---

### 4.5 Why the baseline is weak: stage-wise diagnostics (ACT, Stage A)

**Stage-wise survival** (`collect_final_results.py`, STAGES section: raw
counts, a 95 % Wilson interval, and every training seed separately). Pooled
over 3 seeds × 20 scenes, three objects, seen identities:

| arm | lift | T1 \| lift | T2 \| T1 | T3 \| T2 | task success |
| --- | ---: | ---: | ---: | ---: | ---: |
| rgb | 47/60 | 27/47 (0.57) | 8/27 (0.30) | 2/8 (0.25) | 0.03 |
| rgb_cont | 52/60 | 32/52 (0.62) | 15/32 (0.47) | 10/15 (0.67) | 0.17 |
| rgb_novae | 54/60 | 32/54 (0.59) | 11/32 (0.34) | 7/11 (0.64) | 0.12 |
| kv | 55/60 | 31/55 (0.56) | 13/31 (0.42) | 6/13 (0.46) | 0.10 |
| kv_adaln | 54/60 | 29/54 (0.54) | 17/29 (0.59) | 10/17 (0.59) | 0.17 |
| kv_tokens | 56/60 | 39/56 (0.70) | 22/39 (0.56) | 14/22 (0.64) | 0.23 |

- **The first lift → transfer is the weakest stage in every arm** (0.54–0.70).
- For `kv_tokens` the stages multiply to its success (0.93 × 0.70 × 0.56 ×
  0.64 ≈ 0.23): compounding local error. `rgb` decays stage by stage.
- **Later stages rest on small denominators.** `kv_tokens`' T3 | T2 is 14/22,
  95 % CI [0.43, 0.80], and 0.25, 0.67 and 1.00 on its three seeds. The shape
  is a lead, not a result.

**The CVAE latent is unused.** ACT's KL term falls to 0.005 by 10 k steps and
0.000 by 40 k on all three seeds. Removing it (`rgb_novae`) moves no seen or
held-out contrast against `rgb` below p = 0.18; only novel ≥1 transfer rises
(0.25 → 0.52, p = 0.002), which no other arm shows and is treated as a lead.
So the latent neither helps nor measurably hurts; it stays on by default.

**Where lifted objects fail** (ground truth, never shown to the policy;
`tasks/shelf_restock/events.py`). Every lifted object not on the upper shelf
at the end, three objects, all tiers where the rollout carries events:

| cell | episodes | dropped before the shelf | placed, then knocked off | other |
| --- | ---: | ---: | ---: | ---: |
| F-rgb (seen) | 60 | 26 | 26 | 0 |
| F-rgb_cont | 180 | 97 | 68 | 0 |
| F-rgb_novae | 180 | 97 | 57 | 0 |
| F-kv_tokens (seen) | 60 | 16 | 31 | 1 |
| AF-rgb | 180 | 87 | 72 | 1 |
| CF-rgb | 180 | 80 | 65 | 3 |

- **About 40–65 % of lost objects were placed correctly and later knocked
  off**, mostly onto the floor, a median 80–270 steps after placement — both
  while the next object is handled and when no later object is ever lifted.
  This is contact with objects already on the shelf, not a grasp or release
  failure.
- **Drops happen with the object almost never in hand** (median held
  fraction 0): the object is knocked or flung rather than slipping mid-carry.
  Caveat: "in hand" is calibrated on the oracle's grasp (within 0.17 m of the
  end-effector); a policy holding off-centre could read as not in hand.
- **A gripper classification head would address neither mode**, so it moves
  behind collision-aware behaviour in the queue.

### 4.6 Atomic clips vs continuous runs: the chaining study

**The question.** Stage A trained every arm on *atomic clips*: each oracle run
cut into one clip per transfer, each clip keeping only its own frames. A
policy trained that way never sees the moment one transfer ends and the next
begins: with ACT's 50-step chunk, 17.4 % of training targets are padding
(`action_is_pad`), all of it at clip ends, and at rollout 15 % of the 25-step
executed blocks run past where any clip ended. Does training on the
uncut run instead let the policy chain transfers?

**Design: one oracle execution, two exports.**
- Collecting the same seeds twice does not reproduce a trajectory (cuRobo
  plans stochastically), so a separate collection per format would differ in
  more than the format. Each of 200 seeds was run **once**
  (`EPISODE_KIND=paired`, `$HPCVAULT/octvla-collection-v3paired`) and written
  both as its three atomic clips and as the full run; 112 seeds succeeded.
- Both exports use the same builder settings (unified, binary gripper,
  entity tokens, declared holdout of meshes 0, 5 and 6), the same 75 training
  and 23 validation runs (checked equal), and the absolute-EE view.
- Same data budget: 31,636 training frames as 225 clips (median 176 frames,
  63–189) against 31,787 as 75 runs (median 422, 413–442). The difference is
  the terminal frame of each clip, which atomic cutting drops.
- Padded targets at chunk 50: 17.4 % atomic, 5.8 % full runs.
- The model and recipe are identical: `rgb` ACT, 40 k steps, chunk 50,
  executing 25, seeds 1000–1002, and the same pinned rollouts. `AF-rgb`
  trains on the clips, `CF-rgb` on the runs.

**Result.** Paired over the same (training seed, scene) episodes:

| scope | success, atomic → full | b/c | p | ≥1 transfer | p | mean transfers |
| --- | --- | --- | ---: | --- | ---: | --- |
| 3 objects, seen | 1/60 → 9/60 | 0/8 | **0.008** | 26 → 24 | 0.83 | 0.48 → 0.70 |
| 2 objects | 3/60 → 19/60 | 1/17 | **< 0.001** | 25 → 34 | 0.15 | 0.47 → 0.88 |
| 4 objects | 0 → 0 | — | — | 24 → 31 | 0.23 | 0.42 → 0.58 |
| 3 objects, held-out | 0 → 0 | — | — | 5 → 3 | 0.69 | 0.08 → 0.05 |
| 3 objects, novel | 6/60 → 6/60 | 5/5 | 1.00 | **45 → 24** | **< 0.001** | 1.18 → 0.75 |

Per training seed (success per 20, seeds 1000 / 1001 / 1002):

| scope | atomic | full runs |
| --- | --- | --- |
| 3 objects, seen | 0 / 1 / 0 | **3 / 2 / 4** |
| 2 objects | 0 / 3 / 0 | **5 / 8 / 6** |
| 3 objects, novel (mean transfers) | 1.25 / 1.00 / 1.30 | 0.10 / 0.30 / 1.85 |

Stage by stage (three objects, seen):

| | lift | T1 \| lift | T2 \| T1 | T3 \| T2 |
| --- | ---: | ---: | ---: | ---: |
| atomic (`AF`) | 54/60 | 26/54 (0.48) | **2/26 (0.08)** | 1/2 |
| full runs (`CF`) | 48/60 | 24/48 (0.50) | **9/24 (0.38)** | **9/9** |

With two objects the same holds: T2 | T1 is 3/25 atomic and 19/34 full runs.

**Reading.**
- **Full runs teach chaining; atomic clips do not.** The first transfer is
  unchanged (0.48 against 0.50 after a lift), but an atomic policy that has
  finished one transfer almost never finishes another (2 of 26), while the
  full-run policy does 9 of 24 times and then completes the third every time.
  Success rises on **every training seed**, on three objects and on two.
- **So the atomic format, not a count or a sequencing limit of ACT, caused the
  collapse after the first transfer** that §4.5 measured for `rgb`. Two
  objects is now easier than three, as it should be.
- **It does not fix single transfers.** T1 | lift stays near 0.5, and the
  failure modes are unchanged (§4.5): about half the lost objects are
  dropped, and about 45 % placed then knocked off.
- **Four objects still fails** (0/60): three chained transfers is the longest
  sequence in training.
- **The novel tier goes the other way,** by 45 → 24 episodes with a transfer.
  It is not seed-consistent (full runs are at 1.85 on one seed and 0.10–0.30
  on the others), mesh 5 is one the oracle has never been able to plan, and
  no other tier moves. Treated as unexplained, not as a cost of full runs.

**Against Stage A** (same scenes and pins, different corpus — a lead only):
`CF-rgb` at 40 k steps matches the 80 k-step `F-rgb_cont` on seen success
(9 against 10 of 60, p = 1.0), and beats it at two objects (19 against 3,
p < 0.001) and on ≥1 transfer at four (31 against 16, p = 0.006). It also
beats `F-kv_tokens` on the same two scopes (p < 0.001, p < 0.001), though
`kv_tokens` keeps more ≥1-transfer episodes on three objects (39 against 24,
p = 0.011).

**Decision.** Full runs are the training format from here (`data_protocol.md`).
The remaining ACT interventions are measured on the `CF` corpus.

**Limits.** One chunk (50) and execution horizon (25), one budget (40 k), one
corpus of 75 training runs. The atomic export also drops each clip's terminal
frame, which the full run keeps — part of the difference, by design.

### 4.7 Short chunk and observation history (full-run corpus)

**Short chunk (`CF-rgb_short`: chunk 20, executing 8; done).** Against
`CF-rgb` (chunk 50, executing 25), paired over the same scenes:

| scope | success | p | ≥1 transfer | p | mean transfers |
| --- | --- | ---: | --- | ---: | --- |
| 3 objects, seen | 9 → 7 of 60 | 0.79 | 24 → 35 | 0.061 | 0.70 → 0.93 |
| 3 objects, held-out | 0 → 0 | — | 3 → 13 | **0.021** | 0.05 → 0.25 |
| 3 objects, novel | 6 → 0 | **0.031** | 24 → 1 | **< 0.001** | 0.75 → 0.02 |
| 2 objects | 19 → 7 | **0.004** | 34 → 33 | 1.00 | 0.88 → 0.67 |
| 4 objects | 0 → 0 | — | 31 → 38 | 0.21 | 0.58 → 0.82 |

- **Single transfers improve.** The first transfer after a lift rises from
  24/48 to 35/48 on seen objects, and from 31/54 to 38/52 at four objects;
  held-out objects get transferred at all (13 against 3). Objects placed and
  then knocked off fall from 65 to 23: re-planning every 8 steps instead of
  25 disturbs the shelf less.
- **Chaining gets worse.** The second transfer given the first falls to
  7/33 at two objects (from 19/34), so two-object success drops 19 → 7.
- **The novel mesh collapses on every seed:** lifts in 6 of 60 episodes
  against 42 for `CF-rgb` (3, 1 and 2 of 20 per seed). Execution was checked
  (horizon 8, chunk 20, meshes pinned to 5). Mesh 5 is the one variant the
  oracle has never collected; a short-horizon policy apparently cannot start a
  grasp on an out-of-distribution shape at all. Unexplained.
- **Net:** a trade, not an improvement: better local manipulation, worse
  sequencing and robustness. Mean transfers rise on seen, held-out and four
  objects; success does not.

**Observation history (`CF-rgb_hist`: the same short chunk plus a two-frame
history; rollouts running).** The three models trained; their first rollouts
died at startup on an unregistered policy type (fixed, `ff2ecce`) and are
re-queued. `rgb_hist` against `rgb_short` isolates the history.

**VLA pipeline.** The end-to-end smoke tests (stage 1 → merge → init check →
stage 2 → rollout, all four arms) found three bugs, now fixed:
- **The init check reported every pass as a failure.** A `du` on the run
  directory that an init check never writes, under `pipefail`, failed the job.
- **The init check fed raw uint8 images:** SmolVLA crashed, pi0.5 and GR00T
  saw inputs 255× out of range. It now prepares batches exactly as training
  does, and asserts float images in [0, 1].
- **Conditioned SmolVLA ran its VLM in float32 against a bfloat16 baseline.**
  A conditioned arm is built from `--policy.type`, which starts from class
  defaults, and lost the hub config's `load_vlm_weights=true`.
  `scripts/inherited_policy_args.py` now passes every non-default stage-1
  field through (pi0.5 has none). Reproduced on CPU: a 0.46 % step-0 drift
  before, 0.0 after.

And one design change, made the ACT way: **VLA `kv_tokens` now gates the
entities and keeps the actions' positions** (`method.md` §2.3). Prepending
them used to shift every action's RoPE position and let them into the softmax
at full weight, moving pi0.5's step-0 action chunk by 137 %; behind the gate it moves
0.36 % (SmolVLA 3.1 %), and matches stage 1 when the gate is closed. GR00T
passes all checks. **All three VLAs now pass the full smoke pipeline**
(stage 1 → merge → init check for every arm → stage 2 → rollout).

**Stage A2, gated on RGB chaining:** the conditioned arms on the full-run
corpus, since Q1–Q3 hold only for atomic training. Then the VLAs (Stage B).

## 5. What answers each question

Matrix prefixes: `F` absolute EE, `J` absolute joint, `D` joint delta, `X` EE
delta, all atomic clips; `AF`/`CF` the paired corpus as atomic clips and as
full runs. VLA matrices put the backbone first: `P` pi0.5, `S` SmolVLA, `G`
GR00T.

| # | runs | status |
| --- | --- | --- |
| Q1, Q2, Q3 (ACT) | `F`: rgb, rgb_cont, rgb_novae, kv, kv_adaln, kv_tokens, scratch_kv; count re-run on the nested layout | **done** (§4.1–4.3) |
| Q4 (ACT) | `J`, `D`, `X`: rgb × 3 seeds | **done** (§4.4) |
| chaining | `AF-rgb` vs `CF-rgb` | **done** (§4.6) |
| chunk / history | `CF-rgb_short` (done, §4.7), `CF-rgb_hist` × 3 seeds | rollouts running |
| Stage A2 | conditioned arms on `CF` | after the history study |
| Q1–Q4 (VLAs) | `slurm/submit_vla_experiments.sh` | **ready**: all smoke tests pass |

**Storage rules**, since the vault has 1 TB for everything:
- Every run keeps only its final checkpoint and drops its optimiser state on
  success.
- GR00T saves only its trained head (`slim_groot`); a full save was about
  49 GB.
- Merged pi0.5/SmolVLA checkpoints are made only for the regime that goes to
  stage 2.
- Rollout videos stay on the vault: home allocates 32 MB per file.
- The paired collection is 32 GB; the four datasets built from it, 0.8 GB.

## Update log

- **2026-09-24 (afternoon).** Short chunk on full runs (§4.7): better single
  transfers and far fewer knock-offs, worse chaining (two-object success
  19 → 7 of 60) and a novel-mesh collapse on every seed. All three VLA smoke
  tests pass. History rollouts re-queued after a missing rename-map entry.
- **2026-09-24.** Stage A complete; Q1–Q4 answered for ACT on atomic clips
  (header table). Against the budget control, conditioning helps only
  held-out object sizes, through AdaLN; count generalisation is not helped;
  the delta regimes fail. The chaining study (§4.6): training on full runs
  instead of atomic clips lifts success on every seed, three objects 1 → 9 of
  60 and two objects 3 → 19 of 60, by fixing the transfers after the first.
  Full runs become the default. Failure modes: most lost objects are knocked
  off after placement. History ACT submitted. VLA fixes: init check, config
  inheritance (SmolVLA precision), and gated, position-preserving tokens.
- **2026-09-23 (late).** Review of the stage-wise result. The two-object
  profile was spatially confounded (§3, item 4), so it is now nested in the
  three-object layout and re-run. Atomic vs full-run is now paired from one
  oracle execution. The identity holdout is declared by model id. Rollouts
  carry per-object ground-truth failure modes, and the stage table prints
  counts, Wilson intervals and per-seed rows.
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
