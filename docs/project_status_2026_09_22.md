# OCT-VLA status: where we are, what we retracted, what runs next

For the project team, 2026-09-22. Supersedes `report_2026_09.md`, whose headline
result did not survive replication.

---

## 1. Read this first

**We do not have a working configuration.** The previous report opened with "one
configuration completes the task" — ACT with absolute end-effector actions,
6/20 episodes. Replicating that run on two further training seeds gave **0/20
and 0/20**. Everything else about the two replicas was identical: same data,
same encoding, same steps, same evaluation seeds. Only the training seed moved.

| training seed | offline L1 | success | ≥1 transfer | mean transfers |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 0.0785 | **6/20** | 7/20 | **1.00** |
| 1001 | 0.0767 | 0/20 | 1/20 | 0.05 |
| 1002 | 0.0749 | 0/20 | 1/20 | 0.05 |

So the honest statement is: *ACT with absolute EE actions succeeds on one seed in
three and scores near zero on the others.* Seed variance is larger than every
effect this project has reported.

**This invalidates the two contrasts that reached significance**, because both
compared seed-1000 cells against other seed-1000 cells: absolute-EE versus
absolute-joint (*p* = 0.031) and ACT versus GR00T (*p* = 0.031). Neither is
dead — but neither is established, and both need re-running across seeds.

The rest of this document separates what still stands from what does not.

---

## 2. What still stands

### 2.1 No pretrained VLA beat an 80 M ACT trained from scratch

Absolute joint actions, identical corpus, one seed each.

| backbone | params | offline L1 | success | ≥1 transfer | ≥1 lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACT (scratch) | 0.08 B | 0.0665 | 0/20 | **3/20** | **15/20** |
| SmolVLA | 0.45 B | 0.1713 | 0/20 | 0/20 | 5/20 |
| GR00T N1.7 | 3.16 B | **0.0578** | 0/20 | 1/20 | 12/20 |
| pi0.5 | 3.62 B | 0.1029 | 0/20 | 0/20 | 5/20 |

Four backbones spanning 45× in parameters, **zero successes in 80 episodes** on
this encoding, and zero in all **240** VLA episodes recorded across every
encoding and mechanism. Every one ends at the step limit: no crashes, no IK
failures — the arms move and never finish. GR00T has the best offline action
prediction in the project and converts it into one transfer in twenty episodes.

This survives the seed problem because it is a null result across four
independent models, not a comparison between two cells. The caveat that matters:
the VLA cells ran 8 k steps against ACT's 40 k, and are single-seed and untuned.
Fair reading: *these VLAs, LoRA-finetuned at this budget, did not beat ACT here.*

### 2.2 Offline loss does not predict closed-loop success

Nine cases now, and the replication made it worse rather than better: seed 1002
has the **best** offline L1 of the three A5 replicas (0.0749 against 0.0785) and
the worst rollout. The three are within 5 % offline and 20× apart online.

| pair | offline | closed-loop |
| --- | --- | --- |
| A5 seeds 1000/1001/1002 | 0.0785 / 0.0767 / 0.0749 | 1.00 / 0.05 / 0.05 transfers |
| ACT vs GR00T (abs joint) | 0.0665 vs 0.0578 (GR00T better) | 3/20 vs 1/20 (ACT better) |
| F0 vs F1 (GR00T ±conditioning) | 0.0706 vs 0.0955 | both 0/20 |

**Do not rank cells or select checkpoints on validation loss in this project.**
Use it as a floor check only — `scripts/diagnose_action_head.py` against the
constant predictor.

### 2.3 Object conditioning shows no measurable effect, and the mechanism test was void

On ACT with absolute EE, one seed each:

| arm | observes | success | ≥1 transfer | mean transfers |
| --- | --- | ---: | ---: | ---: |
| A5 | RGB + proprioception | 6/20 | 7/20 | 1.00 |
| C2 | RGB + proprioception + objects | 2/20 | 10/20 | 0.85 |
| C3 | proprioception + objects, no cameras | 0/20 | 1/20 | 0.05 |

The two metrics move in opposite directions and neither survives testing
(success *p* = 0.219, transfer *p* = 0.375 paired). Given §1, both arms are
single-seed and the comparison cannot be read at all.

**C3 is the one part still worth quoting**: object state *without* cameras
collapses. Whatever the EE-space policy uses comes from the images.

The GR00T mechanism comparison (F0/F1/F2, ControlVLA vs pooled vs none) produced
**zero successes in 180 episodes** because its own baseline was at zero. It
measures nothing about the mechanism. That was a gate failure we should have
caught before spending the compute.

### 2.4 Neither arm generalises off the trained object count

Trained on `three_object`, evaluated on all three. Success collapses at 2 and 4
objects for every arm. One suggestive difference: at four objects the
conditioned arm reaches ≥1 transfer in 9/20 against RGB's 4/20 — the only place
an object arm has looked better on a partial-credit measure. Single-seed;
directional only.

---

## 3. What we fixed in the code, and what it invalidated

Ten defects, several of which silently corrupted results that had already been
reported. Listing them because the pattern matters more than any one:

| defect | consequence |
| --- | --- |
| **uint8 images at evaluation** | Any policy declaring `VISUAL=IDENTITY` got inputs **255× out of range**. pi0.5 casts to float without dividing then computes `img*2-1` → `[-1, 509]`. Silent. **Every pi0.5 rollout before this was invalid.** Re-run afterwards: still zero, so the bug does not explain pi0.5's failure. |
| **ControlVLA was not the published method** | We were adding a *pooled* scene vector as one broadcast bias. The paper adds a per-layer attention term over the *unpooled* object set with zero-init K/V. Now implemented properly; the old version is retained as an ablation. |
| **SmolVLA conditioning dead at rollout** | Its `select_action` bypasses `predict_action_chunk`, so the object path was inert at evaluation while training normally. Would have read as "conditioning did not help". |
| **Entity normalisation leak** | GR00T and SmolVLA called `validate_features` in neither place, so entity tokens would have been MEAN_STD-normalised with statistics computed over **train + validation**. |
| **ACT dropout asymmetry** | ACT drops 10 % of its native attention; the object branch dropped none, leaving the object path less regularised — a thumb on the scale in the exact comparison it exists to run. |
| **layerwise pi0.5 module cycle** | `.to(device)` and `save_pretrained` raised `RecursionError`. |
| **GR00T loss dilution** | Pads actions to 132 columns against our 16, so its training loss averages 116 constant zeros — not comparable to any other backbone. |
| **gripper validator, hardcoded encodings, PEFT adapter list** | Each ended or corrupted specific arms. |

Gradient flow through the object encoder is now **verified on GPU**, not assumed:
at step 0 only V_z receives gradient (4 of 18 tensors, exactly as zero-init
requires); after six steps all 10 entity-encoder tensors receive gradient and
have moved; the gradient matches float64 central differences; padded slots
receive exactly zero. 651 tests pass.

---

## 4. The dataset, and a correction

**The corpus already contains six visually distinct objects.** We spent a day
attempting a multi-category re-collection before noticing this — the rollout
videos showed it. `113_coffee-box` ships seven meshes and the corpus contains
six of them, 18 to 81 episodes each, one per episode, never mixed within a scene.

The multi-category attempt is abandoned and was not cheap. What it established:
of six assets passing a geometric screen (gripper width, shelf separation, deck
clearance), **five collect 0 of 8 seeds on their own** — the oracle's planner
cannot solve a top-down grasp around those meshes. **Geometry does not predict
plannability**; a candidate asset must be probed before it is trusted.

### The protocol now

An asset variant's size is a property of its mesh, so identity inverts exactly
from what was recorded — no re-collection, no simulator.

| model_id | size (m) | clips | role |
| ---: | --- | ---: | --- |
| 4 | 0.0581 × 0.0777 × 0.0692 | 81 | train |
| 2 | 0.0708 × 0.0781 × 0.0666 | 72 | train |
| 1 | 0.0729 × 0.0778 × 0.0697 | 69 | train |
| 3 | 0.0565 × 0.0592 × 0.0771 | 63 | train |
| 6 | 0.0540 × 0.0479 × 0.0774 | 24 | **held out** |
| 0 | 0.0662 × 0.0744 × 0.0965 | 18 | **held out** |
| 5 | 0.0216 × 0.0774 × 0.0571 | **0** | never collected |

222 train clips / 74 runs, 63 IID validation clips, 42 clips excluded from the
dataset entirely. The identity axis sits *beside* the seed split rather than
replacing it — without the IID signal, a drop on unseen identities could not be
attributed to identity rather than to the policy simply not working.

`model_id 5` appears in zero of 109 collected runs (≈5e-8 under uniform
sampling), so the oracle systematically fails it. Rollouts do not use the
oracle, so it is a genuinely novel object at evaluation.

### The entity token

32 columns, deliberately balanced 16 geometric against 16 semantic, so neither
block dominates the encoder by width before a weight is trained:

```
[ 0: 3] position          [16:20] entity type one-hot
[ 3: 9] rotation (6D)     [20:32] variant code
[ 9:12] size
[12]    gripper aperture
[13]    log volume
[14:16] two log aspect ratios
```

Identity is a **vector in the dataset**, not an integer resolved against a table
inside the policy: the table would make the dataset non-self-describing, and an
id never trained would have no row, so an unseen mesh would degrade to a reserved
zero instead of something the network can process. The code is hashed from the
rounded size, so it is identical across exports and an unseen mesh still gets one.

**Its limit, stated plainly:** in this corpus geometry *is* identity, so the code
carries no information the geometry lacks. What it adds is accessibility —
identity as a direction rather than something the network must recover by
discretising three millimetre-scale columns. A corpus with two distinct meshes at
equal size would need a code derived from appearance.

---

## 5. What runs next, and what each outcome would mean

Nine training cells and 27 rollouts, queued as one command
(`slurm/submit_final_experiments.sh`), collected as one table
(`scripts/collect_final_results.py`).

**Three arms, identical data, differing only in what the policy reads:**
`rgb` (stock ACT) · `entity` (layerwise conditioning, geometry only) ·
`semantic` (the same, plus the variant code).

**Three training seeds each.** Not negotiable after §1.

**Three identity tiers per rollout:** `seen` (ids 1,2,3,4) · `heldout` (0,6) ·
`novel` (5). Mixing them into one number is what would make the holdout
meaningless.

### What to expect

Set expectations low and read the ranges, not the means.

- **Most likely:** all three arms sit near zero on most seeds, with occasional
  lucky seeds. That is what the data so far predicts. It would be a real result
  — "object-centric conditioning does not rescue a policy that cannot reliably
  do the task" — but it is not the result anyone is hoping for.
- **A difference between arms is only believable if it holds across all three
  seeds.** One arm winning on one seed is precisely the pattern we just
  retracted.
- **The seen-to-heldout drop is the most interesting number**, and it is
  interpretable even if absolute performance is low: it asks whether the policy
  learned the task or the objects. A large drop for `semantic` and a small one
  for `entity` would say the identity channel is being used as a shortcut.
- **`novel` will likely be worst for every arm.** It is one mesh the oracle
  itself cannot handle; treat it as a stress test, not a headline.

The collector prints min–max across seeds beside every mean, refuses to report
single-seed cells, and states that a cell whose range spans zero has established
nothing.

### Cost

~9 GPU-h training, ~8 GPU-h rollouts. Roughly 3 h wall-clock on the cluster in
parallel, or about 19 h sequentially on a 4090 (ACT peaks at 1.69 GB, so it fits
comfortably; the VLA arms do not).

**Currently blocked:** the cluster is under a maintenance reservation covering
every partition, and nothing new will schedule. The dataset build and the full
matrix are queued and will run when it clears.

---

## 6. What we would do with more time

In priority order, and honestly:

1. **Find out why nothing works reliably.** Nine offline-versus-online
   contradictions and a 1-in-3 seed success rate point at the task or the
   execution path, not at the representation. Rollout videos for the seeds that
   fail would cost hours and we have never done it systematically.
2. **Full-run episodes.** Every arm grasps and fails to place (17/20 lift, 7/20
   transfer at best). The policy has never seen a target switch — clips are
   single transfers. The capability to collect continuous runs exists and is
   untested.
3. **More seeds before more arms.** Given §1, adding a fourth representation is
   worth less than a fourth seed on the three we have.
4. **Do not spend more on VLA arms** at this budget without changing the budget.
   Four backbones, two encodings, three mechanisms, 240 episodes, zero successes.

---

## 7. Provenance

Every figure is re-derived from `$OCTVLA_OUTPUT_ROOT/{eval,diagnostics}/*.json`
rather than quoted from a previous document. Significance is paired exact
McNemar over shared evaluation seeds. Code state: `9c63008`.

Superseded: `report_2026_09.md` §1 and §3.1 (seed replication), and every
closed-loop number recorded before `41fc008` for pi0.5 and GR00T (uint8 images).
