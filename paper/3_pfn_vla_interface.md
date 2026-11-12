# PFN-VLA: what the prior hands the policy, and what each stage teaches

Working notes for the paper (2026-11-11). They record why the prior conditions GR00T
through keyframes, why an oracle keyframe is tested first, and the additional
latent-cell arm. After each stage they record what it taught us and what carries
back to GEMIL. Numbers are re-derived from the result JSONs named beside them.
A single-seed number is marked as such and is not a result.

## 1. The story

**Main claim.** A small prior-fitted network (PFN), pretrained only on synthetic
interaction primitives and kept frozen, helps a VLA in the **low-data regime**:
- it reads K support demos plus the instruction;
- it predicts the next interaction;
- that prediction conditions GR00T's flow head through object-centric attention.

**Interfaces.** The main interface is the **next-interaction keyframe**. A second
interface, the PFN's **latent query cells**, is tested as an additional arm. If the
latent arm fails, the paper rests on keyframes alone; nothing in the main story
depends on it.

## 2. Terms

| Term | Meaning |
|---|---|
| **Support demos** ("the support") | The K example demonstrations shown to the PFN at inference time, as in few-shot prompting. They show the rule, e.g. "move the objects rightmost-first". The PFN reads them in context, with no gradient step. |
| **Query** | The current scene, for which the PFN predicts what happens next. |
| **Keyframe** | Three rows in the 17-column entity schema plus a role and an event one-hot (`data/future_keyframes.py`). They give the object at the next event, the acting gripper's pose and aperture at that event, and the object's goal. The event is grasp or place. |
| **Oracle keyframe** | The true keyframe, computed from simulator state with the expert's own planners (`tasks/shelf_restock/keyframes.py`). It is privileged and can never be deployed: a privileged-information reference, not a strict upper bound (it is trained under a different input distribution, so a learned source could beat it). |
| **Latent query cells** | The PFN's internal embedding of each current-scene entity after it has attended to every support demo and to the instruction (`KeyframePFN.query_cells`). This is what the PFN *knows* before its heads decode it into a keyframe. |

## 3. Why test the oracle keyframe first

PFN-VLA is a chain of two links:

1. **The PFN predicts the keyframe** from support plus language.
2. **GR00T uses the keyframe** to act better.

If the full system does not beat kv, a test of the full system cannot say which link
broke: the predictions may be wrong, or GR00T may ignore even correct ones. The oracle
test checks link 2 on its own (gate G2).

- **If the oracle doesn't help,** the interface is the bottleneck, and no PFN can help
  through it. Fix the interface, or stop, before building the learned path.
- **If the oracle helps,** its score is the most a perfect PFN could reach through this
  interface. The learned PFN (G3) is then read as the share of the kv→oracle gap it
  recovers.

**What it supports:** "future-keyframe conditioning through a second KV branch can
improve a VLA beyond current-state object tokens", once it holds on three seeds. In the
paper it is the privileged-reference row.

**What it does not support:**
- that the PFN works, since the PFN is not in the loop;
- in-context rule inference. On the leftmost corpus every training demo has the same
  order, so plain kv already knows which object comes next. There the keyframe mostly
  adds the grasp pose, the goal slot and the timing: motion guidance, not task
  inference;
- that synthetic pretraining helps;
- a deployable method.

## 4. Why the PFN hands GR00T keyframes

This was the plan's design decision, for three reasons:

1. **Every stage can be checked.** A keyframe has a ground truth, so it can be scored
   offline (G1). It has an oracle reference (G2) and a measurable oracle→PFN gap
   (G3). A failure can always be placed in one link.
2. **GEMIL's lesson.** There the end-to-end consumer ignored its context. A narrow,
   supervised bottleneck forces the support's information through. A rich latent lets
   the downstream model learn to do without it.
3. **It speaks the policy's language.** The rows use the same entity schema as the kv
   tokens, so GR00T's second branch reads them like objects, and the branch is
   bit-identical to kv when untrained or when the mask is empty.

**The cost is that a keyframe is lossy:**
- it carries only the *next* event;
- it drops the PFN's uncertainty and its representation of the rule;
- a wrong pointer argmax hands GR00T a confidently wrong target.

## 5. The additional arm: latent query cells

**What it is.** The frozen PFN's query cells go into the same second KV branch through
a learned projection. Only the projection and the branch train, so GR00T's flow head
learns how to turn what the PFN knows into actions.
- It shares all of Step 3's infrastructure: live PFN inputs built from the shelf scene,
  K support demos, and the frozen PFN at train and evaluation time.
- It has no oracle: latent cells have no ground truth.

> **Where the rules are learned.** The head on top must not be where rule knowledge
> comes from. A trainable head can only learn the rules present in the VLA data, and
> the shelf corpus contains one order. So the PFN stays frozen, and the projection and
> branch only adapt how its cells drive the arm. Anything that composes or generalises
> must come from the prior.
>
> **The test is the same for both interfaces:**
> - a held-out rule (Step 4) that GR00T never saw;
> - correct, shuffled and conflicting support at evaluation, with no retraining.
>
> If shuffled support does not hurt the latent arm, it is ignoring its context, which
> is GEMIL's failure again.

> **The comparison.** Step 3 runs three arms on the same seeds against kv:
> - kv + PFN keyframe (interaction prior), the main arm;
> - kv + PFN latent cells (interaction prior), the additional arm;
> - kv + goal-only PFN, which tests whether interaction pretraining matters.
>
> How to read the two interfaces:
>
> | Outcome | Reading |
> |---|---|
> | Latent > keyframe | The bottleneck was costing information. The latent arm becomes a second result. |
> | Keyframe ≥ latent | A narrow, supervised interface is worth its loss. This supports the design and is a finding in itself. |
> | Latent ignores shuffled support | Representation transfer repeats GEMIL's context-ignoring failure. Report it as a negative result on the interface; the keyframe story is unaffected. |
>
> Either way, "what should the interface carry" becomes a finding rather than an
> assumption. The cost is about three extra VLA cells plus a small adapter.

## 6. Stage by stage: what we learn, and what transfers to GEMIL

GEMIL (`~/alfa-d-gemil`, branch `gemil-pfn`) infers a scalar task cost E(Y | D) from
action-free demos with a PFN trained on a synthetic prior. PFN-VLA is the same idea
with a richer target (the next interaction) and a VLA as the consumer. Each stage below
says what it tells GEMIL.

### G0: environment and baseline

- **Question:** does the rebuilt pipeline reproduce kv > rgb_cont?
- **Learned (one seed, s1000, seen tier, `G4F-*-s1000-seen.json`):** success was rgb
  0/20, rgb_cont 0/20, kv 7/20. That is the published direction. One seed, so not a
  result.
- **For GEMIL:** nothing new about the science. The practice is to reproduce the
  consumer's baseline before adding any prior, so a later gain or loss cannot be a
  pipeline change.

### G1: synthetic KeyframePFN, offline

- **Question:** does the PFN infer the next interaction from support, beyond
  context-free and retrieval baselines?
- **Learned** (3 seeds each, `octvla-prior/results/g1-edges-v1.json`). Keyframe
  accuracy, ambiguous instruction, mean [min–max]:

  | Split | PFN, correct support | PFN, shuffled support | PFN, no support | Retrieval | Copy-current |
  |---|---|---|---|---|---|
  | train | 0.70 [0.70–0.70] | 0.26 | 0.27 | 0.32 | 0.11 |
  | composition | 0.51 [0.50–0.52] | 0.27 | 0.23 | 0.20 | 0.05 |
  | shelf_family | 0.86 [0.85–0.86] | 0.39 | 0.29 | 0.40 | 0.15 |

  - Contrast accuracy is 0.45, 0.19 and 0.77 against 0.17, 0.10 and 0.30 for retrieval.
  - Support is used: shuffling it brings the PFN down to no-support level.
  - Held-out composition is the weakest split.
  - **Edges ablation:** without its difference edges the PFN matches the full model on
    every split and regime, with overlapping seed ranges, despite 9.5 % fewer
    parameters. G1 does not rely on pairwise relational edges, and the paper must not
    claim them.
- **For GEMIL:**
  - The PFN's edge module is ported from GEMIL's `edges.py`. The same ablation there is
    one flag and could remove a component that does nothing.
  - The weak composition split reproduces GEMIL's held-out selector×goal failure with
    a different target and architecture. So the failure is probably a property of the
    synthetic-prior approach, not of GEMIL's cost formulation.
  - Conflict episodes and the contrast metric made context use measurable here from day
    one. That is GEMIL's C2R/C2X lesson confirmed in a second setting.

### G1b: PFN offline on shelf clips (pending)

- **Question:** does the synthetic PFN pick the right next object on real collected
  shelf keyframes, for the demonstrated order?
- **Learned** (3 seeds per prior, 52 leftmost + 43 rightmost recorded demos, K = 3
  support, ambiguous instruction, `octvla-prior/results/g1b-shelf-v1.json`):
  - **It follows the support on real scenes.** With no support the prior defaults to
    "leftmost first": on rightmost scenes it picks the rightmost object at frame 0 only
    0.00–0.35 of the time. Rightmost support raises that to 0.98 for interaction seeds
    1–2, but only 0.37 for seed 0; the goal-only prior reaches 0.88–1.00 on all seeds.
  - **Before each grasp it names the right object almost always.** With same-rule
    support, next-object accuracy is 0.99 [0.99–1.00] for the interaction prior,
    against 0.76 without support. While carrying it is 0.81, and after placing 0.98.
  - **It fails during compaction.** The right arm's push is about 28 % of evaluated
    frames, and the prior has no vocabulary for it. Next-object accuracy there drops to
    0.33, event to 0.11, and acting arm to 0.20: it reads the pushing arm as the one
    about to act.
  - **Positions are coarse.** Median errors against the oracle rows are 9.6 cm for the
    object, 9.4 cm for the gripper and 7.1 cm for the goal. The prior's row arrangement
    starts at a deck edge, while the shelf's starts near the deck centre.
  - **The goal-only prior matches or beats the interaction prior** on next-object
    accuracy (0.97 against 0.81 while carrying) and follows support more reliably.
    This is a warning for the "interaction pretraining matters" claim, which Step 3's
    goal-only arm tests directly.
- **For GEMIL:**
  - The sim gap is not uniform. A synthetic prior reads simulator states it has a
    vocabulary for almost perfectly. It fails on a behaviour it has none for (here a
    push), and fails confidently, not by abstaining. GEMIL should check which of its
    targets' behaviours the synthetic prior actually covers before trusting E(Y | D)
    on them.
  - Default preferences show up only without context. The prior's "leftmost first"
    default is invisible when supports agree with it, so the no-support condition is
    needed to see what the support actually contributes.

### Compaction: a coverage decision, not a confidence problem (2026-11-12)

The prior has no vocabulary for the compactor's push, about 28 % of shelf frames. Two
facts settle how to treat it:
- **The oracle never describes the push.** Once a placed object is let go, the oracle
  keyframe points to the next grasp. The G2 policies trained on that.
- **Stage B uses the same definition.** Its real targets on compaction frames are the
  next recorded grasp.

So calibration is the first fix, and G1b′'s per-stage table tests it. If compaction
accuracy stays low, the deployable alternative is a coverage gate: an observable stage
classifier that masks the prior during compaction (exactly kv there). Not the
confidence gate: the frozen prior is overconfident (above), and predictive confidence
does not detect missing vocabulary.

**For GEMIL:** a synthetic prior meets behaviours outside its program grammar. Decide
explicitly whether calibration should teach it to look past them, or whether a
coverage gate should switch it off there; do not expect its confidence to say so.

### Known defects of the frozen prior (found 2026-11-12)

- **Object yaw stuck 180° off.** P1's object-pose head predicts yaw as a unit vector,
  `F.normalize(raw)`. A prediction exactly opposite its target gets a purely radial
  error, which the normalisation projects out, so its gradient is zero: a saddle. All
  three P1 seeds sit in it, on their own synthetic data too (cos-yaw loss 1.93). G1
  scored positions only, so nobody saw it.
  - The frozen-PFN views therefore carry a flipped yaw in the object-keyframe row; the
    goal row is unaffected.
  - Stage B switches to a rotated yaw (current yaw times a delta that starts at
    identity), which has no such saddle.
  - **For GEMIL:** its prior uses the same unit-vector parameterisation. Score angles,
    not only positions, and check for antipodal saddles.
- **Overconfident on real scenes.** Across the nine frozen views, mean confidence is
  0.80–0.98, while the predicted object matches the recording's next one on only
  0.68–0.78 of frames. A confidence gate on the frozen prior is therefore close to
  constant. G1b′ (job 4990170) measures calibration by stage before any gated result is
  read.

### G1b′: calibration on real demos, and the from-scratch control (2026-11-12)

Validation-seed queries, training-seed supports, 3 seeds per prior
(`octvla-prior/results/g1b-val-v1.json`):

| Prior | Rule switch at frame 0 | Compaction next object | Object / gripper / goal error (cm) | ECE |
|---|---|---|---|---|
| P1 frozen | 0.77–1.00 (rightmost scenes 0.31–1.00) | 0.32 | 9.5 / 9.3 / 7.0 | 0.24–0.51 |
| F1 full calibration | 1.00 | 1.00 | 0.3 / 0.2 / 0.9 | 0.00 |
| F2 heads only | 1.00 | 0.91 | 0.6 / 0.5 / 1.1 | 0.09–0.15 |
| S1 from scratch, real only | 1.00 | 1.00 | 0.4 / 0.4 / 1.0 | 0.00–0.01 |

- **Calibration fixes everything G1b found.** Compaction is solved by teaching the
  prior's targets to look past the push, so no coverage gate is needed.
- **But the from-scratch control matches it.** On the rules and scene type the real
  data covers (3 objects, leftmost/rightmost), synthetic pretraining buys nothing.
  Its value can only show where the real data runs out:
  - a held-out rule (middle-first);
  - other object counts (2 and 4);
  - held-out sizes;
  - fewer real demonstrations.

  Those are the tests that decide the pretraining claim; an in-distribution validation
  cannot.
- **For GEMIL:** the same trap. Pretraining has to be judged off the fine-tuning
  distribution, and against a from-scratch model trained on the same real data.
  GEMIL's own finding (pretraining bought speed, not accuracy) is what an
  in-distribution test shows.

### Steering probe: another rule's oracle keyframe on single-rule policies (2026-11-12)

The three leftmost-trained oracle-keyframe policies (G5F), given rightmost keyframes:
- 0/20 success and 0 transfers on every seed (7–13/20 with leftmost keyframes);
- the robot lifts something in only 6–8 of 20 scenes, and its first pick is mostly the
  **middle** object (5 of 6–8), between its learned leftmost habit and the keyframe's
  rightmost target (`score_pick_order.py`).

GR00T does not ignore the keyframe, but trained on one rule it cannot follow another:
closed-loop switching needs a VLA trained on at least two rules.

### Oracle keyframe off-distribution (2026-11-12)

On held-out sizes and 4 objects the oracle often cannot plan: 12–38 % of steps have an
empty keyframe, against almost none on seen scenes. The policy trained with the
keyframe nearly always present then runs blind. The oracle row is therefore unreliable
exactly there; report it with its empty share.

| Condition | kv | oracle |
|---|---|---|
| held-out | 3/60 | 2/60 |
| 2 objects | 31/60 | 34/60 |
| 4 objects | 5/60 | 1/60 |

A learned prior always predicts, so it has no such gap.

### G2: oracle keyframe into GR00T

- **Question:** can GR00T use a keyframe at all (Section 3)?
- **Learned (3 training seeds, 20 seen scenes each, `G4F-kv-s100{0,1,2}-seen.json`
  against `G5F-kv-s100{0,1,2}-seen.json`):**

  | Seed | kv: success · ≥1 transfer · mean transfers | kv + oracle |
  |---|---|---|
  | 1000 | 7 · 11 · 1.25 | 13 · 17 · 2.20 |
  | 1001 | 10 · 13 · 1.70 | 7 · 12 · 1.35 |
  | 1002 | 7 · 11 · 1.30 | 13 · 16 · 2.10 |
  | pooled | 24/60, range 7–10 | 33/60, range 7–13 |

  - The oracle arm wins on two seeds by +6 and loses on one by −3. The ranges overlap,
    so the advantage is real in direction but **not reproducible on every seed**.
  - **Where it wins:** placed objects rise from 25 to 44 and from 26 to 42, and
    objects lost after placing fall from 17 to 6 and from 14 to 9. The gain is in
    completing and keeping transfers, which is what the keyframe's grasp, goal and
    timing information should give.
  - **Where it loses (s1001):** objects never lifted double, from 11 to 21. The loss is
    at the first grasp, not later in the transfer. It looks like ordinary training-seed
    variance, which CLAUDE.md shows can exceed any effect here.
  - On 2–3 scenes per seed the policy reached states the expert cannot plan from, so the
    keyframe was empty for part of the episode (24–418 of 600 steps). Those steps ran as
    kv, so they can only understate the oracle arm.
- **Engineering lessons:**
  - An oracle that crashes on off-distribution states kills the rollout. An oracle that
    silently degrades hides how often it was absent. The fix is an empty keyframe that
    is counted (`future_keyframe_unplannable_steps`).
  - A zero-initialised branch with a zero-initialised scalar gate would get no gradient.
    The branch therefore has no gate and only zero `to_k`/`to_v`, and the
    stage-2 initialisation check (`scripts/check_stage2_init.py`) now asserts that
    gradient reaches it.
- **For GEMIL:**
  - GEMIL has no equivalent split. It should report an **oracle-cost reference**: the
    planner minimising the true task distance d_z. That separates "the cost was
    inferred wrongly" from "a correct cost was used badly", which GEMIL's end-to-end
    numbers cannot.
  - The off-distribution lesson carries over directly: a planner driven by an inferred
    cost will reach states its demos never covered, and the cost must be defined, and
    counted, there.

### G3: learned PFN into GR00T (pending), with the latent arm

- **Question:** does the gain survive with learned predictions? How much of the
  kv→oracle gap does the PFN recover? Keyframe or latent cells (Section 5)?
- **Learned:** pending.
- **For GEMIL:**
  - GEMIL's scalar energy is the narrowest possible interface. If keyframes ≥ latent
    here, that supports a narrow, supervised output. If latent > keyframe, a single
    scalar E is probably discarding usable information, and a structured output (e.g.
    a keyframe-like goal) may serve GEMIL's planner better.
  - The shuffled-support check on the latent arm is GEMIL's context probe applied to
    representation transfer. It shows whether context-ignoring is specific to
    end-to-end training or appears whenever a consumer is handed a rich latent.

### Low-data regime: 19 training runs (pending)

- **Question:** the main claim. Does the PFN's gain over kv grow as VLA training data
  shrinks?
- **Learned:** pending.
- **For GEMIL:** GEMIL found pretraining bought convergence speed, not final accuracy.
  If the PFN's gain here is largest at 19 runs and shrinks at 38, that is the same
  phenomenon seen as sample efficiency. It suggests GEMIL should report data-efficiency
  curves, not end-point accuracy.

### Step 4: held-out ordering rule, and RoboTwin (pending)

- **Question:** can support demos alone convey a rule GR00T never trained on
  (middle-first)? Does the gripper head matter on a bimanual task, where the acting arm
  varies?
- **Wording rule:** a held-out rule the PFN saw in synthetic pretraining is reported as
  *transfer of pretrained task knowledge plus in-context adaptation*, not as
  compositional generalisation. Composition is claimed only if a held-out pair
  succeeds.
- **Learned:** pending.
- **For GEMIL:**
  - The held-out rule is GEMIL's central claim (a new task from a few action-free
    demos) in a harder consumer.
  - A success here would be the first evidence that the synthetic-prior route
    identifies a rule a downstream learner could not.
  - A failure, given G1's weak composition split, would point both projects to the
    prior's coverage rather than the consumer.
