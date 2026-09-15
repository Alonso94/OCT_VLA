# Brief: object-centric VLA sweep on NHR@FAU Alex

**Audience:** a Claude Code session running *on the cluster*, with no prior
context from the session that wrote this. Read it end to end before acting.

Repo-relative paths are from the checkout root (`$OCTVLA_REPO`). Paths starting
`lerobot/` are inside the installed LeRobot package in the policy virtualenv —
find it with
`$OCTVLA_POLICY_PYTHON -c "import lerobot,os;print(os.path.dirname(lerobot.__file__))"`.
Line numbers were accurate at commit `7647ca4`; re-grep if they have moved.

Your first job is **not** to launch training. It is to measure the cluster
envelope, because the repo's own docs do not record it and every sizing
decision below depends on numbers nobody has yet collected.

---

## 1. What the project is

RoboTwin/SAPIEN shelf-restocking task. A scripted oracle produces
demonstrations; VLA policies are LoRA-finetuned on them and evaluated
closed-loop in simulation through a client/server bridge.

The scientific question: **does conditioning a VLA on object-centric state
help, versus RGB alone?** Two backbones (pi0.5, SmolVLA), with and without RGB,
using ground-truth object state now and perception-derived state later.

### What already works (verified, committed)

| Piece | Where |
| --- | --- |
| Data collection (2/3/4-object profiles) | `slurm/submit_shelf_restock_collection.sh` |
| Split building | `scripts/build_shelf_restock_splits.py` |
| pi0.5 LoRA finetune, `rgb` and `object` variants | `slurm/train_pi05_shelf_restock.sbatch` |
| Object-conditioned plugin (ControlVLA-style zero-init residual) | `src/oct_vla/policies/control_pi05/` |
| Closed-loop eval via sim bridge | `slurm/eval_shelf_restock.sbatch`, `src/oct_vla/serve/` |
| Workflow documentation | `docs/running_experiments.md`, `docs/setup_nhr_alex.md` |

The object variant was verified end to end on a GPU: it trains, saves a
checkpoint whose recorded type is `control_pi05`, reloads, and runs
single-observation inference. Crucially `object_injection.weight` moves off its
zero init (9.36e-05 after 6 steps), which is the only evidence that object
tokens actually reach the model — `batch.get()` returns `None` for a missing
key rather than raising, so a healthy loss curve does **not** prove the
conditioning is live. Re-run that check for every new variant.

### Datasets

Only `three_object` exists: 123 clips (90 train / 33 val), published as
`3liyounes/oct-vla-shelf-restock-three_object-{rgb,object}` v1.0.0, private.
Download with `scripts/hub_dataset.py download ... --release v1.0.0`.

The offline **test** blocks reserved in `docs/dataset_protocol.md` (seeds
250-349, 400-549, 600-749) were never collected. Count-shift is measured
closed-loop in simulation instead. Either collect them or state plainly that
closed-loop sim is the primary evidence and the 33 val clips are the only
offline held-out set.

---

## 2. The finding that shapes the whole design

`src/oct_vla/data/object_tokens.py` builds a 15-d token: position (3),
orientation (4), size (3), visibility + confidence (2), and a **role one-hot**
(3) — target / previous-neighbour / other — taken from `TaskContext`. Tokens
are additionally ordered **target-first**.

So the object-conditioned policy is *told which object to move next and which
to compact against*. The RGB policy must infer both from pixels.

A naive "object beats RGB" result is therefore uninterpretable: it may be
task-state leakage rather than object-centric perception, and it is the first
thing a reviewer will find. **Stage 2 below compares three arms, not two, and a
positive result only counts if it survives role-stripping.**

---

## 3. Stage 0 — analyse the infrastructure (do this first)

`docs/setup_nhr_alex.md` records the environment build but almost nothing about
Slurm limits. Establish and write down:

**Cluster shape**
- Partition names and which to use for a40 vs a100; account/QoS conventions.
- GPUs per node, node counts, and whether multi-GPU jobs are worth it here.
- Max wall-time per partition. The sbatch files request 24 h (train) and 8 h
  (eval) — confirm those are actually allowed.
- Concurrency: `MaxSubmitJobs`, `MaxJobs`, per-user running-job caps, fair-share
  behaviour. This decides whether a 12-run stage lands in parallel or serialises.
  Useful: `sacctmgr show assoc user=$USER format=account,partition,maxjobs,maxsubmit`,
  `sinfo -o "%P %l %G %D %t"`, `scontrol show partition`.
- `MaxArraySize` is documented as **10000** on Alex (`docs/setup_nhr_alex.md:126`)
  — confirm, since the array launcher depends on it.

**Storage** (documented, verify): `$HOME` 100 GB backed up; `$HPCVAULT` 1 TB,
mounted on compute nodes, holds the ~32 GB RoboTwin assets, HF cache and
training outputs. `$WORK` was deliberately avoided (group inode quota). Check
free space before a sweep — each checkpoint set is non-trivial and the sweep
produces dozens.

**Throughput — the measurement that matters most**
- Run **one full 20k-step pi0.5 RGB finetune** and record wall-clock. Nothing in
  the repo states how long this takes; `--time=24:00:00` is an allocation, not a
  measurement. Everything below is sized off this number.
- Determine whether the **GPU or the dataloader** is the limit. Without
  torchcodec, decoding is 204.8 ms/sample vs 21.6 ms
  (`docs/setup_nhr_alex.md:130-141`); at batch 16 that would dominate. Confirm
  torchcodec is actually active in the training env via
  `OCTVLA_TORCHCODEC_LIBS`.
- Batch sizing: A40 is 49 GB; batch 2 measured at 10.3 GB (~8.2 GB weights,
  ~1 GB/sample with gradient checkpointing), so batch 16 fits. No A100 guidance
  exists — measure.
- Time one closed-loop eval job (3 profiles × N seeds, `max_steps` 600) so eval
  can be budgeted alongside training.

**Report back** a short table: GPU-hours per training run, per eval run, max
useful parallelism, and total GPU-hours for each stage below.

---

## 4. Stage 1 — build the missing sweep machinery

None of this exists today; all of it is needed before a sweep is meaningful.

| Gap | Fix |
| --- | --- |
| `--seed` hard-coded to 1000 (`slurm/train_pi05_shelf_restock.sbatch:109`) | make it an env override |
| Run naming is job-id only, so cells are indistinguishable | encode `{backbone}_{conditioning}_{tokens}_{seed}` in the output dir |
| No training sweep launcher | add a training array job, modelled on the existing collection array launcher |
| No aggregation, no confidence intervals | `scripts/aggregate_eval.py`: merge per-job eval JSONs, Wilson intervals, **paired** per-seed differences |
| `ObjectTokenSpec` has no role-stripped mode | add one (drop the 3 role dims *and* the target-first ordering) |
| No shuffled-token control | add an eval-time flag permuting tokens across objects |
| Dataset dir naming mismatch: sbatch expects `three_object_rgb`, the download is `three_object-rgb` | fix before anything runs |

On statistics: **evaluate every cell on identical scene seeds and report paired
differences.** Pairing removes scene difficulty from the comparison and matters
more than raw sample size — with ~30 episodes an unpaired 95 % Wilson interval
is roughly ±0.18, wide enough to hide the effect being measured.

---

## 5. Stage 2 — the decisive experiment (pi0.5 only, 12 runs)

| Arm | Conditioning | Purpose |
| --- | --- | --- |
| A | RGB only | baseline |
| B | RGB + object tokens (full, GT) | the headline claim |
| C | RGB + object tokens, **role-stripped** | is B real, or leakage? |
| — | B re-evaluated with **shuffled** tokens | does the policy use them at all? (eval-only, free) |

3 training seeds each. Evaluate across all three count profiles (2/3/4 objects
— the harness already supports this; it is a free generalisation axis).

**Gate.** If B does not beat A, the rest of the matrix is moot — and that is a
publishable negative. If B beats A but C does not, the effect is leakage and the
token schema must change before anything else runs.

---

## 6. Stage 3 — breadth, only if Stage 2 passes (12 runs)

- **SmolVLA** × {RGB, object} × 3 seeds. Close to 1:1 with the existing plugin:
  `VLAFlowMatching.embed_suffix` (`<site-packages>/lerobot/policies/smolvla/modeling_smolvla.py:646`)
  is the analogue of the pi0.5 hook but returns a 3-tuple, not 4; PEFT targets
  already exist (same file, `:420`), so the `modules_to_save` trick transfers verbatim.
  `lerobot/smolvla_base` should be in the HF cache — confirm on the cluster.
- **Object-only (masked RGB)** × both backbones × 3 seeds. Both policies raise
  when no image key is present (`lerobot/policies/pi05/modeling_pi05.py:1060`,
  `lerobot/policies/smolvla/modeling_smolvla.py:341-345`), so images stay and are masked rather than removed.
  SmolVLA honours a real `..._padding_mask` (smolvla `:365-368`); pi0.5 does not and
  needs a `_preprocess_images` override. Because the VLM is off-distribution on
  blank images, this **lower-bounds** RGB's value — say so when reporting.

---

## 7. Stage 4 — recommended additions

Not requested, but the sweep as specified measures neither of object-centric
conditioning's actual theoretical advantages (sample efficiency, appearance
robustness). In priority order:

1. **Data-scaling curve** (10 / 30 / 90 episodes × {RGB, object} × 3 seeds).
   ControlVLA's claim is few-shot adaptation; this tests it directly, the runs
   are short, and it is the most likely place object conditioning wins.
2. **Distractor / novel-appearance profile.** The strongest argument for object
   tokens is appearance invariance, and it is currently untestable: the asset is
   hardcoded `OBJECT_MODEL = "113_coffee-box"`
   (`src/oct_vla/tasks/shelf_restock/robotwin_env.py:73`) with one
   variant per episode. Add a task subclass plus a `PROFILE_TASKS` entry
   (`src/oct_vla/serve/server.py:36-40`) — same pattern as the count profiles.
3. **Token-content ablation** (position only → +size → +role). Cheap; tells you
   *what* the policy uses, not merely whether.
4. **Privileged-state upper bound.** A small state+object MLP, no vision. If it
   solves the task, the VLA comparison has little headroom and the sweep is
   measuring noise. Cheap insurance, run once.

---

## 8. Later — SAM-based perception (deferred, do not start yet)

This is the intended follow-up after the stages above, not part of this sweep.
Recorded now only so the interface is not designed into a corner.

Ground-truth tokens answer "does object conditioning help at all". A
perception-derived estimate then answers "how much survives real perception
noise" — and the gap between them is the sim-to-real result. That comparison is
only interesting once GT tokens have been shown to win.

The seam already exists: `ObjectStateEstimator` Protocol
(`src/oct_vla/perception/base.py:9-11`), constructed in exactly one place
(`src/oct_vla/serve/server.py:178-180`). A SAM2/detector/tracker estimator
implements `estimate(observation) -> ObjectScene` and drops in.

Two constraints to respect when the time comes: `track_id` must be **persistent
across frames**, because token ordering and roles key on it; and
`visibility`/`confidence` are currently hardcoded to 1.0 by the ground-truth
source, so real detector scores would finally carry information. Evaluate
existing checkpoints with perception tokens first (no retraining), then
finetune with them — the train/test mismatch is itself a result worth
reporting.

---

## 9. Verification discipline

1. Before trusting `aggregate_eval.py`, confirm it reproduces the summary
   numbers already present in an existing eval JSON.
2. Every new backbone/variant gets a 6-step smoke run first, then the
   injection-weight check (`object_injection.weight` non-zero after training).
   That check is the only thing separating live conditioning from a silent
   no-op.
3. One full-length run must complete and evaluate end to end before any array
   is launched at scale.
4. Report closed-loop success rate with Wilson intervals and paired per-seed
   differences; include `transfers_completed` (partial credit, already recorded)
   and the per-profile count-shift breakdown.
