# OCT-VLA

Does conditioning a robot policy on object-centric state help over RGB alone?
Dual-arm Panda, RoboTwin/SAPIEN, LeRobot policies (ACT, pi0.5, SmolVLA, GR00T).

Read `docs/project_status_2026_09_22.md` first — it is the current state of the
science and it supersedes `docs/report_2026_09.md`.

## Environment

```bash
source ~/octvla/env-leftmost.sh            # sets every OCTVLA_* path
PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" -m pytest tests/ -q
```

- `$OCTVLA_POLICY_PYTHON` is the **only** interpreter with torch + LeRobot.
  A bare `python` has neither, and a suite that "passes" there has skipped
  every policy test. Always run tests with it.
- The simulator lives in a *separate* Python (`$OCTVLA_ROBOTWIN_PYTHON`,
  NumPy 1.26 / Torch 2.4) because it cannot share an interpreter with the
  policy env. That is why evaluation runs a server and a client.
- Login nodes have **no GPU**. Anything touching SAPIEN needs a batch job.

## Rules that were learned the expensive way

**Validation loss does not predict closed-loop success here.** Nine documented
cases; twice the *better* offline cell was the worse policy. Never rank cells,
select checkpoints, or gate experiments on it. Use
`scripts/diagnose_action_head.py` as a floor check against the constant
predictor, nothing more.

**A single-seed cell is not a result.** Our headline finding (6/20) scored 0/20
and 0/20 on two further training seeds. Seed variance exceeds every effect this
project has measured. Run ≥3 seeds; `scripts/collect_final_results.py` prints
min–max across seeds and refuses to report a single-seed cell.

**Re-derive every number from `$OCTVLA_OUTPUT_ROOT/{eval,diagnostics}/*.json`.**
Do not quote a previous document. Doing this has caught real arithmetic errors
in our own reports more than once.

**Measure before concluding a change broke something.** A collection probe was
reported as "0 of 20 collected" and blamed on a code change; the real cause was
grepping for status `"collected"` when the success value is `"ok"`. Check the
data format before trusting a count.

**Geometry does not predict plannability.** Five of six assets that passed a
gripper-width / clearance / separation screen collected 0 of 8 seeds — the
oracle could not plan a grasp around those meshes. Probe any new asset on a few
seeds before scaling.

**RoboTwin's `move()` silently no-ops after a planning failure**
(`_base_task.py`: `if self.plan_success is False: return False`), where our port
raises. Any loop driving a built-in `play_once` must check `task.plan_success`
afterwards or it records no-op episodes as successes.

**Silent failures are the norm in this codebase's bug history** — uint8 images
fed 255× out of range, conditioning inert at evaluation only, statistics fitted
over train+validation, a branch left undropped while its host used dropout.
When adding a path, ask what it would look like if it silently did nothing, and
add the assertion that would catch it.

## Conventions

- Evaluation: seeds **800–819**, outside every reserved block
  (`docs/dataset_protocol.md`), 600 steps, `n_action_steps=25`.
- Scoring is atomic: success, ≥1 transfer, ≥1 lift, mean transfers. Report
  against each cell's own oracle ceiling, not against 100 %.
- Datasets are one **unified** export with per-encoding views projected from it
  (`scripts/project_dataset_view.py`); video is shared by hard link.
- Object conditioning: `object_injection_mode` ∈ `layerwise` (published
  ControlVLA, per host attention layer) | `controlvla` (single seam) | `pooled`
  (weakest, kept so old checkpoints load).
- Entity tokens: v2 is 17 columns of geometry; v3 is 32, balanced 16 geometric
  against 16 semantic (`src/oct_vla/data/entity_semantics.py`).

## Before you commit

Run the full suite with `$OCTVLA_POLICY_PYTHON`. Do not commit with a failing
test — that has happened and was not caught until the next run.

Ask before deleting checkpoints or datasets, and before pushing anything to an
external service. Deletions here are irreversible and some artifacts are the
only provenance behind published numbers.
