# BimanualPFN code (`src/oct_vla/bpfn/`)

The paper is `paper/BimanualPFN.tex`. The package follows the layout of TRB (`src/trb/`).

**Standalone.** `oct_vla.bpfn` imports nothing else from `oct_vla`; `tests/bpfn/test_standalone.py` enforces that. Where it keeps its own copy of something the rest of the repository also defines, a test outside the package holds the two equal:
- the recording format (`data/real_data/robotwin.py` against `oct_vla.data.store`);
- the shelf decks (`SHELF_DECKS` against the task spec);
- the let-go rule (`benchmark_tools/timelines.py` against `oct_vla.data.effects`).

The executor-side code (the token embedding, residual actions, ACT/GR00T) lives with the policies and imports `bpfn`, never the reverse. The stage runners in `scripts/bpfn/` connect the package to the rest: the evaluation server client, LeRobot views, the collectors.

```
bpfn/
  schema.py, quat.py        the pose-window table contract and quaternion helpers
  data/
    syn_data/               the analytic prior: programs, trajectories, branches, observation
    real_data/robotwin.py   recorded RoboTwin runs -> Episode (states only, no cameras)
    real.py                 RealData: runs per rule, rule check, bounded episode cache
    batches.py              items: (support table, query table, target window), packing
    bank.py                 the pre-generated episode bank (storage and loading)
    synthetic.py            SyntheticData: builds the bank and the evaluation packs
    splits.py, packs.py     split registry, fixed evaluation packs
  models/
    __init__.py             build(config) / load(run_dir) for either architecture
    pose_window_pfn.py      "axial": PoseWindowPFN, the first model
    table_pfn.py            "table": TablePFN, cells of pose + header (TabPFN / ConTextTab style)
  training_tools/
    data.py                 samplers (bank / on the fly), the item stream, the data loader
    trainer.py              TrainConfig, PFNTrainer, the overfit gate
    validator.py            dev-pack scores during training (logged, never used to select)
    losses/window.py        the window loss (paper Eq. loss)
  benchmark_tools/
    evaluator.py            Evaluator: correct / shuffled / no support, hold and replay baselines
    real_evaluator.py       RealEvaluator: same-rule / other-rule / no support on recordings
    metrics.py              pose errors, the first-pick decision rule, paired both-correct
    atlas.py, render.py     the marker scan and figures
    checker.py              independent geometric checker of episodes
    timelines.py            let-go rows in a recorded run (decision times)
    fixtures.py             the first-slice acceptance fixtures
  envs/
    controllers.py          PFNController, HoldController: read an Episode, return [2, 7] poses
    robotwin/
      history.py            server observations -> the measured history as an Episode
      grippers.py           ObservableGripper, OracleGripper (privileged)
      oracle.py             ExpertWindowController (privileged bridge check)
      rollout.py            run_episode against the evaluation server
      future_source.py      PoseTokenFutureSource: the executor's pose tokens (stage 3)
  downstream/
    tokens.py               the pose-token format (window_tokens, support_tokens, bank)
    views.py                choose_supports, PoseTokenBuilder: per-frame pose tokens for executor views
```

**Adding an environment.** Controllers read only an `Episode`, so they carry over to any environment. A new environment gets its own `envs/<name>/`, with:
- a history adapter (its observations to `Episode` rows);
- its gripper sources;
- a rollout loop against its own client;
- for stage 3, a future source.

Recorded demonstrations from it enter as a new `data/real_data/<name>.py` reader.

## Stage runners

| Paper stage | Runner (`scripts/bpfn/`) |
|---|---|
| 1, prior data | `stage1_generate_data.py`: the episode bank and the evaluation packs |
| 1, prior training | `stage1_train_prior.py`: one arm x seed; `--overfit` runs the CPU memorisation gate |
| 2, offline evaluation | `stage2_evaluate.py --data synthetic` (a pack), `--data real` (recordings), `--data atlas` (marker scan) |
| 2, online pilot | `stage2_online_pilot.py`: the frozen PFN (or the expert / hold) drives the wrists on the server |
| 3, executor views | `stage3_build_views.py`: pose-window (PFN) and support-table (direct demo) views |
| 3, executor training and evaluation | the existing `train_*` / `eval_shelf_restock` sbatches |

On the cluster, `slurm/submit_bpfn.sh <step>` submits each stage, with steps named `stage1-canary`, `stage1-probe`, `stage1-collect`, `stage1-generate`, `stage1-train`, `stage2-evaluate`, `stage2-online`, `stage3-views`, `stage3-act`, `stage3-groot` and `stage3-camera-check`. `DRY_RUN=1` prints the sbatch lines without submitting.

`stage1_generate_data.py --fixtures OUT` runs the first-slice acceptance fixtures. Stage tools: `stage1_canary_report.py` (gates the collection), `stage1_calibrate_tool.py` (tool offset and timing from recordings), `stage3_check_camera_routes.py` (camera wiring through each executor).

## Items (`data/batches.py`)

**Support rows.** A completed demonstration is subsampled to at most `S_ROWS` rows. The sampler uses only what any recording shows:
- rows where a gripper aperture changes;
- rows where an object starts or stops moving;
- the first and last rows;
- uniform fill for the rest.

Synthetic and recorded supports go through the same function. A privileged phase boundary therefore cannot become a timing cue that disappears at deployment.

**Query times** are stratified over three kinds of row, so a predictor cannot win by keeping both arms still:
- moving rows: a wrist moves;
- transition rows: within `TRANSITION` rows of an observable contact change;
- idle rows: everything else.

**Branch families.** A handover query may come from a delayed-receiver branch. Both the delayed and the nominal member then mask targets past the cue.

**Conflict pairs** share one query scene under two programs whose first picks differ. The query time is drawn before the two executions diverge. The inputs are therefore identical, and only the support says which continuation is correct.

## The episode bank (`data/bank.py`)

**Why a bank.** Generating episodes is the expensive part of an item, at about 40 ms each. In recipe 1 the GPU was busy only 36 % of the time because of it. So episodes are generated once, and everything that makes an item distinct is drawn per item.

| Stored once | Drawn per item |
|---|---|
| pool episodes per (program, eligible count) | which support and which query (independent scenes) |
| conflict families: one scene under two programs with different first picks | which family, its two supports, a query time before the divergence |
| handover families: nominal and delayed receiver from one scene | which branch |
| | query time, column shuffles, relabeling, observation faults, support rows |

**One bank serves every arm.** The units cover the full arm's programs at each eligible count, and every other arm's programs are a subset of those. A sampler keeps only the units its arm may see.

**Determinism.** Every unit draws from its own hashed random stream, named by (bank name, kind, program, count, slot). The content does not depend on how many shards or processes built it.

**Layout.** A shard is one pickle of flat float32 arrays plus small per-episode records.
- **Writes are atomic** (temporary name, then rename). A killed job never leaves half a shard, and a rerun builds only the missing ones.
- **Loading:** training loads every shard into RAM before the data-loader workers fork, so the workers share one copy (`gc.freeze` keeps the garbage collector off those pages).
- **Size:** the default spec holds 145 k episodes in 8.7 GiB. A 16-CPU job gets 60 GB, and `Bank` refuses to load a bank larger than half the job's memory.

## Grouped batches (`--per-support K`)

**Why.** Encoding the support is about 87 % of a forward pass: 128 support rows against 16 query rows. With `K > 1`, a batch of B queries holds B / K supports. Each support is encoded once and read by its K queries (`support_index`), and the backward pass sums their gradients into it. The idea follows TRB's batched ConTextTab, where one context serves many query rows. Here the encoder trains rather than staying frozen.

**What a group holds.**
- An ordinary group is one support and K queries of the same program, drawn from independent scenes.
- A conflict draw gives two groups, with supports of programs a and b, over K conflict families of the pair (a, b). Query j of each group shares a scene and a query time, so every item in those groups is a paired item.
- The pair-item share is the same as with single items: 2f / (1 + f) at conflict fraction f.

**Checks.** The tests check three things:
- a grouped batch predicts exactly what the same items predict with one support copy each;
- a wrong `support_index` changes the answer, so the index cannot be silently ignored;
- an index out of range is refused.

## Fresh supports (`--fresh-supports`)

Two pilots, C and D, trained from the bank and scored 0.87 and 0.84 on dev, but only 0.05 and 0.17 on the held-out handover×colour composition. Every run trained on freshly generated episodes composed: recipe 1 scored 0.53–0.65, and pilot A 0.75 at 65k steps.

The likely cause is support memorisation. With 72 support episodes per program, a model can recognise the support and recall its program's behaviour, instead of reading the rule from what the demonstration does.

`--fresh-supports` keeps the bank's queries and conflict families but generates every support anew. Grouping makes this affordable: one new support per K queries. Training logs the held-out and count packs beside dev (`--val`), so a recipe that loses composition shows up during training. These scores are logged only and never used to select a checkpoint.

## Training diagnostics and speed

**Training log.** `history.jsonl` holds window means over `log_every` steps:
- the loss and its translation and rotation terms;
- the gradient norm before clipping, and the share of clipped steps;
- data wait and compute seconds per step, items per second, and peak GPU memory.

**Validation records**, every `eval_every` steps for each pack:
- the pose errors (overall, at h1 / h4 / h8 / h16, and the median and 90th percentile while an arm moves);
- paired both-correct and decision accuracy per (manipulation, selection) family;
- the right-colour rate for colour rules;
- on dev, both-correct under shuffled supports.

`scripts/bpfn/plot_runs.py` draws these curves for several runs. They are for judging runs, never for picking a checkpoint.

**Speed.** A training step runs about 17,000 small tensor operations, so launch overhead, not arithmetic, bounds it. `--compile` fuses the training forward with `torch.compile` into one graph: the data-dependent checks run only when the model is not being compiled, so they still guard validation, evaluation and tests. AdamW runs fused on CUDA. The loss statistics stay on the GPU until each logging window, so the CPU waits for the GPU only there. `--lr-floor` ends the cosine decay at a fraction of the peak learning rate instead of zero.

## The table model (`--arch table`)

`models/table_pfn.py` follows TabPFN and ConTextTab. A cell is a value plus a column identity, and nothing else:
- an entity cell is its pose plus its header (kind, colour, category, size);
- time is its own column, as in the tabular time-series PFNs: one cell per row holding the seconds since that table's first row. A query therefore starts at t = 0 wherever it was cut from its episode;
- each row also has a CLS cell.

There is no support/query marker, observation flag or measurement age. The blocks alternate attention across a row's cells and along each column. A table may have up to 11 entity columns; padded columns are masked.

The demonstration's row CLS cells are pooled by attention (learned seeds) into task tokens. The query reads them in one of three ways (`--task-read`):
- `cell`: as extra cells of every query row;
- `film`: scaling and shifting every query cell;
- `cross`: its row CLS cells attend to the demonstration's rows and task tokens before each block.

There are two outputs (`--output`):
- `head`: learned (arm, horizon) queries read the last query row;
- `rows`: future rows are appended to the query, time known and every value masked, and their wrist cells are read out, as TabPFN fills a test row's target.

### Table-model options (each off by default; A/B against FiLM + rows)

| Option | What it does | Why |
|---|---|---|
| `--frame relative` | each table's positions are taken around its own movable objects' centroid at its first row | a layout shifted as a whole looks the same (instance normalisation, as RevIN in forecasting) |
| `--hypotheses K` | K futures per query plus confidence logits; winner-takes-all loss with a small share of the mean, and cross-entropy on the winner | errors plateau after h8 and the 90th percentile is 3–4× the median: one prediction averages plausible futures |
| `--aux-factors W` | heads on the task vector predict the rule's factors (manipulation, eligibility, order, repeat, compaction), weight W | supervising factors, never the program identity, should speed learning the task vector and favour composition |
| `--support-patch P` | P consecutive demonstration rows become one token | the demonstration costs 62 % of the forward pass for one task vector |
| `--bf16-trunk` | axial blocks in bfloat16 on CUDA; embeddings, pooling and output in float32 | speed, with positions kept in full precision |
| `--ema D` | an exponential moving average of the weights (decay D) is validated and saved | smoother evaluation, often better generalisation |

The rule factors are packed as `s_factors` (−1 where unknown). They are labels: the model never reads them, and a test holds the model inputs bit-identical when the metadata changes.
