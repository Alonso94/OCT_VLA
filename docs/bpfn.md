# BimanualPFN code (`src/oct_vla/bpfn/`)

The paper is `paper/BimanualPFN.tex`. The package follows the layout of TRB (`src/trb/`).

```
bpfn/
  schema.py, quat.py        the pose-window table contract and quaternion helpers
  data/
    syn_data/               the analytic prior: programs, trajectories, branches, observation
    real_data/              recorded episodes (RoboTwin)
    batches.py              items: (support table, query table, target window), packing
    bank.py                 the pre-generated episode bank (storage and loading)
    synthetic.py            SyntheticData: builds the bank and the evaluation packs
    splits.py, packs.py     split registry, fixed evaluation packs
```

The remaining layers (`models/`, `training_tools/`, `benchmark_tools/`, `envs/`, `downstream/`) are described below as they are added.

## Stage runners

| Paper stage | Runner (`scripts/bpfn/`) |
|---|---|
| 1, prior data | `stage1_generate_data.py`: the episode bank and the evaluation packs |

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
