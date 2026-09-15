# Measured cluster envelope: NHR@FAU Alex

Stage 0 of `docs/experiment_sweep.md`. `docs/setup_nhr_alex.md` records how the
environment was built; this records what it actually costs to run, because every
sizing decision in the sweep depends on numbers nobody had collected.

Measured 2026-09-15 on job `4247962` (pi0.5 LoRA, `three_object_rgb`, A40) and
eval jobs `4250647` / `4250724`. Re-measure if the policy, batch size or the
dataset changes.

## Cluster shape

| | |
| --- | --- |
| Partitions | `a40` (default), `a100`, `rtxpro6k`, `a100mig` |
| GPUs per node | 8 on all GPU partitions |
| `a40` capacity | 44 nodes, 352 GPUs, 128 CPU / 510 GB per node |
| `a100` capacity | 37 nodes, 128 CPU / 1030 GB per node |
| **Max wall-time** | **24 h on every partition** — the sbatch files' `--time=24:00:00` is the ceiling, not a margin |
| Max nodes per job | **1** (`MaxNodes=1` on `a40`) — multi-node training is not available |
| `MaxArraySize` | 10000 (confirmed) |
| `MaxJobCount` | 50000 |
| Per-user job caps | **None.** The `normal` QoS sets no `MaxJobsPU` or `MaxSubmitPU`, so a 9-cell stage lands in parallel, subject only to fair-share |
| Account / QoS | `g107ea`, QoS `normal`; `--gres=gpu:<type>:1` is mandatory on every job, including CPU-only work |
| Default CPU/GPU | `DefCpuPerGPU=16`, `DefMemPerCPU=3750M` (so 16 CPUs ⇒ 60 GB) |

## Storage

| Path | Used | Quota | Note |
| --- | --- | --- | --- |
| `$HOME` | 97 GB | 100 GB soft / 200 GB hard | **93% full.** Holds both venvs, RoboTwin and the datasets. Not for checkpoints |
| `$HPCVAULT` | 85 GB | 1 TB soft / 2 TB hard | Training outputs and HF cache live here. Ample room |
| `$WORK` | unused | — | Deliberately avoided (group inode quota) |

A checkpoint set is ~5 MB (LoRA adapter only), ~96 MB per run with optimizer
state across 10 saves. A 9-cell stage is well under 1 GB — storage is not a
constraint on the sweep; `$HOME` headroom is.

## Throughput

**Training is GPU-bound, not dataloader-bound.** At batch 16 on an A40:

| Quantity | Measured |
| --- | --- |
| `step_s` | 4.01 s |
| Wall s/step incl. periodic eval + checkpoint saves | **4.13 s** |
| `data_s` | 0.001 s — torchcodec is active; decoding is not the limit |
| GPU utilisation | 100%, 307 W against a 300 W cap, SM 1545 MHz |
| GPU memory | 15.6 GB of 49 GB |
| Startup (job start → first logged step) | ~13 min |

Because the card is power-capped at 100% utilisation with 33 GB of memory
spare, **larger batches will not help** — the only real lever on step time is
turning off `--policy.gradient_checkpointing`, which trades some of that spare
memory for compute. Untested.

Derived training cost, per run:

| Steps | Epochs (90 clips, batch 16) | Wall-clock |
| --- | --- | --- |
| 2,000 | 2.6 | 2.5 h |
| 6,000 | 7.7 | 7.1 h |
| 20,000 | 25.6 | **23.2 h — inside a 24 h limit by ~50 min** |

20,000 steps is the current default and it is dangerous: a single slow
checkpoint save or a contended node kills the run before it writes its final
checkpoint. It is also probably far more than needed — offline eval loss
plateaus by ~step 1500 (0.109 at 1k → 0.095 at 1.8k, non-monotonic after 1.6k).
Offline loss is a weak proxy for closed-loop success on a VLA, so the step count
is being chosen from a closed-loop checkpoint sweep instead
(`slurm/submit_checkpoint_sweep.sh`).

## Evaluation

| Quantity | Measured |
| --- | --- |
| Fixed overhead (SAPIEN + cuRobo warm-up, pi0.5 load) | ~3 min per job |
| Closed-loop step | ~0.8 s (287 steps in a 6:54 job) |
| Full 600-step episode | ~8 min (estimated; no policy yet survives 600 steps) |

At ~8 min per episode, one cell evaluated on 3 profiles × 30 seeds is **~12 h —
over the eval sbatch's 8 h limit.** Evaluation must be split, one job per
profile (~4 h each), not run as a single 90-episode job. Episodes that
terminate early are much cheaper, so this is a worst case.

## Stage budgets

Stage 2 is 3 arms × 3 training seeds = **9 runs** (the brief's heading says 12;
its fourth row is the eval-only shuffled control, which trains nothing).

| Stage | Training | Eval (worst case) |
| --- | --- | --- |
| Stage 2 @ 6k steps | 9 × 7.1 h = **64 GPU-h** | 12 cells × 12 h = ~150 GPU-h |
| Stage 2 @ 20k steps | 9 × 23.2 h = **209 GPU-h** | unchanged |
| Stage 3 (+SmolVLA, +masked-RGB) | ~2× Stage 2 | ~2× Stage 2 |

With no per-user job cap, all 9 cells run concurrently, so Stage 2's training
wall-clock is one run's duration plus queueing — about 7 h at 6k steps.
