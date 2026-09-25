# Representative rollout videos

One video per row that `research_questions.md` reports, copied into
`docs/videos/`. Every rollout's full set of episodes stays on the vault under
`$HPCVAULT/octvla-rollout-videos/final/`.

**"Representative" is a rule, not a choice by eye.** Picking the best
episode is how this project was misled before (a 6/20 headline that scored
0/20 on two more seeds). So for each row:
1. take the **median training seed** by mean transfers (the lower of the two
   middle seeds on a tie), so a lucky seed is not the face of the arm;
2. within it, the episode whose transfers are **closest to that seed's mean**,
   then whose lifts are closest to its mean lifts, then the lowest scene seed.

A row averaging 0.4 transfers therefore shows an episode with no transfer,
which is what a viewer should expect to see. The `transfers` and `success`
columns describe the chosen episode; the "row mean transfers" column is the
row's mean on each of the three training seeds.

Settings: 3 objects run 600 steps, pinned to seen (meshes 1–4), held-out
(0, 6) or novel (5) identities; 2 and 4 objects get 200 steps per object,
seen identities, the two-object scene on the nested layout.

| folder | what it shows | result prefix | section |
| --- | --- | --- | --- |
| `videos/stage_a_all_arms/` | Stage A: every ACT arm, atomic clips | `F` | §4.1–4.3 |
| `videos/chaining_atomic_clips/` | plain RGB ACT on the paired corpus, atomic clips | `AF` | §4.6 |
| `videos/full_runs_all_arms/` | every ACT arm on continuous runs: the chaining study's full-run policy (`rgb`), Stage A2, and the baseline ablations | `CF` | §4.6–4.9 |
| `videos/vla_stage1/` | pi0.5, SmolVLA and GR00T, RGB stage 1, both absolute regimes | `PF` … `GJ` | §4.10 |
| `videos/groot_stage2/` | GR00T stage 2: the budget control and the conditioned arms, seen and held-out objects | `GF` | §4.11 |

## Stage A: every arm, atomic clips (`videos/stage_a_all_arms/`, `F`)

| arm | setting | training seed | scene seed | meshes | transfers | lifted | success | row mean transfers (per seed) | video |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| rgb | 3 objects, seen identities | 1001 | 804 | seen | 0 | 2 | no | 0.90, 0.50, 0.45 | [rgb__three_seen.mp4](videos/stage_a_all_arms/rgb__three_seen.mp4) |
| rgb | 3 objects, heldout identities | 1002 | 801 | heldout | 0 | 1 | no | 0.10, 0.00, 0.00 | [rgb__three_heldout.mp4](videos/stage_a_all_arms/rgb__three_heldout.mp4) |
| rgb | 3 objects, novel identities | 1002 | 800 | novel | 0 | 1 | no | 0.55, 0.15, 0.15 | [rgb__three_novel.mp4](videos/stage_a_all_arms/rgb__three_novel.mp4) |
| rgb | 2 objects, seen | 1001 | 802 | seen | 0 | 1 | no | 0.20, 0.20, 0.25 | [rgb__two.mp4](videos/stage_a_all_arms/rgb__two.mp4) |
| rgb | 4 objects, seen | 1000 | 800 | seen | 0 | 1 | no | 0.40, 0.50, 0.30 | [rgb__four.mp4](videos/stage_a_all_arms/rgb__four.mp4) |
| rgb_cont | 3 objects, seen identities | 1002 | 804 | seen | 1 | 2 | no | 1.40, 0.50, 0.95 | [rgb_cont__three_seen.mp4](videos/stage_a_all_arms/rgb_cont__three_seen.mp4) |
| rgb_cont | 3 objects, heldout identities | 1000 | 801 | heldout | 0 | 1 | no | 0.20, 0.05, 0.25 | [rgb_cont__three_heldout.mp4](videos/stage_a_all_arms/rgb_cont__three_heldout.mp4) |
| rgb_cont | 3 objects, novel identities | 1001 | 800 | novel | 0 | 1 | no | 0.75, 0.15, 0.00 | [rgb_cont__three_novel.mp4](videos/stage_a_all_arms/rgb_cont__three_novel.mp4) |
| rgb_cont | 2 objects, seen | 1000 | 802 | seen | 0 | 1 | no | 0.45, 0.15, 0.55 | [rgb_cont__two.mp4](videos/stage_a_all_arms/rgb_cont__two.mp4) |
| rgb_cont | 4 objects, seen | 1001 | 800 | seen | 0 | 2 | no | 0.45, 0.35, 0.15 | [rgb_cont__four.mp4](videos/stage_a_all_arms/rgb_cont__four.mp4) |
| rgb_novae | 3 objects, seen identities | 1000 | 801 | seen | 1 | 2 | no | 0.65, 1.40, 0.45 | [rgb_novae__three_seen.mp4](videos/stage_a_all_arms/rgb_novae__three_seen.mp4) |
| rgb_novae | 3 objects, heldout identities | 1002 | 800 | heldout | 0 | 1 | no | 0.05, 0.00, 0.00 | [rgb_novae__three_heldout.mp4](videos/stage_a_all_arms/rgb_novae__three_heldout.mp4) |
| rgb_novae | 3 objects, novel identities | 1001 | 804 | novel | 1 | 1 | no | 1.25, 0.95, 0.30 | [rgb_novae__three_novel.mp4](videos/stage_a_all_arms/rgb_novae__three_novel.mp4) |
| rgb_novae | 2 objects, seen | 1001 | 806 | seen | 0 | 1 | no | 0.50, 0.35, 0.15 | [rgb_novae__two.mp4](videos/stage_a_all_arms/rgb_novae__two.mp4) |
| rgb_novae | 4 objects, seen | 1000 | 801 | seen | 0 | 2 | no | 0.40, 1.00, 0.35 | [rgb_novae__four.mp4](videos/stage_a_all_arms/rgb_novae__four.mp4) |
| kv | 3 objects, seen identities | 1001 | 801 | seen | 1 | 2 | no | 0.55, 0.60, 1.35 | [kv__three_seen.mp4](videos/stage_a_all_arms/kv__three_seen.mp4) |
| kv | 3 objects, heldout identities | 1000 | 801 | heldout | 0 | 1 | no | 0.30, 0.00, 0.35 | [kv__three_heldout.mp4](videos/stage_a_all_arms/kv__three_heldout.mp4) |
| kv | 3 objects, novel identities | 1002 | 801 | novel | 0 | 1 | no | 0.75, 0.00, 0.00 | [kv__three_novel.mp4](videos/stage_a_all_arms/kv__three_novel.mp4) |
| kv | 2 objects, seen | 1000 | 802 | seen | 0 | 1 | no | 0.35, 0.25, 0.40 | [kv__two.mp4](videos/stage_a_all_arms/kv__two.mp4) |
| kv | 4 objects, seen | 1000 | 801 | seen | 0 | 2 | no | 0.45, 0.45, 0.35 | [kv__four.mp4](videos/stage_a_all_arms/kv__four.mp4) |
| kv_adaln | 3 objects, seen identities | 1000 | 810 | seen | 1 | 2 | no | 0.95, 0.75, 1.10 | [kv_adaln__three_seen.mp4](videos/stage_a_all_arms/kv_adaln__three_seen.mp4) |
| kv_adaln | 3 objects, heldout identities | 1000 | 800 | heldout | 0 | 1 | no | 0.50, 0.35, 0.80 | [kv_adaln__three_heldout.mp4](videos/stage_a_all_arms/kv_adaln__three_heldout.mp4) |
| kv_adaln | 3 objects, novel identities | 1001 | 801 | novel | 0 | 1 | no | 0.35, 0.25, 0.10 | [kv_adaln__three_novel.mp4](videos/stage_a_all_arms/kv_adaln__three_novel.mp4) |
| kv_adaln | 2 objects, seen | 1002 | 800 | seen | 0 | 1 | no | 0.80, 0.20, 0.30 | [kv_adaln__two.mp4](videos/stage_a_all_arms/kv_adaln__two.mp4) |
| kv_adaln | 4 objects, seen | 1001 | 805 | seen | 0 | 1 | no | 0.15, 0.20, 0.25 | [kv_adaln__four.mp4](videos/stage_a_all_arms/kv_adaln__four.mp4) |
| kv_tokens | 3 objects, seen identities | 1000 | 800 | seen | 1 | 2 | no | 1.30, 1.05, 1.40 | [kv_tokens__three_seen.mp4](videos/stage_a_all_arms/kv_tokens__three_seen.mp4) |
| kv_tokens | 3 objects, heldout identities | 1002 | 801 | heldout | 0 | 1 | no | 0.20, 0.05, 0.15 | [kv_tokens__three_heldout.mp4](videos/stage_a_all_arms/kv_tokens__three_heldout.mp4) |
| kv_tokens | 3 objects, novel identities | 1002 | 800 | novel | 0 | 1 | no | 0.75, 0.00, 0.00 | [kv_tokens__three_novel.mp4](videos/stage_a_all_arms/kv_tokens__three_novel.mp4) |
| kv_tokens | 2 objects, seen | 1002 | 800 | seen | 0 | 1 | no | 0.45, 0.25, 0.40 | [kv_tokens__two.mp4](videos/stage_a_all_arms/kv_tokens__two.mp4) |
| kv_tokens | 4 objects, seen | 1002 | 801 | seen | 0 | 1 | no | 0.25, 0.35, 0.25 | [kv_tokens__four.mp4](videos/stage_a_all_arms/kv_tokens__four.mp4) |
| scratch_kv | 3 objects, seen identities | 1002 | 800 | seen | 0 | 2 | no | 0.30, 0.55, 0.50 | [scratch_kv__three_seen.mp4](videos/stage_a_all_arms/scratch_kv__three_seen.mp4) |
| scratch_kv | 3 objects, heldout identities | 1002 | 803 | heldout | 0 | 1 | no | 0.10, 0.00, 0.05 | [scratch_kv__three_heldout.mp4](videos/stage_a_all_arms/scratch_kv__three_heldout.mp4) |
| scratch_kv | 3 objects, novel identities | 1001 | 800 | novel | 0 | 1 | no | 0.05, 0.25, 0.60 | [scratch_kv__three_novel.mp4](videos/stage_a_all_arms/scratch_kv__three_novel.mp4) |
| scratch_kv | 2 objects, seen | 1002 | 800 | seen | 0 | 1 | no | 0.15, 0.30, 0.20 | [scratch_kv__two.mp4](videos/stage_a_all_arms/scratch_kv__two.mp4) |
| scratch_kv | 4 objects, seen | 1001 | 800 | seen | 0 | 1 | no | 0.05, 0.05, 0.85 | [scratch_kv__four.mp4](videos/stage_a_all_arms/scratch_kv__four.mp4) |

## Chaining study, atomic clips (`videos/chaining_atomic_clips/`, `AF`)

Compare with the `rgb` rows of the next section: the same oracle executions,
kept whole. The difference to look for is after the first transfer.

| arm | setting | training seed | scene seed | meshes | transfers | lifted | success | row mean transfers (per seed) | video |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| rgb | 3 objects, seen identities | 1002 | 801 | seen | 0 | 1 | no | 0.55, 0.45, 0.45 | [rgb__three_seen.mp4](videos/chaining_atomic_clips/rgb__three_seen.mp4) |
| rgb | 3 objects, heldout identities | 1002 | 801 | heldout | 0 | 1 | no | 0.15, 0.00, 0.10 | [rgb__three_heldout.mp4](videos/chaining_atomic_clips/rgb__three_heldout.mp4) |
| rgb | 3 objects, novel identities | 1000 | 801 | novel | 1 | 2 | no | 1.25, 1.00, 1.30 | [rgb__three_novel.mp4](videos/chaining_atomic_clips/rgb__three_novel.mp4) |
| rgb | 2 objects, seen | 1000 | 800 | seen | 0 | 1 | no | 0.50, 0.55, 0.35 | [rgb__two.mp4](videos/chaining_atomic_clips/rgb__two.mp4) |
| rgb | 4 objects, seen | 1002 | 803 | seen | 0 | 2 | no | 0.60, 0.30, 0.35 | [rgb__four.mp4](videos/chaining_atomic_clips/rgb__four.mp4) |

## Full runs: every ACT arm (`videos/full_runs_all_arms/`, `CF`)

`rgb` is the chaining study's full-run policy; `rgb_cont` is the budget
control of Stage A2; `kv`, `kv_adaln`, `kv_tokens` and `scratch_kv` are the
conditioned arms (§4.9); the rest are the baseline ablations (short chunk,
history, head-camera dropout, image shift, chunk-relative targets; §4.7–4.9).

| arm | setting | training seed | scene seed | meshes | transfers | lifted | success | row mean transfers (per seed) | video |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| rgb | 3 objects, seen identities | 1000 | 803 | seen | 1 | 2 | no | 0.80, 0.40, 0.90 | [rgb__three_seen.mp4](videos/full_runs_all_arms/rgb__three_seen.mp4) |
| rgb | 3 objects, heldout identities | 1001 | 801 | heldout | 0 | 1 | no | 0.00, 0.05, 0.10 | [rgb__three_heldout.mp4](videos/full_runs_all_arms/rgb__three_heldout.mp4) |
| rgb | 3 objects, novel identities | 1001 | 801 | novel | 0 | 1 | no | 0.10, 0.30, 1.85 | [rgb__three_novel.mp4](videos/full_runs_all_arms/rgb__three_novel.mp4) |
| rgb | 2 objects, seen | 1002 | 804 | seen | 1 | 1 | no | 0.70, 1.05, 0.90 | [rgb__two.mp4](videos/full_runs_all_arms/rgb__two.mp4) |
| rgb | 4 objects, seen | 1002 | 807 | seen | 1 | 2 | no | 0.45, 0.75, 0.55 | [rgb__four.mp4](videos/full_runs_all_arms/rgb__four.mp4) |
| rgb_cont | 3 objects, seen identities | 1000 | 819 | seen | 1 | 2 | no | 1.45, 1.20, 1.50 | [rgb_cont__three_seen.mp4](videos/full_runs_all_arms/rgb_cont__three_seen.mp4) |
| rgb_cont | 3 objects, heldout identities | 1000 | 800 | heldout | 0 | 1 | no | 0.30, 0.45, 0.15 | [rgb_cont__three_heldout.mp4](videos/full_runs_all_arms/rgb_cont__three_heldout.mp4) |
| rgb_cont | 3 objects, novel identities | 1001 | 805 | novel | 1 | 2 | no | 0.05, 0.60, 2.55 | [rgb_cont__three_novel.mp4](videos/full_runs_all_arms/rgb_cont__three_novel.mp4) |
| rgb_cont | 2 objects, seen | 1002 | 804 | seen | 1 | 2 | no | 0.75, 1.05, 1.00 | [rgb_cont__two.mp4](videos/full_runs_all_arms/rgb_cont__two.mp4) |
| rgb_cont | 4 objects, seen | 1001 | 801 | seen | 1 | 2 | no | 0.85, 0.60, 0.55 | [rgb_cont__four.mp4](videos/full_runs_all_arms/rgb_cont__four.mp4) |
| rgb_short | 3 objects, seen identities | 1000 | 803 | seen | 1 | 2 | no | 0.95, 1.10, 0.75 | [rgb_short__three_seen.mp4](videos/full_runs_all_arms/rgb_short__three_seen.mp4) |
| rgb_short | 3 objects, heldout identities | 1001 | 800 | heldout | 0 | 1 | no | 0.45, 0.25, 0.05 | [rgb_short__three_heldout.mp4](videos/full_runs_all_arms/rgb_short__three_heldout.mp4) |
| rgb_short | 3 objects, novel identities | 1002 | 800 | novel | 0 | 0 | no | 0.05, 0.00, 0.00 | [rgb_short__three_novel.mp4](videos/full_runs_all_arms/rgb_short__three_novel.mp4) |
| rgb_short | 2 objects, seen | 1002 | 804 | seen | 1 | 1 | no | 0.85, 0.45, 0.70 | [rgb_short__two.mp4](videos/full_runs_all_arms/rgb_short__two.mp4) |
| rgb_short | 4 objects, seen | 1002 | 802 | seen | 1 | 1 | no | 0.55, 1.20, 0.70 | [rgb_short__four.mp4](videos/full_runs_all_arms/rgb_short__four.mp4) |
| rgb_hist | 3 objects, seen identities | 1002 | 803 | seen | 1 | 2 | no | 0.80, 0.45, 0.65 | [rgb_hist__three_seen.mp4](videos/full_runs_all_arms/rgb_hist__three_seen.mp4) |
| rgb_hist | 3 objects, heldout identities | 1001 | 800 | heldout | 0 | 1 | no | 0.05, 0.10, 0.10 | [rgb_hist__three_heldout.mp4](videos/full_runs_all_arms/rgb_hist__three_heldout.mp4) |
| rgb_hist | 3 objects, novel identities | 1002 | 800 | novel | 0 | 0 | no | 0.00, 0.10, 0.00 | [rgb_hist__three_novel.mp4](videos/full_runs_all_arms/rgb_hist__three_novel.mp4) |
| rgb_hist | 2 objects, seen | 1001 | 804 | seen | 0 | 1 | no | 0.55, 0.40, 0.25 | [rgb_hist__two.mp4](videos/full_runs_all_arms/rgb_hist__two.mp4) |
| rgb_hist | 4 objects, seen | 1002 | 801 | seen | 1 | 2 | no | 0.45, 0.80, 0.65 | [rgb_hist__four.mp4](videos/full_runs_all_arms/rgb_hist__four.mp4) |
| rgb_headdrop | 3 objects, seen identities | 1000 | 800 | seen | 0 | 2 | no | 0.45, 0.20, 0.55 | [rgb_headdrop__three_seen.mp4](videos/full_runs_all_arms/rgb_headdrop__three_seen.mp4) |
| rgb_headdrop | 3 objects, heldout identities | 1001 | 800 | heldout | 0 | 1 | no | 0.05, 0.15, 0.20 | [rgb_headdrop__three_heldout.mp4](videos/full_runs_all_arms/rgb_headdrop__three_heldout.mp4) |
| rgb_headdrop | 3 objects, novel identities | 1001 | 803 | novel | 1 | 2 | no | 0.10, 0.65, 1.30 | [rgb_headdrop__three_novel.mp4](videos/full_runs_all_arms/rgb_headdrop__three_novel.mp4) |
| rgb_headdrop | 2 objects, seen | 1000 | 807 | seen | 1 | 1 | no | 0.75, 0.70, 1.05 | [rgb_headdrop__two.mp4](videos/full_runs_all_arms/rgb_headdrop__two.mp4) |
| rgb_headdrop | 4 objects, seen | 1002 | 800 | seen | 0 | 2 | no | 0.55, 0.10, 0.30 | [rgb_headdrop__four.mp4](videos/full_runs_all_arms/rgb_headdrop__four.mp4) |
| rgb_shift | 3 objects, seen identities | 1002 | 800 | seen | 1 | 2 | no | 0.70, 0.50, 0.55 | [rgb_shift__three_seen.mp4](videos/full_runs_all_arms/rgb_shift__three_seen.mp4) |
| rgb_shift | 3 objects, heldout identities | 1001 | 802 | heldout | 0 | 1 | no | 0.00, 0.15, 0.20 | [rgb_shift__three_heldout.mp4](videos/full_runs_all_arms/rgb_shift__three_heldout.mp4) |
| rgb_shift | 3 objects, novel identities | 1002 | 802 | novel | 1 | 2 | no | 1.50, 0.90, 1.45 | [rgb_shift__three_novel.mp4](videos/full_runs_all_arms/rgb_shift__three_novel.mp4) |
| rgb_shift | 2 objects, seen | 1000 | 803 | seen | 1 | 1 | no | 0.65, 1.20, 0.50 | [rgb_shift__two.mp4](videos/full_runs_all_arms/rgb_shift__two.mp4) |
| rgb_shift | 4 objects, seen | 1001 | 801 | seen | 1 | 2 | no | 0.20, 0.50, 0.60 | [rgb_shift__four.mp4](videos/full_runs_all_arms/rgb_shift__four.mp4) |
| rgb_rel | 3 objects, seen identities | 1001 | 805 | seen | 1 | 2 | no | 0.40, 0.65, 1.15 | [rgb_rel__three_seen.mp4](videos/full_runs_all_arms/rgb_rel__three_seen.mp4) |
| rgb_rel | 3 objects, heldout identities | 1002 | 800 | heldout | 0 | 1 | no | 0.15, 0.30, 0.20 | [rgb_rel__three_heldout.mp4](videos/full_runs_all_arms/rgb_rel__three_heldout.mp4) |
| rgb_rel | 3 objects, novel identities | 1002 | 801 | novel | 0 | 1 | no | 0.60, 0.10, 0.15 | [rgb_rel__three_novel.mp4](videos/full_runs_all_arms/rgb_rel__three_novel.mp4) |
| rgb_rel | 2 objects, seen | 1001 | 803 | seen | 1 | 1 | no | 0.65, 0.55, 0.45 | [rgb_rel__two.mp4](videos/full_runs_all_arms/rgb_rel__two.mp4) |
| rgb_rel | 4 objects, seen | 1002 | 803 | seen | 1 | 2 | no | 0.35, 0.55, 0.50 | [rgb_rel__four.mp4](videos/full_runs_all_arms/rgb_rel__four.mp4) |
| rgb_rel_short | 3 objects, seen identities | 1001 | 802 | seen | 1 | 2 | no | 0.75, 0.80, 0.90 | [rgb_rel_short__three_seen.mp4](videos/full_runs_all_arms/rgb_rel_short__three_seen.mp4) |
| rgb_rel_short | 3 objects, heldout identities | 1002 | 800 | heldout | 0 | 1 | no | 0.25, 0.35, 0.30 | [rgb_rel_short__three_heldout.mp4](videos/full_runs_all_arms/rgb_rel_short__three_heldout.mp4) |
| rgb_rel_short | 3 objects, novel identities | 1001 | 804 | novel | 0 | 1 | no | 0.10, 0.05, 0.00 | [rgb_rel_short__three_novel.mp4](videos/full_runs_all_arms/rgb_rel_short__three_novel.mp4) |
| rgb_rel_short | 2 objects, seen | 1002 | 804 | seen | 1 | 1 | no | 0.65, 0.80, 0.75 | [rgb_rel_short__two.mp4](videos/full_runs_all_arms/rgb_rel_short__two.mp4) |
| rgb_rel_short | 4 objects, seen | 1002 | 800 | seen | 0 | 1 | no | 0.50, 0.95, 0.50 | [rgb_rel_short__four.mp4](videos/full_runs_all_arms/rgb_rel_short__four.mp4) |
| kv | 3 objects, seen identities | 1001 | 801 | seen | 1 | 2 | no | 1.15, 1.25, 1.80 | [kv__three_seen.mp4](videos/full_runs_all_arms/kv__three_seen.mp4) |
| kv | 3 objects, heldout identities | 1002 | 800 | heldout | 0 | 1 | no | 0.20, 0.45, 0.30 | [kv__three_heldout.mp4](videos/full_runs_all_arms/kv__three_heldout.mp4) |
| kv | 3 objects, novel identities | 1001 | 802 | novel | 1 | 2 | no | 0.05, 0.80, 2.35 | [kv__three_novel.mp4](videos/full_runs_all_arms/kv__three_novel.mp4) |
| kv | 2 objects, seen | 1000 | 800 | seen | 1 | 1 | no | 1.00, 0.80, 1.00 | [kv__two.mp4](videos/full_runs_all_arms/kv__two.mp4) |
| kv | 4 objects, seen | 1001 | 801 | seen | 1 | 2 | no | 0.20, 0.65, 0.65 | [kv__four.mp4](videos/full_runs_all_arms/kv__four.mp4) |
| kv_adaln | 3 objects, seen identities | 1002 | 802 | seen | 2 | 2 | no | 1.65, 1.10, 1.25 | [kv_adaln__three_seen.mp4](videos/full_runs_all_arms/kv_adaln__three_seen.mp4) |
| kv_adaln | 3 objects, heldout identities | 1000 | 802 | heldout | 1 | 2 | no | 0.55, 0.45, 0.80 | [kv_adaln__three_heldout.mp4](videos/full_runs_all_arms/kv_adaln__three_heldout.mp4) |
| kv_adaln | 3 objects, novel identities | 1001 | 804 | novel | 0 | 1 | no | 0.15, 0.35, 1.80 | [kv_adaln__three_novel.mp4](videos/full_runs_all_arms/kv_adaln__three_novel.mp4) |
| kv_adaln | 2 objects, seen | 1002 | 818 | seen | 1 | 2 | no | 1.10, 0.35, 0.75 | [kv_adaln__two.mp4](videos/full_runs_all_arms/kv_adaln__two.mp4) |
| kv_adaln | 4 objects, seen | 1001 | 801 | seen | 0 | 2 | no | 0.35, 0.40, 0.60 | [kv_adaln__four.mp4](videos/full_runs_all_arms/kv_adaln__four.mp4) |
| kv_tokens | 3 objects, seen identities | 1001 | 810 | seen | 2 | 3 | no | 1.55, 1.60, 1.70 | [kv_tokens__three_seen.mp4](videos/full_runs_all_arms/kv_tokens__three_seen.mp4) |
| kv_tokens | 3 objects, heldout identities | 1000 | 806 | heldout | 0 | 1 | no | 0.30, 0.15, 0.30 | [kv_tokens__three_heldout.mp4](videos/full_runs_all_arms/kv_tokens__three_heldout.mp4) |
| kv_tokens | 3 objects, novel identities | 1001 | 800 | novel | 0 | 1 | no | 0.10, 0.30, 1.85 | [kv_tokens__three_novel.mp4](videos/full_runs_all_arms/kv_tokens__three_novel.mp4) |
| kv_tokens | 2 objects, seen | 1000 | 803 | seen | 1 | 1 | no | 1.00, 0.85, 1.00 | [kv_tokens__two.mp4](videos/full_runs_all_arms/kv_tokens__two.mp4) |
| kv_tokens | 4 objects, seen | 1001 | 800 | seen | 1 | 2 | no | 0.65, 0.60, 0.25 | [kv_tokens__four.mp4](videos/full_runs_all_arms/kv_tokens__four.mp4) |
| scratch_kv | 3 objects, seen identities | 1002 | 802 | seen | 0 | 2 | no | 0.60, 0.35, 0.40 | [scratch_kv__three_seen.mp4](videos/full_runs_all_arms/scratch_kv__three_seen.mp4) |
| scratch_kv | 3 objects, heldout identities | 1000 | 800 | heldout | 0 | 1 | no | 0.10, 0.15, 0.00 | [scratch_kv__three_heldout.mp4](videos/full_runs_all_arms/scratch_kv__three_heldout.mp4) |
| scratch_kv | 3 objects, novel identities | 1002 | 803 | novel | 1 | 2 | no | 0.20, 1.10, 1.05 | [scratch_kv__three_novel.mp4](videos/full_runs_all_arms/scratch_kv__three_novel.mp4) |
| scratch_kv | 2 objects, seen | 1002 | 804 | seen | 1 | 1 | no | 0.70, 0.85, 0.70 | [scratch_kv__two.mp4](videos/full_runs_all_arms/scratch_kv__two.mp4) |
| scratch_kv | 4 objects, seen | 1000 | 801 | seen | 0 | 2 | no | 0.45, 0.50, 0.25 | [scratch_kv__four.mp4](videos/full_runs_all_arms/scratch_kv__four.mp4) |

## VLA stage 1 (`videos/vla_stage1/`, `PF`, `PJ`, `SF`, `SJ`, `GF`, `GJ`)

RGB only, seen identities, three objects.

| backbone | regime | setting | training seed | scene seed | meshes | transfers | lifted | success | row mean transfers (per seed) | video |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| pi0.5 | absolute EE | 3 objects, all identities | 1001 | 806 | seen | 0 | 1 | no | 0.00, 0.00, 0.00 | [rgb__three_all.mp4](videos/vla_stage1/pi05_abs_ee/rgb__three_all.mp4) |
| pi0.5 | absolute joint | 3 objects, all identities | 1001 | 803 | seen | 0 | 0 | no | 0.00, 0.00, 0.00 | [rgb__three_all.mp4](videos/vla_stage1/pi05_abs_joint/rgb__three_all.mp4) |
| SmolVLA | absolute EE | 3 objects, all identities | 1002 | 805 | seen | 0 | 1 | no | 0.10, 0.00, 0.00 | [rgb__three_all.mp4](videos/vla_stage1/smolvla_abs_ee/rgb__three_all.mp4) |
| SmolVLA | absolute joint | 3 objects, all identities | 1001 | 800 | seen | 0 | 0 | no | 0.00, 0.00, 0.00 | [rgb__three_all.mp4](videos/vla_stage1/smolvla_abs_joint/rgb__three_all.mp4) |
| GR00T N1.7 | absolute EE | 3 objects, all identities | 1001 | 801 | seen | 1 | 2 | no | 0.85, 0.55, 0.15 | [rgb__three_all.mp4](videos/vla_stage1/groot_abs_ee/rgb__three_all.mp4) |
| GR00T N1.7 | absolute joint | 3 objects, all identities | 1002 | 802 | seen | 0 | 1 | no | 0.25, 0.35, 0.30 | [rgb__three_all.mp4](videos/vla_stage1/groot_abs_joint/rgb__three_all.mp4) |

## GR00T stage 2 (`videos/groot_stage2/`, `GF`)

Absolute EE, full-run corpus. `rgb` is stage 1 (the same video as
`vla_stage1/groot_abs_ee/`); `rgb_cont` is the budget control, trained 8 k more
steps; `kv` and `kv_adaln` are the conditioned arms, trained the same 8 k from
the same checkpoint. `kv_tokens` is unsupported on GR00T (§4.11). What to look
for: the conditioned arms grasp and carry reliably, and most of what they lose
is a placed object knocked off later; the control drops objects on the way.

The representative episode is not a success even for `kv_adaln`, whose mean
is two transfers of three: that is what the rule is for.

| arm | setting | training seed | scene seed | meshes | transfers | lifted | success | row mean transfers (per seed) | video |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| rgb | 3 objects, seen identities | 1001 | 801 | seen | 1 | 2 | no | 0.85, 0.55, 0.15 | [rgb__three_all.mp4](videos/vla_stage1/groot_abs_ee/rgb__three_all.mp4) |
| rgb_cont | 3 objects, seen identities | 1002 | 800 | seen | 1 | 2 | no | 1.10, 0.75, 1.05 | [rgb_cont__three_seen.mp4](videos/groot_stage2/rgb_cont__three_seen.mp4) |
| rgb_cont | 3 objects, heldout identities | 1000 | 803 | heldout | 0 | 1 | no | 0.35, 0.35, 0.25 | [rgb_cont__three_heldout.mp4](videos/groot_stage2/rgb_cont__three_heldout.mp4) |
| kv | 3 objects, seen identities | 1002 | 803 | seen | 2 | 2 | no | 1.55, 2.00, 1.70 | [kv__three_seen.mp4](videos/groot_stage2/kv__three_seen.mp4) |
| kv | 3 objects, heldout identities | 1002 | 803 | heldout | 1 | 2 | no | 0.50, 1.05, 0.85 | [kv__three_heldout.mp4](videos/groot_stage2/kv__three_heldout.mp4) |
| kv_adaln | 3 objects, seen identities | 1001 | 800 | seen | 2 | 3 | no | 2.00, 2.00, 2.10 | [kv_adaln__three_seen.mp4](videos/groot_stage2/kv_adaln__three_seen.mp4) |
| kv_adaln | 3 objects, heldout identities | 1000 | 819 | heldout | 1 | 3 | no | 0.65, 1.10, 0.50 | [kv_adaln__three_heldout.mp4](videos/groot_stage2/kv_adaln__three_heldout.mp4) |

## Regenerating

```bash
source ~/octvla/env-leftmost.sh
declare -A DEST=([F]=stage_a_all_arms [AF]=chaining_atomic_clips [CF]=full_runs_all_arms
                 [PF]=vla_stage1/pi05_abs_ee [PJ]=vla_stage1/pi05_abs_joint
                 [SF]=vla_stage1/smolvla_abs_ee [SJ]=vla_stage1/smolvla_abs_joint
                 [GF]=vla_stage1/groot_abs_ee [GJ]=vla_stage1/groot_abs_joint)
# GR00T stage 2 shares prefix GF with its stage 1: select it into its own folder
# and drop the rgb row's copy, which is vla_stage1/groot_abs_ee/rgb__three_all.mp4.
#   select_rollout_videos.py ... --prefix GF --dest docs/videos/groot_stage2
for p in "${!DEST[@]}"; do
  PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" scripts/select_rollout_videos.py \
    "$OCTVLA_OUTPUT_ROOT/eval" --prefix $p --dest docs/videos/${DEST[$p]} --table /tmp/$p.md
done
```

Home allocates 32 MB per file, so these 112 videos take about 3.6 GB of the
home quota (roughly 45 MB of data).
