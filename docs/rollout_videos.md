# Representative rollout videos

One video per row that `research_questions.md` reports, copied into
`docs/videos/`. Regenerate with `scripts/select_rollout_videos.py` (the
commands are at the bottom); every rollout's full set of episodes stays on the
vault under `$HPCVAULT/octvla-rollout-videos/final/`.

**"Representative" is a rule, not a choice by eye.** Picking the best
episode is how this project was misled before (a 6/20 headline that scored
0/20 on two more seeds). So for each row:
1. take the **median training seed** by mean transfers (the lower of the two
   middle seeds on a tie), so a lucky seed is not the face of the arm;
2. within it, the episode whose transfers are **closest to that seed's mean**,
   then whose lifts are closest to its mean lifts, then the lowest scene seed.

A row averaging 0.4 transfers therefore shows an episode with no transfer,
which is what a viewer should expect to see. The `transfers` and `success`
columns describe the chosen episode; the last numeric column is the row's
mean transfers on each of the three training seeds.

Settings: 3 objects run 600 steps, pinned to seen (meshes 1–4), held-out
(0, 6) or novel (5) identities; 2 and 4 objects get 200 steps per object,
seen identities, the two-object scene on the nested layout.

## Stage A: every arm, atomic clips, absolute EE (`F`)

The matrix behind Q1–Q3 (`research_questions.md` §4.1–4.3). `rgb_cont` is the
budget control every conditioned arm is paired against.

| arm | setting | training seed | scene seed | meshes | transfers | lifted | success | row mean transfers (per seed) | video |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| rgb | 3 objects, seen identities | 1001 | 804 | seen | 0 | 2 | no | 0.90, 0.50, 0.45 | [rgb__three_seen.mp4](videos/F/rgb__three_seen.mp4) |
| rgb | 3 objects, heldout identities | 1002 | 801 | heldout | 0 | 1 | no | 0.10, 0.00, 0.00 | [rgb__three_heldout.mp4](videos/F/rgb__three_heldout.mp4) |
| rgb | 3 objects, novel identities | 1002 | 800 | novel | 0 | 1 | no | 0.55, 0.15, 0.15 | [rgb__three_novel.mp4](videos/F/rgb__three_novel.mp4) |
| rgb | 2 objects, seen | 1001 | 802 | seen | 0 | 1 | no | 0.20, 0.20, 0.25 | [rgb__two.mp4](videos/F/rgb__two.mp4) |
| rgb | 4 objects, seen | 1000 | 800 | seen | 0 | 1 | no | 0.40, 0.50, 0.30 | [rgb__four.mp4](videos/F/rgb__four.mp4) |
| rgb_cont | 3 objects, seen identities | 1002 | 804 | seen | 1 | 2 | no | 1.40, 0.50, 0.95 | [rgb_cont__three_seen.mp4](videos/F/rgb_cont__three_seen.mp4) |
| rgb_cont | 3 objects, heldout identities | 1000 | 801 | heldout | 0 | 1 | no | 0.20, 0.05, 0.25 | [rgb_cont__three_heldout.mp4](videos/F/rgb_cont__three_heldout.mp4) |
| rgb_cont | 3 objects, novel identities | 1001 | 800 | novel | 0 | 1 | no | 0.75, 0.15, 0.00 | [rgb_cont__three_novel.mp4](videos/F/rgb_cont__three_novel.mp4) |
| rgb_cont | 2 objects, seen | 1000 | 802 | seen | 0 | 1 | no | 0.45, 0.15, 0.55 | [rgb_cont__two.mp4](videos/F/rgb_cont__two.mp4) |
| rgb_cont | 4 objects, seen | 1001 | 800 | seen | 0 | 2 | no | 0.45, 0.35, 0.15 | [rgb_cont__four.mp4](videos/F/rgb_cont__four.mp4) |
| rgb_novae | 3 objects, seen identities | 1000 | 801 | seen | 1 | 2 | no | 0.65, 1.40, 0.45 | [rgb_novae__three_seen.mp4](videos/F/rgb_novae__three_seen.mp4) |
| rgb_novae | 3 objects, heldout identities | 1002 | 800 | heldout | 0 | 1 | no | 0.05, 0.00, 0.00 | [rgb_novae__three_heldout.mp4](videos/F/rgb_novae__three_heldout.mp4) |
| rgb_novae | 3 objects, novel identities | 1001 | 804 | novel | 1 | 1 | no | 1.25, 0.95, 0.30 | [rgb_novae__three_novel.mp4](videos/F/rgb_novae__three_novel.mp4) |
| rgb_novae | 2 objects, seen | 1001 | 806 | seen | 0 | 1 | no | 0.50, 0.35, 0.15 | [rgb_novae__two.mp4](videos/F/rgb_novae__two.mp4) |
| rgb_novae | 4 objects, seen | 1000 | 801 | seen | 0 | 2 | no | 0.40, 1.00, 0.35 | [rgb_novae__four.mp4](videos/F/rgb_novae__four.mp4) |
| kv | 3 objects, seen identities | 1001 | 801 | seen | 1 | 2 | no | 0.55, 0.60, 1.35 | [kv__three_seen.mp4](videos/F/kv__three_seen.mp4) |
| kv | 3 objects, heldout identities | 1000 | 801 | heldout | 0 | 1 | no | 0.30, 0.00, 0.35 | [kv__three_heldout.mp4](videos/F/kv__three_heldout.mp4) |
| kv | 3 objects, novel identities | 1002 | 801 | novel | 0 | 1 | no | 0.75, 0.00, 0.00 | [kv__three_novel.mp4](videos/F/kv__three_novel.mp4) |
| kv | 2 objects, seen | 1000 | 802 | seen | 0 | 1 | no | 0.35, 0.25, 0.40 | [kv__two.mp4](videos/F/kv__two.mp4) |
| kv | 4 objects, seen | 1000 | 801 | seen | 0 | 2 | no | 0.45, 0.45, 0.35 | [kv__four.mp4](videos/F/kv__four.mp4) |
| kv_adaln | 3 objects, seen identities | 1000 | 810 | seen | 1 | 2 | no | 0.95, 0.75, 1.10 | [kv_adaln__three_seen.mp4](videos/F/kv_adaln__three_seen.mp4) |
| kv_adaln | 3 objects, heldout identities | 1000 | 800 | heldout | 0 | 1 | no | 0.50, 0.35, 0.80 | [kv_adaln__three_heldout.mp4](videos/F/kv_adaln__three_heldout.mp4) |
| kv_adaln | 3 objects, novel identities | 1001 | 801 | novel | 0 | 1 | no | 0.35, 0.25, 0.10 | [kv_adaln__three_novel.mp4](videos/F/kv_adaln__three_novel.mp4) |
| kv_adaln | 2 objects, seen | 1002 | 800 | seen | 0 | 1 | no | 0.80, 0.20, 0.30 | [kv_adaln__two.mp4](videos/F/kv_adaln__two.mp4) |
| kv_adaln | 4 objects, seen | 1001 | 805 | seen | 0 | 1 | no | 0.15, 0.20, 0.25 | [kv_adaln__four.mp4](videos/F/kv_adaln__four.mp4) |
| kv_tokens | 3 objects, seen identities | 1000 | 800 | seen | 1 | 2 | no | 1.30, 1.05, 1.40 | [kv_tokens__three_seen.mp4](videos/F/kv_tokens__three_seen.mp4) |
| kv_tokens | 3 objects, heldout identities | 1002 | 801 | heldout | 0 | 1 | no | 0.20, 0.05, 0.15 | [kv_tokens__three_heldout.mp4](videos/F/kv_tokens__three_heldout.mp4) |
| kv_tokens | 3 objects, novel identities | 1002 | 800 | novel | 0 | 1 | no | 0.75, 0.00, 0.00 | [kv_tokens__three_novel.mp4](videos/F/kv_tokens__three_novel.mp4) |
| kv_tokens | 2 objects, seen | 1002 | 800 | seen | 0 | 1 | no | 0.45, 0.25, 0.40 | [kv_tokens__two.mp4](videos/F/kv_tokens__two.mp4) |
| kv_tokens | 4 objects, seen | 1002 | 801 | seen | 0 | 1 | no | 0.25, 0.35, 0.25 | [kv_tokens__four.mp4](videos/F/kv_tokens__four.mp4) |
| scratch_kv | 3 objects, seen identities | 1002 | 800 | seen | 0 | 2 | no | 0.30, 0.55, 0.50 | [scratch_kv__three_seen.mp4](videos/F/scratch_kv__three_seen.mp4) |
| scratch_kv | 3 objects, heldout identities | 1002 | 803 | heldout | 0 | 1 | no | 0.10, 0.00, 0.05 | [scratch_kv__three_heldout.mp4](videos/F/scratch_kv__three_heldout.mp4) |
| scratch_kv | 3 objects, novel identities | 1001 | 800 | novel | 0 | 1 | no | 0.05, 0.25, 0.60 | [scratch_kv__three_novel.mp4](videos/F/scratch_kv__three_novel.mp4) |
| scratch_kv | 2 objects, seen | 1002 | 800 | seen | 0 | 1 | no | 0.15, 0.30, 0.20 | [scratch_kv__two.mp4](videos/F/scratch_kv__two.mp4) |
| scratch_kv | 4 objects, seen | 1001 | 800 | seen | 0 | 1 | no | 0.05, 0.05, 0.85 | [scratch_kv__four.mp4](videos/F/scratch_kv__four.mp4) |

## The chaining study: atomic clips (`AF`) vs full runs (`CF`)

Plain RGB ACT on the paired corpus, trained on the same oracle executions cut
into clips or kept whole (§4.6). The difference to look for is after the
first transfer: the atomic policy rarely starts a second, the full-run policy
often does.

### Atomic clips (`AF-rgb`)

| arm | setting | training seed | scene seed | meshes | transfers | lifted | success | row mean transfers (per seed) | video |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| rgb | 3 objects, seen identities | 1002 | 801 | seen | 0 | 1 | no | 0.55, 0.45, 0.45 | [rgb__three_seen.mp4](videos/AF/rgb__three_seen.mp4) |
| rgb | 3 objects, heldout identities | 1002 | 801 | heldout | 0 | 1 | no | 0.15, 0.00, 0.10 | [rgb__three_heldout.mp4](videos/AF/rgb__three_heldout.mp4) |
| rgb | 3 objects, novel identities | 1000 | 801 | novel | 1 | 2 | no | 1.25, 1.00, 1.30 | [rgb__three_novel.mp4](videos/AF/rgb__three_novel.mp4) |
| rgb | 2 objects, seen | 1000 | 800 | seen | 0 | 1 | no | 0.50, 0.55, 0.35 | [rgb__two.mp4](videos/AF/rgb__two.mp4) |
| rgb | 4 objects, seen | 1002 | 803 | seen | 0 | 2 | no | 0.60, 0.30, 0.35 | [rgb__four.mp4](videos/AF/rgb__four.mp4) |

### Full runs (`CF-rgb`)

| arm | setting | training seed | scene seed | meshes | transfers | lifted | success | row mean transfers (per seed) | video |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| rgb | 3 objects, seen identities | 1000 | 803 | seen | 1 | 2 | no | 0.80, 0.40, 0.90 | [rgb__three_seen.mp4](videos/CF/rgb__three_seen.mp4) |
| rgb | 3 objects, heldout identities | 1001 | 801 | heldout | 0 | 1 | no | 0.00, 0.05, 0.10 | [rgb__three_heldout.mp4](videos/CF/rgb__three_heldout.mp4) |
| rgb | 3 objects, novel identities | 1001 | 801 | novel | 0 | 1 | no | 0.10, 0.30, 1.85 | [rgb__three_novel.mp4](videos/CF/rgb__three_novel.mp4) |
| rgb | 2 objects, seen | 1002 | 804 | seen | 1 | 1 | no | 0.70, 1.05, 0.90 | [rgb__two.mp4](videos/CF/rgb__two.mp4) |
| rgb | 4 objects, seen | 1002 | 807 | seen | 1 | 2 | no | 0.45, 0.75, 0.55 | [rgb__four.mp4](videos/CF/rgb__four.mp4) |

## Regenerating

```bash
source ~/octvla/env-leftmost.sh
for p in F AF CF; do
  PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" scripts/select_rollout_videos.py \
    "$OCTVLA_OUTPUT_ROOT/eval" --prefix $p --dest docs/videos/$p --table /tmp/$p.md
done
```

Home allocates 32 MB per file, so these 45 videos (18 MB of data) take about
1.5 GB of the home quota.
