# Rollout videos

One clip per reported case: **a success on top, a typical failure below**,
played in step, so the two behaviours can be compared frame by frame. Every
clip shows the three cameras (head, left wrist, right wrist). The label bars
name the case, the training and scene seeds, and how often the case succeeds.

**The clips are chosen by rule, not by eye** (`scripts/build_showcase_videos.py`):
- both come from the case's **median training seed** where they can, so a
  lucky seed is not the face of the arm;
- the **success** is that seed's first success (lowest scene seed), or the
  nearest seed's when it has none;
- the **typical failure** shows the case's **most common failure mode**: each
  failed episode is labelled by how far its furthest lost object got, and the
  clip is a median-seed episode with the commonest label and transfers closest
  to the seed's mean over failures.

A success is therefore an existence proof, not the typical outcome: the
`success` column says how often it happens. Settings: three objects run 600
steps, two 400, four 800; seen, held-out and novel meshes as in
`research_questions.md` §1. Every episode of every rollout was recorded to the
vault staging area, from which these were taken.

Two cases carry no failure mode, labelled "runs out of steps (outcome not
recorded)": two-object failures recorded before 2026-09-30, which the
step-limit bug ended before the server wrote their per-object outcomes
(`research_questions.md` §1), and the September atomic-clip rollouts
(`F`, held-out), which predate the outcome tracker. Their failure clip is the
median-seed failure with transfers closest to the seed's mean.

## GR00T stage 2: the headline (`videos/groot_stage2/`)

GR00T N1.7 on absolute EE, continuous runs. `rgb` is stage 1; `rgb_cont` the budget control, trained the same 8 k extra steps; `kv` and `kv_adaln` the object-conditioned arms from the same checkpoint (research_questions §4.11).

| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |
| --- | --- | ---: | --- | --- | --- |
| [rgb__seen.mp4](videos/groot_stage2/rgb__seen.mp4) | `GF-rgb`, stage 1, RGB; 3 objects, seen objects | 4/60 | seed 1001, scene 800: 3 transfers, 3 lifted | places an object, then knocks it off (29/56) | seed 1001, scene 802: 0 transfers, 2 lifted |
| [rgb_cont__seen.mp4](videos/groot_stage2/rgb_cont__seen.mp4) | `GF-rgb_cont`, budget control; 3 objects, seen objects | 9/60 | seed 1002, scene 802: 3 transfers, 3 lifted | places an object, then knocks it off (33/51) | seed 1002, scene 800: 1 transfer, 2 lifted |
| [rgb_cont__heldout.mp4](videos/groot_stage2/rgb_cont__heldout.mp4) | `GF-rgb_cont`, budget control; 3 objects, held-out sizes | 0/60 | — | places an object, then knocks it off (36/60) | seed 1000, scene 803: 0 transfers, 1 lifted |
| [kv__seen.mp4](videos/groot_stage2/kv__seen.mp4) | `GF-kv`, KV conditioning; 3 objects, seen objects | 31/60 | seed 1002, scene 801: 3 transfers, 3 lifted | places an object, then knocks it off (16/29) | seed 1002, scene 800: 1 transfer, 3 lifted |
| [kv__heldout.mp4](videos/groot_stage2/kv__heldout.mp4) | `GF-kv`, KV conditioning; 3 objects, held-out sizes | 7/60 | seed 1002, scene 814: 3 transfers, 3 lifted | places an object, then knocks it off (48/53) | seed 1002, scene 800: 1 transfer, 3 lifted |
| [kv_adaln__seen.mp4](videos/groot_stage2/kv_adaln__seen.mp4) | `GF-kv_adaln`, KV + AdaLN; 3 objects, seen objects | 35/60 | seed 1001, scene 801: 3 transfers, 3 lifted | places an object, then knocks it off (16/25) | seed 1001, scene 802: 0 transfers, 2 lifted |
| [kv_adaln__heldout.mp4](videos/groot_stage2/kv_adaln__heldout.mp4) | `GF-kv_adaln`, KV + AdaLN; 3 objects, held-out sizes | 11/60 | seed 1000, scene 812: 3 transfers, 3 lifted | places an object, then knocks it off (42/49) | seed 1000, scene 800: 0 transfers, 2 lifted |

## GR00T: information, composition and SIGReg controls (`videos/groot_controls/`)

Same budget as `kv_adaln` (§4.13). `kv_adaln_shuffled` has its exact parameters but trains on another scene's objects; `scene_attn` / `scene_mean` drop KV and keep only a pooled scene vector; `kv_adaln_sigreg` adds SIGReg on the object embeddings.

| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |
| --- | --- | ---: | --- | --- | --- |
| [kv_adaln_shuffled__seen.mp4](videos/groot_controls/kv_adaln_shuffled__seen.mp4) | `GF-kv_adaln_shuffled`, wrong scene's objects; 3 objects, seen objects | 8/60 | seed 1000, scene 800: 3 transfers, 3 lifted | places an object, then knocks it off (30/52) | seed 1000, scene 809: 1 transfer, 3 lifted |
| [scene_attn__seen.mp4](videos/groot_controls/scene_attn__seen.mp4) | `GF-scene_attn`, attention-pooled scene, no KV; 3 objects, seen objects | 12/60 | seed 1002, scene 812: 3 transfers, 3 lifted | places an object, then knocks it off (29/48) | seed 1002, scene 802: 1 transfer, 2 lifted |
| [scene_mean__seen.mp4](videos/groot_controls/scene_mean__seen.mp4) | `GF-scene_mean`, mean-pooled scene, no KV; 3 objects, seen objects | 8/60 | seed 1000, scene 800: 3 transfers, 3 lifted | places an object, then knocks it off (26/52) | seed 1000, scene 813: 1 transfer, 2 lifted |
| [kv_adaln_sigreg__seen.mp4](videos/groot_controls/kv_adaln_sigreg__seen.mp4) | `GF-kv_adaln_sigreg`, KV + AdaLN + SIGReg; 3 objects, seen objects | 25/60 | seed 1000, scene 800: 3 transfers, 3 lifted | places an object, then knocks it off (25/35) | seed 1000, scene 802: 0 transfers, 1 lifted |
| [kv_adaln_sigreg__heldout.mp4](videos/groot_controls/kv_adaln_sigreg__heldout.mp4) | `GF-kv_adaln_sigreg`, KV + AdaLN + SIGReg; 3 objects, held-out sizes | 18/60 | seed 1000, scene 802: 3 transfers, 3 lifted | places an object, then knocks it off (34/42) | seed 1000, scene 800: 1 transfer, 2 lifted |

## GR00T: object count (trained on three) (`videos/groot_object_count/`)

Two objects on the nested layout (400 steps), four at 800 steps, seen objects (§4.14).

| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |
| --- | --- | ---: | --- | --- | --- |
| [rgb_cont__two.mp4](videos/groot_object_count/rgb_cont__two.mp4) | `GF-rgb_cont`, budget control; 2 objects, seen objects | 14/60 | seed 1000, scene 807: 2 transfers, 2 lifted | runs out of steps (outcome not recorded) (46/46) | seed 1000, scene 803: 0 transfers, 1 lifted |
| [kv__two.mp4](videos/groot_object_count/kv__two.mp4) | `GF-kv`, KV conditioning; 2 objects, seen objects | 47/60 | seed 1000, scene 800: 2 transfers, 2 lifted | runs out of steps (outcome not recorded) (13/13) | seed 1000, scene 804: 0 transfers, 1 lifted |
| [kv_adaln__two.mp4](videos/groot_object_count/kv_adaln__two.mp4) | `GF-kv_adaln`, KV + AdaLN; 2 objects, seen objects | 37/60 | seed 1000, scene 803: 2 transfers, 2 lifted | runs out of steps (outcome not recorded) (23/23) | seed 1000, scene 800: 0 transfers, 1 lifted |
| [rgb_cont__four.mp4](videos/groot_object_count/rgb_cont__four.mp4) | `GF-rgb_cont`, budget control; 4 objects, seen objects | 0/60 | — | places an object, then knocks it off (33/60) | seed 1002, scene 802: 1 transfer, 3 lifted |
| [kv__four.mp4](videos/groot_object_count/kv__four.mp4) | `GF-kv`, KV conditioning; 4 objects, seen objects | 9/60 | seed 1002, scene 803: 4 transfers, 4 lifted | places an object, then knocks it off (43/51) | seed 1002, scene 800: 0 transfers, 3 lifted |
| [kv_adaln__four.mp4](videos/groot_object_count/kv_adaln__four.mp4) | `GF-kv_adaln`, KV + AdaLN; 4 objects, seen objects | 4/60 | seed 1001, scene 801: 4 transfers, 4 lifted | places an object, then knocks it off (53/56) | seed 1001, scene 800: 0 transfers, 4 lifted |

## GR00T: visual shift (`videos/groot_visual_shift/`)

Unseen backgrounds and random lighting (RoboTwin's randomisation), seen objects (§4.13).

| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |
| --- | --- | ---: | --- | --- | --- |
| [rgb_cont__vshift.mp4](videos/groot_visual_shift/rgb_cont__vshift.mp4) | `GF-rgb_cont`, budget control; 3 objects, seen objects, visual shift | 2/60 | seed 1002, scene 814: 3 transfers, 3 lifted | places an object, then knocks it off (31/58) | seed 1002, scene 800: 0 transfers, 2 lifted |
| [kv__vshift.mp4](videos/groot_visual_shift/kv__vshift.mp4) | `GF-kv`, KV conditioning; 3 objects, seen objects, visual shift | 25/60 | seed 1002, scene 800: 3 transfers, 3 lifted | places an object, then knocks it off (27/35) | seed 1002, scene 801: 0 transfers, 3 lifted |
| [kv_adaln__vshift.mp4](videos/groot_visual_shift/kv_adaln__vshift.mp4) | `GF-kv_adaln`, KV + AdaLN; 3 objects, seen objects, visual shift | 24/60 | seed 1001, scene 801: 3 transfers, 3 lifted | places an object, then knocks it off (26/36) | seed 1001, scene 800: 0 transfers, 2 lifted |

## GR00T: fewer training runs (`videos/groot_data_fraction/`)

GR00T trained on 19 (`GQF`) or 38 (`GHF`) of the 75 runs, 8 k steps per stage (§4.13).

| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |
| --- | --- | ---: | --- | --- | --- |
| [gqf_rgb_cont__seen.mp4](videos/groot_data_fraction/gqf_rgb_cont__seen.mp4) | `GQF-rgb_cont`, 19 runs, budget control; 3 objects, seen objects | 2/60 | seed 1002, scene 812: 3 transfers, 3 lifted | places an object, then knocks it off (21/58) | seed 1002, scene 800: 0 transfers, 2 lifted |
| [gqf_kv_adaln__seen.mp4](videos/groot_data_fraction/gqf_kv_adaln__seen.mp4) | `GQF-kv_adaln`, 19 runs, KV + AdaLN; 3 objects, seen objects | 8/60 | seed 1001, scene 807: 3 transfers, 3 lifted | places an object, then knocks it off (29/52) | seed 1001, scene 802: 0 transfers, 3 lifted |
| [ghf_rgb_cont__seen.mp4](videos/groot_data_fraction/ghf_rgb_cont__seen.mp4) | `GHF-rgb_cont`, 38 runs, budget control; 3 objects, seen objects | 4/60 | seed 1002, scene 814: 3 transfers, 3 lifted | places an object, then knocks it off (30/56) | seed 1002, scene 805: 1 transfer, 3 lifted |
| [ghf_kv_adaln__seen.mp4](videos/groot_data_fraction/ghf_kv_adaln__seen.mp4) | `GHF-kv_adaln`, 38 runs, KV + AdaLN; 3 objects, seen objects | 28/60 | seed 1002, scene 801: 3 transfers, 3 lifted | places an object, then knocks it off (22/32) | seed 1002, scene 802: 1 transfer, 3 lifted |

## ACT: atomic clips against continuous runs (`videos/act_chaining/`)

The same RGB ACT trained on one-transfer clips (`AF`) or on whole runs (`CF`) of the same corpus (§4.6): only the run-trained policy chains transfers.

| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |
| --- | --- | ---: | --- | --- | --- |
| [atomic__seen.mp4](videos/act_chaining/atomic__seen.mp4) | `AF-rgb`, trained on atomic clips; 3 objects, seen objects | 1/60 | seed 1001, scene 814: 3 transfers, 3 lifted | places an object, then knocks it off (27/59) | seed 1002, scene 806: 1 transfer, 2 lifted |
| [atomic__two.mp4](videos/act_chaining/atomic__two.mp4) | `AF-rgb`, trained on atomic clips; 2 objects, seen objects | 3/60 | seed 1001, scene 807: 2 transfers, 2 lifted | runs out of steps (outcome not recorded) (57/57) | seed 1000, scene 800: 0 transfers, 1 lifted |
| [full_runs__seen.mp4](videos/act_chaining/full_runs__seen.mp4) | `CF-rgb`, trained on full runs; 3 objects, seen objects | 9/60 | seed 1000, scene 809: 3 transfers, 3 lifted | places an object, then knocks it off (22/51) | seed 1000, scene 808: 0 transfers, 2 lifted |
| [full_runs__two.mp4](videos/act_chaining/full_runs__two.mp4) | `CF-rgb`, trained on full runs; 2 objects, seen objects | 19/60 | seed 1002, scene 800: 2 transfers, 2 lifted | runs out of steps (outcome not recorded) (41/41) | seed 1002, scene 802: 0 transfers, 2 lifted |
| [full_runs__four.mp4](videos/act_chaining/full_runs__four.mp4) | `CF-rgb`, trained on full runs; 4 objects, seen objects | 0/60 | — | places an object, then knocks it off (26/60) | seed 1001, scene 802: 0 transfers, 2 lifted |

## ACT: object conditioning (`videos/act_conditioning/`)

Stage 2 on full runs (`CF`, §4.9) and on the atomic identity corpus (`F`, §4.1). Against the budget control ACT gains only on held-out sizes, through AdaLN.

| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |
| --- | --- | ---: | --- | --- | --- |
| [cf_rgb_cont__seen.mp4](videos/act_conditioning/cf_rgb_cont__seen.mp4) | `CF-rgb_cont`, budget control; 3 objects, seen objects | 21/60 | seed 1000, scene 800: 3 transfers, 3 lifted | places an object, then knocks it off (22/39) | seed 1000, scene 808: 1 transfer, 3 lifted |
| [cf_rgb_cont__heldout.mp4](videos/act_conditioning/cf_rgb_cont__heldout.mp4) | `CF-rgb_cont`, budget control; 3 objects, held-out sizes | 0/60 | — | places an object, then knocks it off (27/60) | seed 1000, scene 802: 0 transfers, 1 lifted |
| [cf_kv_adaln__seen.mp4](videos/act_conditioning/cf_kv_adaln__seen.mp4) | `CF-kv_adaln`, KV + AdaLN; 3 objects, seen objects | 22/60 | seed 1002, scene 800: 3 transfers, 3 lifted | places an object, then knocks it off (24/38) | seed 1002, scene 803: 0 transfers, 1 lifted |
| [cf_kv_adaln__heldout.mp4](videos/act_conditioning/cf_kv_adaln__heldout.mp4) | `CF-kv_adaln`, KV + AdaLN; 3 objects, held-out sizes | 7/60 | seed 1000, scene 810: 3 transfers, 3 lifted | places an object, then knocks it off (32/53) | seed 1000, scene 801: 0 transfers, 2 lifted |
| [cf_kv_tokens__seen.mp4](videos/act_conditioning/cf_kv_tokens__seen.mp4) | `CF-kv_tokens`, KV + entity tokens; 3 objects, seen objects | 26/60 | seed 1001, scene 800: 3 transfers, 3 lifted | places an object, then knocks it off (16/34) | seed 1001, scene 802: 0 transfers, 1 lifted |
| [f_rgb_cont__heldout.mp4](videos/act_conditioning/f_rgb_cont__heldout.mp4) | `F-rgb_cont`, atomic clips, budget control; 3 objects, held-out sizes | 0/60 | — | drops the object before the shelf (22/60) | seed 1000, scene 801: 0 transfers, 1 lifted |
| [f_kv_adaln__heldout.mp4](videos/act_conditioning/f_kv_adaln__heldout.mp4) | `F-kv_adaln`, atomic clips, KV + AdaLN; 3 objects, held-out sizes | 6/60 | seed 1000, scene 814: 3 transfers, 3 lifted | runs out of steps (outcome not recorded) (54/54) | seed 1000, scene 800: 0 transfers, 1 lifted |

## ACT: changes to the RGB baseline (`videos/act_baseline_ablations/`)

Against `CF-rgb` (§4.7, §4.8).

| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |
| --- | --- | ---: | --- | --- | --- |
| [rgb_short__seen.mp4](videos/act_baseline_ablations/rgb_short__seen.mp4) | `CF-rgb_short`, chunk 20, execute 8; 3 objects, seen objects | 7/60 | seed 1000, scene 809: 3 transfers, 3 lifted | never lifts an object (23/53) | seed 1000, scene 814: 1 transfer, 1 lifted |
| [rgb_hist__seen.mp4](videos/act_baseline_ablations/rgb_hist__seen.mp4) | `CF-rgb_hist`, two-frame history; 3 objects, seen objects | 2/60 | seed 1002, scene 811: 3 transfers, 3 lifted | drops the object before the shelf (25/58) | seed 1002, scene 803: 1 transfer, 2 lifted |
| [rgb_rel__seen.mp4](videos/act_baseline_ablations/rgb_rel__seen.mp4) | `CF-rgb_rel`, chunk-relative actions; 3 objects, seen objects | 4/60 | seed 1000, scene 803: 3 transfers, 3 lifted | places an object, then knocks it off (35/56) | seed 1001, scene 801: 1 transfer, 3 lifted |
| [rgb_shift__novel.mp4](videos/act_baseline_ablations/rgb_shift__novel.mp4) | `CF-rgb_shift`, random camera shift; 3 objects, novel mesh | 5/60 | seed 1002, scene 811: 3 transfers, 3 lifted | never lifts an object (29/55) | seed 1002, scene 803: 1 transfer, 1 lifted |

## ACT: control regime (`videos/act_control_regime/`)

RGB ACT on the atomic identity corpus in four action spaces (§4.4): both delta regimes never complete a transfer.

| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |
| --- | --- | ---: | --- | --- | --- |
| [absolute_ee__seen.mp4](videos/act_control_regime/absolute_ee__seen.mp4) | `F-rgb`, absolute EE; 3 objects, seen objects | 2/60 | seed 1001, scene 819: 3 transfers, 3 lifted | never lifts an object (24/58) | seed 1001, scene 803: 0 transfers, 0 lifted |
| [absolute_joint__seen.mp4](videos/act_control_regime/absolute_joint__seen.mp4) | `J-rgb`, absolute joint; 3 objects, seen objects | 1/60 | seed 1000, scene 808: 3 transfers, 3 lifted | drops the object before the shelf (21/59) | seed 1000, scene 803: 0 transfers, 2 lifted |
| [joint_delta__seen.mp4](videos/act_control_regime/joint_delta__seen.mp4) | `D-rgb`, joint delta; 3 objects, seen objects | 0/60 | — | never lifts an object (50/60) | seed 1001, scene 800: 0 transfers, 0 lifted |
| [ee_delta__seen.mp4](videos/act_control_regime/ee_delta__seen.mp4) | `X-rgb`, EE delta; 3 objects, seen objects | 0/60 | — | never lifts an object (45/60) | seed 1001, scene 800: 0 transfers, 0 lifted |

## pi0.5 and SmolVLA (`videos/vla/`)

LoRA fine-tuned VLAs on continuous runs: the default LoRA (`PF`) and the rank-32 expert LoRA (`PXF`, `SXF`), stage 2 against its budget control (§4.10, §4.12).

| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |
| --- | --- | ---: | --- | --- | --- |
| [pi05_default_lora__seen.mp4](videos/vla/pi05_default_lora__seen.mp4) | `PF-rgb`, pi0.5 stage 1, default LoRA; 3 objects, seen objects | 0/60 | — | never lifts an object (35/60) | seed 1001, scene 800: 0 transfers, 0 lifted |
| [pi05_rgb_cont__seen.mp4](videos/vla/pi05_rgb_cont__seen.mp4) | `PXF-rgb_cont`, pi0.5 budget control; 3 objects, seen objects | 0/60 | — | drops the object before the shelf (27/60) | seed 1001, scene 800: 0 transfers, 1 lifted |
| [pi05_kv__seen.mp4](videos/vla/pi05_kv__seen.mp4) | `PXF-kv`, pi0.5 KV conditioning; 3 objects, seen objects | 1/60 | seed 1002, scene 807: 3 transfers, 3 lifted | drops the object before the shelf (23/59) | seed 1000, scene 803: 0 transfers, 1 lifted |
| [smolvla_rgb_cont__seen.mp4](videos/vla/smolvla_rgb_cont__seen.mp4) | `SXF-rgb_cont`, SmolVLA budget control; 3 objects, seen objects | 0/60 | — | never lifts an object (20/60) | seed 1001, scene 800: 0 transfers, 0 lifted |
| [smolvla_kv__seen.mp4](videos/vla/smolvla_kv__seen.mp4) | `SXF-kv`, SmolVLA KV conditioning; 3 objects, seen objects | 4/60 | seed 1000, scene 814: 3 transfers, 3 lifted | drops the object before the shelf (20/56) | seed 1000, scene 800: 0 transfers, 2 lifted |

## Regenerating

```bash
source ~/octvla/env-leftmost.sh
PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" scripts/build_showcase_videos.py \
    "$OCTVLA_OUTPUT_ROOT/eval" --dest docs/videos --doc docs/rollout_videos.md
```

It needs the vault staging area (`$HPCVAULT/octvla-rollout-videos/`), which
holds every recorded episode.
