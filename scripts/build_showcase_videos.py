#!/usr/bin/env python
"""The project's showcase videos: for each reported case, a success above a
typical failure, in one clip, so the two behaviours can be compared directly.

Every rollout recorded all its episodes to a staging area on the vault. This
picks two per case by rule -- never by eye, because the tempting choice is the
best episode and this project has been misled by that before -- stacks them
into one labelled video under docs/videos/<section>/<case>.mp4, and writes
docs/rollout_videos.md describing exactly what each clip shows.

The rules, per case (an arm, a setting, pooled over its training seeds):

* **Median training seed**: the seed whose mean transfers is the median of the
  case's seeds (the lower of the two middle ones on a tie). Both picks come
  from it when they can, so a lucky seed is not the face of the arm.
* **Success**: the median seed's first success (lowest scene seed); if it has
  none, the first success of the seed nearest it in that ordering. A case with
  no success anywhere shows a placeholder saying so.
* **Typical failure**: each failed episode is labelled by the furthest a
  non-placed object got before it was lost (its per-object outcome from
  tasks/shelf_restock/events.py; `never_lifted` only when nothing was lifted).
  The case's most common such label is its failure mode. The failure shown is
  a median-seed episode with that mode, whose transfers are closest to the
  seed's mean over its failures, then the lowest scene seed.

    scripts/build_showcase_videos.py $OCTVLA_OUTPUT_ROOT/eval --dest docs/videos \\
        --doc docs/rollout_videos.md [--dry-run]

Home allocates 32 MB per file, so the set is kept to one clip per case.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "collect_final_results", Path(__file__).resolve().parent / "collect_final_results.py"
)
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)

OUTCOME_ORDER = ("never_lifted", "dropped_before_shelf", "carried_not_reached",
                 "reached_not_placed_held", "reached_not_placed_dropped", "placed_then_lost")
MODE_LABEL = {
    "never_lifted": "never lifts an object",
    "dropped_before_shelf": "drops the object before the shelf",
    "carried_not_reached": "carries the object but never reaches the shelf",
    "reached_not_placed_held": "reaches the shelf but never releases",
    "reached_not_placed_dropped": "reaches the shelf and drops the object",
    "placed_then_lost": "places an object, then knocks it off",
    "unknown": "runs out of steps (outcome not recorded)",
}
TIER_LABEL = {"seen": "seen objects", "heldout": "held-out sizes", "novel": "novel mesh"}
PROFILE_LABEL = {"two_object": "2 objects", "three_object": "3 objects", "four_object": "4 objects"}

#: (section folder, section title, what to look for, cases). Each case:
#: (file name, prefix, eval subdirectory or "", arm, tier, profile, caption).
SECTIONS = [
    ("groot_stage2", "GR00T stage 2: the headline",
     "GR00T N1.7 on absolute EE, continuous runs. `rgb` is stage 1; `rgb_cont` the "
     "budget control, trained the same 8 k extra steps; `kv` and `kv_adaln` the "
     "object-conditioned arms from the same checkpoint (research_questions §4.11).",
     [("rgb__seen", "GF", "", "rgb", "seen", "three_object", "stage 1, RGB"),
      ("rgb_cont__seen", "GF", "", "rgb_cont", "seen", "three_object", "budget control"),
      ("rgb_cont__heldout", "GF", "", "rgb_cont", "heldout", "three_object", "budget control"),
      ("kv__seen", "GF", "", "kv", "seen", "three_object", "KV conditioning"),
      ("kv__heldout", "GF", "", "kv", "heldout", "three_object", "KV conditioning"),
      ("kv_adaln__seen", "GF", "", "kv_adaln", "seen", "three_object", "KV + AdaLN"),
      ("kv_adaln__heldout", "GF", "", "kv_adaln", "heldout", "three_object", "KV + AdaLN")]),
    ("groot_controls", "GR00T: information, composition and SIGReg controls",
     "Same budget as `kv_adaln` (§4.13). `kv_adaln_shuffled` has its exact "
     "parameters but trains on another scene's objects; `scene_attn` / `scene_mean` "
     "drop KV and keep only a pooled scene vector; `kv_adaln_sigreg` adds SIGReg "
     "on the object embeddings.",
     [("kv_adaln_shuffled__seen", "GF", "", "kv_adaln_shuffled", "seen", "three_object", "wrong scene's objects"),
      ("scene_attn__seen", "GF", "", "scene_attn", "seen", "three_object", "attention-pooled scene, no KV"),
      ("scene_mean__seen", "GF", "", "scene_mean", "seen", "three_object", "mean-pooled scene, no KV"),
      ("kv_adaln_sigreg__seen", "GF", "", "kv_adaln_sigreg", "seen", "three_object", "KV + AdaLN + SIGReg"),
      ("kv_adaln_sigreg__heldout", "GF", "", "kv_adaln_sigreg", "heldout", "three_object", "KV + AdaLN + SIGReg")]),
    ("groot_object_count", "GR00T: object count (trained on three)",
     "Two objects on the nested layout (400 steps), four at 800 steps, seen "
     "objects (§4.14).",
     [(f"{arm}__{n}", "GF", "", arm, "seen", f"{n}_object", cap)
      for n in ("two", "four")
      for arm, cap in (("rgb_cont", "budget control"), ("kv", "KV conditioning"),
                       ("kv_adaln", "KV + AdaLN"))]),
    ("groot_visual_shift", "GR00T: visual shift",
     "Unseen backgrounds and random lighting (RoboTwin's randomisation), seen "
     "objects (§4.13).",
     [(f"{arm}__vshift", "GF", "vshift", arm, "seen", "three_object", cap)
      for arm, cap in (("rgb_cont", "budget control"), ("kv", "KV conditioning"),
                       ("kv_adaln", "KV + AdaLN"))]),
    ("groot_data_fraction", "GR00T: fewer training runs",
     "GR00T trained on 19 (`GQF`) or 38 (`GHF`) of the 75 runs, 8 k steps per "
     "stage (§4.13).",
     [(f"{p.lower()}_{arm}__seen", p, "", arm, "seen", "three_object", f"{runs} runs, {cap}")
      for p, runs in (("GQF", 19), ("GHF", 38))
      for arm, cap in (("rgb_cont", "budget control"), ("kv_adaln", "KV + AdaLN"))]),
    ("act_chaining", "ACT: atomic clips against continuous runs",
     "The same RGB ACT trained on one-transfer clips (`AF`) or on whole runs "
     "(`CF`) of the same corpus (§4.6): only the run-trained policy chains transfers.",
     [("atomic__seen", "AF", "", "rgb", "seen", "three_object", "trained on atomic clips"),
      ("atomic__two", "AF", "", "rgb", "seen", "two_object", "trained on atomic clips"),
      ("full_runs__seen", "CF", "", "rgb", "seen", "three_object", "trained on full runs"),
      ("full_runs__two", "CF", "", "rgb", "seen", "two_object", "trained on full runs"),
      ("full_runs__four", "CF", "", "rgb", "seen", "four_object", "trained on full runs")]),
    ("act_conditioning", "ACT: object conditioning",
     "Stage 2 on full runs (`CF`, §4.9) and on the atomic identity corpus (`F`, "
     "§4.1). Against the budget control ACT gains only on held-out sizes, through AdaLN.",
     [("cf_rgb_cont__seen", "CF", "", "rgb_cont", "seen", "three_object", "budget control"),
      ("cf_rgb_cont__heldout", "CF", "", "rgb_cont", "heldout", "three_object", "budget control"),
      ("cf_kv_adaln__seen", "CF", "", "kv_adaln", "seen", "three_object", "KV + AdaLN"),
      ("cf_kv_adaln__heldout", "CF", "", "kv_adaln", "heldout", "three_object", "KV + AdaLN"),
      ("cf_kv_tokens__seen", "CF", "", "kv_tokens", "seen", "three_object", "KV + entity tokens"),
      ("f_rgb_cont__heldout", "F", "", "rgb_cont", "heldout", "three_object", "atomic clips, budget control"),
      ("f_kv_adaln__heldout", "F", "", "kv_adaln", "heldout", "three_object", "atomic clips, KV + AdaLN")]),
    ("act_baseline_ablations", "ACT: changes to the RGB baseline",
     "Against `CF-rgb` (§4.7, §4.8).",
     [("rgb_short__seen", "CF", "", "rgb_short", "seen", "three_object", "chunk 20, execute 8"),
      ("rgb_hist__seen", "CF", "", "rgb_hist", "seen", "three_object", "two-frame history"),
      ("rgb_rel__seen", "CF", "", "rgb_rel", "seen", "three_object", "chunk-relative actions"),
      ("rgb_shift__novel", "CF", "", "rgb_shift", "novel", "three_object", "random camera shift")]),
    ("act_control_regime", "ACT: control regime",
     "RGB ACT on the atomic identity corpus in four action spaces (§4.4): both "
     "delta regimes never complete a transfer.",
     [("absolute_ee__seen", "F", "", "rgb", "seen", "three_object", "absolute EE"),
      ("absolute_joint__seen", "J", "", "rgb", "seen", "three_object", "absolute joint"),
      ("joint_delta__seen", "D", "", "rgb", "seen", "three_object", "joint delta"),
      ("ee_delta__seen", "X", "", "rgb", "seen", "three_object", "EE delta")]),
    ("vla", "pi0.5 and SmolVLA",
     "LoRA fine-tuned VLAs on continuous runs: the default LoRA (`PF`) and the "
     "rank-32 expert LoRA (`PXF`, `SXF`), stage 2 against its budget control (§4.10, §4.12).",
     [("pi05_default_lora__seen", "PF", "", "rgb", "seen", "three_object", "pi0.5 stage 1, default LoRA"),
      ("pi05_rgb_cont__seen", "PXF", "", "rgb_cont", "seen", "three_object", "pi0.5 budget control"),
      ("pi05_kv__seen", "PXF", "", "kv", "seen", "three_object", "pi0.5 KV conditioning"),
      ("smolvla_rgb_cont__seen", "SXF", "", "rgb_cont", "seen", "three_object", "SmolVLA budget control"),
      ("smolvla_kv__seen", "SXF", "", "kv", "seen", "three_object", "SmolVLA KV conditioning")]),
]


def failure_mode(row: dict) -> str:
    outcomes = [o for o in row["outcomes"] if o != "placed"]
    if not row["outcomes"]:
        return "unknown"
    lifted = [o for o in outcomes if o != "never_lifted"]
    pool = lifted or outcomes or ["never_lifted"]
    return max(pool, key=OUTCOME_ORDER.index)


def seed_order(rows: list[dict]) -> list[int]:
    """Training seeds, the median seed first, then by distance from it."""
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(r["train_seed"], []).append(r)
    ranked = sorted(by_seed, key=lambda s: (statistics.mean(r["transfers"] for r in by_seed[s]), s))
    middle = (len(ranked) - 1) // 2
    return sorted(ranked, key=lambda s: (abs(ranked.index(s) - middle), ranked.index(s)))


def pick(rows: list[dict]) -> tuple[dict | None, dict | None, str, Counter]:
    seeds = seed_order(rows)
    success = None
    for seed in seeds:
        wins = sorted((r for r in rows if r["train_seed"] == seed and r["success"]),
                      key=lambda r: r["eval_seed"])
        if wins:
            success = wins[0]
            break
    failed = [r for r in rows if not r["success"]]
    modes = Counter(failure_mode(r) for r in failed)
    mode = modes.most_common(1)[0][0] if modes else ""
    failure = None
    for seed in seeds:
        pool = [r for r in failed if r["train_seed"] == seed]
        if not pool:
            continue
        mean_t = statistics.mean(r["transfers"] for r in pool)
        typical = [r for r in pool if failure_mode(r) == mode]
        if typical:
            failure = min(typical, key=lambda r: (abs(r["transfers"] - mean_t), r["eval_seed"]))
            break
    return success, failure, mode, modes


def compose(top, bottom, top_label, bottom_label, out: Path, np, cv2, av) -> None:
    """Stack two episode videos, each under a label bar; the shorter one holds
    its last frame, dimmed, once its episode has ended."""
    def frames(path):
        if path is None:
            return None
        container = av.open(str(path))
        out_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        return out_frames

    a, b = frames(top), frames(bottom)
    ref = a or b
    height, width = ref[0].shape[:2]
    bar = 30
    blank = np.zeros((height, width, 3), np.uint8)
    length = max(len(a or [blank]), len(b or [blank]))

    def panel(seq, index, label, colour, empty_text):
        img = np.zeros((height + bar, width, 3), np.uint8)
        img[:bar] = colour
        cv2.putText(img, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if seq is None:
            cv2.putText(img, empty_text, (width // 2 - 170, bar + height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
            return img
        if index < len(seq):
            img[bar:] = seq[index]
        else:
            img[bar:] = (seq[-1] * 0.45).astype(np.uint8)
            cv2.putText(img, "episode ended", (width // 2 - 80, bar + height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out), mode="w")
    stream = container.add_stream("libx264", rate=15)
    stream.width, stream.height = width, 2 * (height + bar)
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "30", "preset": "slow"}
    for i in range(length):
        image = np.concatenate([
            panel(a, i, top_label, (40, 110, 40), "no success in any episode"),
            panel(b, i, bottom_label, (140, 45, 45), "no failure in any episode"),
        ])
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def transfers(r: dict) -> str:
    n = r["transfers"]
    return f"{n:g} transfer" + ("" if n == 1 else "s")


def describe(r: dict | None) -> str:
    if r is None:
        return "—"
    return (f"seed {r['train_seed']}, scene {r['eval_seed']}: {transfers(r)}, "
            f"{r['lifted']} lifted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        import av
        import cv2
        import numpy as np

    loaded: dict[tuple[str, str], list[dict]] = {}
    doc = [DOC_HEAD]
    problems = []
    for folder, title, note, cases in SECTIONS:
        doc += [f"## {title} (`videos/{folder}/`)", "", note, "",
                "| clip | case | success | success shown | typical failure: mode (share of failures) | failure shown |",
                "| --- | --- | ---: | --- | --- | --- |"]
        for name, prefix, subdir, arm, tier, profile, caption in cases:
            key = (prefix, subdir)
            if key not in loaded:
                loaded[key], _ = collect.load(args.eval_dir / subdir if subdir else args.eval_dir, prefix)
            rows = collect.select(loaded[key], arm=arm, tier=tier, profile=profile)
            if not rows:
                problems.append(f"{folder}/{name}: no episodes")
                continue
            success, failure, mode, modes = pick(rows)
            for r in (success, failure):
                if r is not None and not (r["video"] and Path(r["video"]).is_file()):
                    problems.append(f"{folder}/{name}: missing video {r['video'] or '(none recorded)'}")
            wins = sum(r["success"] for r in rows)
            setting = f"{PROFILE_LABEL[profile]}, {TIER_LABEL[tier]}" + (", visual shift" if subdir == "vshift" else "")
            head = f"{prefix}-{arm} | {setting}"
            share = f"{modes[mode]}/{sum(modes.values())}" if modes else "0"
            top = (f"SUCCESS ({wins}/{len(rows)}) | {head} | "
                   + (f"train seed {success['train_seed']}, scene {success['eval_seed']}" if success else "none"))
            bottom = (f"TYPICAL FAILURE: {MODE_LABEL.get(mode, mode)} ({share}) | "
                      + (f"train seed {failure['train_seed']}, scene {failure['eval_seed']}, "
                         + transfers(failure) if failure else "none"))
            path = args.dest / folder / f"{name}.mp4"
            if not args.dry_run:
                compose(Path(success["video"]) if success else None,
                        Path(failure["video"]) if failure else None,
                        top, bottom, path, np, cv2, av)
            doc.append(f"| [{name}.mp4](videos/{folder}/{name}.mp4) | `{prefix}-{arm}`, {caption}; "
                       f"{setting} | {wins}/{len(rows)} | {describe(success)} | "
                       f"{MODE_LABEL.get(mode, mode) if modes else '—'} ({share}) | {describe(failure)} |")
            print(f"{folder}/{name}: success {wins}/{len(rows)}; failure mode {mode} ({share})")
        doc.append("")
    doc.append(DOC_TAIL)
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    if not args.dry_run:
        args.doc.write_text("\n".join(doc))
        size = sum(p.stat().st_size for p in args.dest.rglob("*.mp4")) / 1e6
        print(f"\nwrote {args.doc} and {len(list(args.dest.rglob('*.mp4')))} clips ({size:.1f} MB)")
    return 0


DOC_HEAD = """# Rollout videos

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
"""

DOC_TAIL = """## Regenerating

```bash
source ~/octvla/env-leftmost.sh
PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" scripts/build_showcase_videos.py \\
    "$OCTVLA_OUTPUT_ROOT/eval" --dest docs/videos --doc docs/rollout_videos.md
```

It needs the vault staging area (`$HPCVAULT/octvla-rollout-videos/`), which
holds every recorded episode.
"""


if __name__ == "__main__":
    raise SystemExit(main())
