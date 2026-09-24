#!/usr/bin/env python
"""How much a trained policy relies on each camera just before a grasp.

Rollouts show the arm reaching the right object and then knocking it: the
approach is right to a few centimetres, the grasp needs a few millimetres, and
only the wrist cameras see that closely. This asks, offline and on held-out
validation frames, whether the policy uses them.

For every gripper-close event in the validation episodes (the acting arm's
binary gripper command switching to closed), and for frames 0-30 steps before
it:

* **grasp error**: distance between the grasp position the policy predicts
  for the close step (from its action chunk) and the oracle's actual one;
* **reliance on camera c**: how far that predicted grasp position moves when
  camera c's frame is replaced by the *same camera* at a random other moment
  of the same episode (in-distribution, wrong content), and when it is blanked.

A policy that aligns the grasp with a wrist camera moves its predicted grasp
when that camera is swapped; one that ignores it does not.

    diagnose_camera_reliance.py <checkpoint> <dataset root> [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Offsets of the left and right arm's (x, y, z, ..., gripper) in the 16-wide
#: absolute EE action (lerobot_export.py).
ARMS = {"left": 0, "right": 8}
GRIPPER = 7


def close_events(actions, threshold=0.5):
    """(frame, arm) where an arm's gripper command goes from open to closed."""
    events = []
    for arm, offset in ARMS.items():
        g = actions[:, offset + GRIPPER]
        for t in range(1, len(g)):
            if g[t - 1] > threshold >= g[t]:
                events.append((t, arm))
    return events


def main() -> int:
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    import oct_vla.policies  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--leads", default="0,5,10,15,20,25,30",
                        help="Frames before the close event to predict from")
    parser.add_argument("--max-events", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=("val", "train"), default="val",
                        help="train measures fit rather than generalisation")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    leads = [int(v) for v in args.leads.split(",")]
    rng = random.Random(0)

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = args.device
    policy = get_policy_class(config.type).from_pretrained(args.checkpoint, config=config)
    policy.eval().to(args.device)
    pre, post = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    if getattr(config, "observation_delta_indices", None):
        raise SystemExit("history policies need stacked frames; not supported here yet")
    cameras = list(config.image_features)
    chunk = config.chunk_size

    manifest = json.loads((args.dataset / "split_manifest.json").read_text())
    first, last = manifest[args.split]["episodes"]
    dataset = LeRobotDataset("local/diag", root=args.dataset,
                             episodes=list(range(first, last + 1)))

    by_episode: dict[int, list[int]] = {}
    for index in range(len(dataset.hf_dataset)):
        episode = int(dataset.hf_dataset[index]["episode_index"])
        by_episode.setdefault(episode, []).append(index)

    def frame(i):
        item = dataset[i]
        return {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in item.items()}

    @torch.no_grad()
    def grasp(batch, arm, steps_ahead):
        """Predicted position of `arm` at `steps_ahead` in the chunk, in metres."""
        observation = {k: v for k, v in batch.items() if k.startswith("observation.")}
        observation["task"] = batch.get("task", [""])
        observation = pre(observation)
        actions = post(policy.predict_action_chunk(observation))[0].float().cpu().numpy()
        offset = ARMS[arm]
        return actions[steps_ahead, offset : offset + 3]

    rows = []
    for episode, indices in sorted(by_episode.items()):
        actions = np.stack([dataset.hf_dataset[i]["action"] for i in indices])
        actions = np.asarray(actions, dtype=np.float64)
        for close, arm in close_events(actions):
            truth = actions[close, ARMS[arm] : ARMS[arm] + 3]
            for lead in leads:
                t = close - lead
                if t < 0 or lead >= chunk:
                    continue
                base_batch = frame(indices[t])
                base = grasp(base_batch, arm, lead)
                row = {"episode": episode, "arm": arm, "lead": lead,
                       "error_mm": float(np.linalg.norm(base - truth) * 1000),
                       "error_xy_mm": float(np.linalg.norm((base - truth)[:2]) * 1000)}
                other = indices[rng.choice([i for i in range(len(indices)) if abs(i - t) > 30])]
                other_batch = frame(other)
                for camera in cameras:
                    swapped = dict(base_batch)
                    swapped[camera] = other_batch[camera]
                    moved = grasp(swapped, arm, lead) - base
                    blank = dict(base_batch)
                    blank[camera] = torch.full_like(base_batch[camera], 0.5)
                    moved_blank = grasp(blank, arm, lead) - base
                    name = camera.rsplit(".", 1)[-1]
                    row[f"swap_{name}_mm"] = float(np.linalg.norm(moved) * 1000)
                    row[f"blank_{name}_mm"] = float(np.linalg.norm(moved_blank) * 1000)
                rows.append(row)
            if len({(r["episode"], r["arm"]) for r in rows}) >= args.max_events:
                break
        if len({(r["episode"], r["arm"]) for r in rows}) >= args.max_events:
            break

    names = [c.rsplit(".", 1)[-1] for c in cameras]
    summary = {}
    print(f"{len(rows)} (event, lead) samples from "
          f"{len({(r['episode'], r['arm']) for r in rows})} grasps\n")
    print(f"{'lead':>4} {'n':>4} {'grasp err':>10} {'xy err':>8}  "
          + "  ".join(f"{'swap ' + n:>16}" for n in names) + "   "
          + "  ".join(f"{'blank ' + n:>17}" for n in names))
    for lead in leads:
        sel = [r for r in rows if r["lead"] == lead]
        if not sel:
            continue
        med = lambda key: statistics.median(r[key] for r in sel)  # noqa: E731
        summary[lead] = {"n": len(sel), "error_mm": med("error_mm"), "error_xy_mm": med("error_xy_mm"),
                         **{f"swap_{n}_mm": med(f"swap_{n}_mm") for n in names},
                         **{f"blank_{n}_mm": med(f"blank_{n}_mm") for n in names}}
        s = summary[lead]
        print(f"{lead:>4} {len(sel):>4} {s['error_mm']:>8.1f}mm {s['error_xy_mm']:>6.1f}mm  "
              + "  ".join(f"{s[f'swap_{n}_mm']:>14.1f}mm" for n in names) + "   "
              + "  ".join(f"{s[f'blank_{n}_mm']:>15.1f}mm" for n in names))
    print("\nmedians; grasp err = predicted vs oracle grasp position; swap/blank = how far the "
          "predicted grasp moves when that camera's frame is replaced")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "summary": summary,
                                        "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
