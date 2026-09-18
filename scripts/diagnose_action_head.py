#!/usr/bin/env python
"""Ask whether a trained policy predicts its action target at all, offline.

Closed-loop evaluation answers "did it do the task", which has been 0 for every
arm of every sweep. That is the wrong granularity to debug from: a zero is
equally consistent with a broken bridge, a bad action encoding and a model that
never learned. This script asks the narrower question the sweep cannot -- given
the observation, how close is the predicted action to the recorded one -- and
answers it against reference predictors that need no model at all.

Three things it measures, all on the *validation* split and all in the policy's
own normalisation units so the numbers are comparable to the training log's
`eval_loss`:

1. **Reference predictors.** A constant (the training mean of each action
   dimension) and persistence (repeat the previous action). These bracket the
   problem. A model that does not beat the constant has learned nothing; a
   persistence score far below the model's says the target is strongly
   determined by its own history, which matters because a single-frame policy
   cannot see that history.

2. **L1 by chunk offset**, through `predict_action_chunk` -- the inference path.
   This is deliberately not the training log's `eval_loss`, which comes from
   `forward()` with the VAE encoder conditioned on the ground-truth action
   chunk, and is therefore optimistic about what inference can do. It is also
   averaged over the whole chunk, which hides whether the near-term prediction
   is good and the far end is bad or whether everything is uniformly poor. That
   distinction decides whether shortening the execution horizon can help.

3. **L1 by action dimension** at offset 0. The gripper channels are 2 of 16
   dimensions, so a decisive error there is nearly invisible in the aggregate
   while being fatal to a task that requires grasping.

Runs on the policy environment, needs a GPU for anything but the reference rows,
and touches no simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Offsets reported individually. The whole chunk is measured; these are the
#: rows printed, chosen to show the near term densely and the far end sparsely.
REPORTED_OFFSETS = (0, 1, 2, 5, 10, 25, 49)


def episode_bounds(dataset_root: Path) -> int:
    """First validation episode index, from the split the exporter recorded.

    Read from split_manifest.json rather than recomputed: the manifest is what
    training's `eval_split` was derived from, so reading it here guarantees this
    diagnostic scores the same held-out episodes the training log did.
    """
    manifest = json.loads((dataset_root / "split_manifest.json").read_text())
    return int(manifest["train"]["count"])


def reference_baselines(dataset_root: Path, first_val_episode: int) -> dict:
    """Constant and persistence, in MEAN_STD units, without loading a model.

    Both are computed from the parquet directly rather than through LeRobot so
    that they stay available when a checkpoint cannot be loaded at all, which is
    exactly when they are most worth having.
    """
    import glob

    import numpy as np
    import pandas as pd

    stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    mean = np.array(stats["action"]["mean"])
    std = np.array(stats["action"]["std"])

    files = sorted(glob.glob(str(dataset_root / "data" / "**" / "*.parquet"), recursive=True))
    frames = pd.concat(
        [pd.read_parquet(f, columns=["action", "episode_index"]) for f in files]
    )
    train = frames[frames.episode_index < first_val_episode]
    val = frames[frames.episode_index >= first_val_episode]
    if train.empty or val.empty:
        raise SystemExit(
            f"Split at episode {first_val_episode} leaves one side empty "
            f"({len(train)} train, {len(val)} val frames)"
        )

    val_z = (np.stack(val["action"].values) - mean) / std
    train_mean_z = (np.stack(train["action"].values).mean(axis=0) - mean) / std
    constant = float(np.abs(val_z - train_mean_z).mean())

    # Within episode only: the step across a boundary is not a step the robot
    # ever took, and counting it would understate how smooth the signal is.
    gaps = []
    for _, group in val.groupby("episode_index"):
        z = (np.stack(group["action"].values) - mean) / std
        if len(z) > 1:
            gaps.append(np.abs(z[1:] - z[:-1]))
    persistence = float(np.concatenate(gaps).mean()) if gaps else float("nan")

    return {
        "train_frames": int(len(train)),
        "val_frames": int(len(val)),
        "constant": constant,
        "persistence": persistence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, help="Trained policy directory")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=64,
        help="Validation batches to score. The default is enough to separate the "
        "reference predictors, which are hundredths apart.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, help="Write the measurements as JSON")
    args = parser.parse_args()

    first_val = episode_bounds(args.dataset_root)
    refs = reference_baselines(args.dataset_root, first_val)

    print(f"dataset       : {args.dataset_root}")
    print(f"split         : {refs['train_frames']} train / {refs['val_frames']} val frames "
          f"(validation starts at episode {first_val})")
    print("\nreference predictors, val L1 in MEAN_STD units")
    print(f"  constant (train mean)      : {refs['constant']:.3f}")
    print(f"  persistence (prev action)  : {refs['persistence']:.3f}")

    report = {
        "dataset_root": str(args.dataset_root),
        "first_val_episode": first_val,
        "baselines": refs,
    }

    if args.checkpoint is None:
        print("\nNo --checkpoint given; reference rows only.")
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    import oct_vla.policies  # noqa: F401  -- registers control_pi05
    from oct_vla.data.policy_inputs import rename_map_for

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = args.checkpoint
    rename_map = rename_map_for(config.type)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)

    chunk = int(getattr(config, "chunk_size", 0) or 0)
    if chunk <= 0:
        raise SystemExit(f"{config.type} declares no chunk_size; nothing to profile")

    fps = metadata.fps
    episodes = list(range(first_val, metadata.total_episodes))
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps={"action": [i / fps for i in range(chunk)]},
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    policy = make_policy(cfg=config, ds_meta=metadata, rename_map=rename_map)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    action_dim = metadata.features["action"]["shape"][0]
    total = np.zeros((chunk, action_dim))
    counts = np.zeros((chunk, action_dim))

    for index, batch in enumerate(loader):
        if index >= args.max_batches:
            break
        processed = preprocessor(batch)
        with torch.no_grad():
            predicted = policy.predict_action_chunk(processed)
        # Compare inside the policy's normalised space: `predicted` is what the
        # network emitted, and the batch's target is normalised by the same
        # preprocessor, so neither side is postprocessed back to radians.
        # Two widths can disagree here, both because the policy emits its own
        # shape rather than the dataset's. GR00T pads the action to a fixed
        # embodiment width (max_action_dim 132 against our 16) with zeros it
        # never predicts, and returns only n_action_steps rows (25) of a
        # chunk_size-40 target. Scoring the padding would average in 116 columns
        # of constant zero -- which is also why GR00T's *training* loss is not
        # comparable to any other backbone's.
        steps = min(predicted.shape[1], processed["action"].shape[1])
        target = processed["action"][:, :steps, :action_dim]
        error = (predicted[:, :steps, :action_dim] - target).abs().float().cpu().numpy()
        # Padded chunk positions repeat the episode's final action; training
        # masks them out of the loss, so counting them here would measure
        # something the model was never asked to fit.
        pad = batch.get("action_is_pad")
        if pad is not None:
            pad = pad[:, :steps]
        valid = (
            (~pad).float().cpu().numpy()[..., None]
            if pad is not None
            else np.ones(error.shape[:2] + (1,))
        )
        # Accumulators are chunk-sized; a policy returning fewer rows fills a
        # prefix of them, so the per-offset table stays aligned with the
        # dataset's offsets rather than being shifted by the difference.
        total[:steps] += (error * valid).sum(axis=0)
        counts[:steps] += np.broadcast_to(valid, error.shape).sum(axis=0)

    scored = counts.sum(axis=1) > 0
    if not scored.any():
        raise SystemExit("No unpadded chunk positions were scored")
    by_offset = np.divide(
        total.sum(axis=1), counts.sum(axis=1), out=np.full(chunk, np.nan), where=scored
    )
    overall = float(total.sum() / counts.sum())

    print(f"\ncheckpoint    : {args.checkpoint}")
    print(f"policy        : {config.type}, chunk_size {chunk}, "
          f"n_action_steps {getattr(config, 'n_action_steps', '?')}")
    print(f"batches       : {min(args.max_batches, len(loader))} x {args.batch_size}")
    print("\nval L1 by chunk offset (inference path, no VAE conditioning)")
    for offset in REPORTED_OFFSETS:
        if offset < chunk and scored[offset]:
            print(f"  k={offset:<3d} : {by_offset[offset]:.3f}")
    print(f"  chunk mean : {overall:.3f}")
    print(f"  (constant {refs['constant']:.3f} | persistence {refs['persistence']:.3f})")

    names = (metadata.features["action"].get("names") or {}).get("motors") or [
        str(i) for i in range(action_dim)
    ]
    per_dim = np.divide(
        total[0], counts[0], out=np.full(action_dim, np.nan), where=counts[0] > 0
    )
    print("\nval L1 by action dimension at k=0")
    for i, name in enumerate(names):
        marker = "  <-- gripper" if "gripper" in str(name) else ""
        print(f"  {i:2d} {str(name):<20s} : {per_dim[i]:.3f}{marker}")

    report["policy"] = {
        "type": config.type,
        "chunk_size": chunk,
        "n_action_steps": getattr(config, "n_action_steps", None),
        "checkpoint": str(args.checkpoint),
    }
    report["by_offset"] = {
        str(k): (None if np.isnan(v) else float(v)) for k, v in enumerate(by_offset)
    }
    report["chunk_mean"] = overall
    report["by_dimension"] = {
        str(name): (None if np.isnan(per_dim[i]) else float(per_dim[i]))
        for i, name in enumerate(names)
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
