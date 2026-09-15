#!/usr/bin/env python
"""Score a trained policy closed-loop against a running simulator server.

LeRobot policy environment only: this talks to RoboTwin over the bridge
(`oct_vla.serve`) and never imports it. Start scripts/robotwin_eval_server.py in
the simulator environment first.

Object count is the generalization axis: the same policy is rolled out on the
two-, three- and four-object profiles, so a drop can be attributed to scene
density rather than to any change in the policy or its conditioning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Bridge camera name -> dataset feature key, matching data/lerobot_export.py.
#: A mismatch here would feed the policy its wrist view where it expects the
#: head view, which degrades silently rather than raising.
CAMERA_FEATURES = {
    "head_camera": "observation.images.head",
    "left_wrist_camera": "observation.images.left_wrist",
    "right_wrist_camera": "observation.images.right_wrist",
}


def state_vector(eef) -> list[float]:
    """Exactly `lerobot_export._state_vector`'s layout: left pos/quat/gripper
    then right. Rebuilt from the same fields so evaluation conditions the
    policy on the vector layout it was trained on."""
    return [
        *eef.left.pose.position,
        *eef.left.pose.orientation,
        eef.left.gripper,
        *eef.right.pose.position,
        *eef.right.pose.orientation,
        eef.right.gripper,
    ]


def build_observation(obs, *, object_token_spec, torch, np):
    from oct_vla.data.object_tokens import object_tokens

    batch = {}
    for camera, feature in CAMERA_FEATURES.items():
        frame = obs.frame(camera)
        image = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        # uint8 CHW with a leading batch axis: what LeRobotDataset yields with
        # return_uint8=True, which is how the training dataloader was built.
        batch[feature] = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
    batch["observation.state"] = torch.tensor(
        [state_vector(obs.eef)], dtype=torch.float32
    )
    if object_token_spec is not None:
        tokens, mask = object_tokens(obs.scene, obs.context, spec=object_token_spec)
        batch["observation.object_tokens"] = torch.tensor([tokens], dtype=torch.float32)
        batch["observation.object_token_mask"] = torch.tensor(
            [[float(v) for v in mask]], dtype=torch.float32
        )
    batch["task"] = [obs.context.instruction]
    return batch


def run_episode(
    client, policy, preprocessor, postprocessor, *,
    seed, profile, max_steps, object_token_spec, torch, np,
) -> dict:
    observation = client.reset(seed, profile)
    policy.reset()
    result = {"seed": seed, "profile": profile, "success": False,
              "steps": 0, "transfers_completed": 0, "reason": "step_limit"}
    for step in range(max_steps):
        batch = build_observation(
            observation, object_token_spec=object_token_spec, torch=torch, np=np
        )
        with torch.no_grad():
            action = policy.select_action(preprocessor(batch))
        action = postprocessor(action)
        vector = action.squeeze(0).float().cpu().tolist()
        outcome = client.step(vector)
        observation = outcome.observation
        result.update(
            steps=step + 1,
            transfers_completed=outcome.transfers_completed,
            success=outcome.success,
            reason=outcome.reason or result["reason"],
        )
        if outcome.done:
            break
    return result


def main() -> int:
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors

    from oct_vla.data.object_tokens import ObjectTokenSpec
    from oct_vla.data.policy_inputs import rename_map_for
    from oct_vla.serve.client import ShelfRestockEvalClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained policy directory")
    parser.add_argument(
        "--dataset-root", required=True, type=Path, help="Dataset the policy was fit on"
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--profiles", default="two_object,three_object,four_object")
    parser.add_argument("--seeds", required=True, help="Comma-separated, or FIRST-LAST")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--object-tokens", action="store_true")
    parser.add_argument(
        "--shuffle-tokens",
        action="store_true",
        help="Permute object tokens across objects. The control for 'does the "
        "policy use the tokens at all?': a policy that reads them degrades, one "
        "that ignores them scores the same. Evaluation only -- it overrides the "
        "checkpoint's own setting and never changes the weights.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        first, last = args.seeds.split("-", maxsplit=1)
        seeds = list(range(int(first), int(last) + 1))
    else:
        seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]

    # Registers any policy plugin this repo defines (control_pi05) before the
    # checkpoint's own config is resolved. LeRobot maps a saved policy type to
    # a class through its draccus registry, and a plugin only enters that
    # registry when its module is imported -- so without this an
    # object-conditioned checkpoint fails to load with an unknown-type error,
    # while an RGB checkpoint is unaffected.
    import oct_vla.policies  # noqa: F401

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = args.checkpoint
    # Derived from the checkpoint, not assumed: pi0.5 needs openpi's camera
    # names, SmolVLA takes the dataset's own, and hard-coding either one would
    # silently feed the other backbone a wrist view as its scene view.
    rename_map = rename_map_for(config.type)
    if args.shuffle_tokens:
        if not hasattr(config, "object_token_shuffle"):
            raise SystemExit(
                "--shuffle-tokens needs an object-conditioned checkpoint; "
                f"{args.checkpoint} is a {getattr(config, 'type', 'unknown')} policy, "
                "which has no object tokens to shuffle."
            )
        config.object_token_shuffle = True
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    policy = make_policy(cfg=config, ds_meta=metadata, rename_map=rename_map)
    policy.eval()
    # The same rename map training used. Unlike training, inference would not
    # complain about getting this wrong: the cameras share shape and dtype, so a
    # wrong mapping just feeds the policy a wrist view as its scene view and
    # shows up only as a mysteriously poor success rate.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    spec = ObjectTokenSpec() if args.object_tokens else None

    results = []
    with ShelfRestockEvalClient(args.host, args.port) as client:
        for profile in profiles:
            for seed in seeds:
                outcome = run_episode(
                    client, policy, preprocessor, postprocessor,
                    seed=seed, profile=profile, max_steps=args.max_steps,
                    object_token_spec=spec, torch=torch, np=np,
                )
                results.append(outcome)
                print(json.dumps(outcome), flush=True)

    summary = {}
    for profile in profiles:
        rows = [r for r in results if r["profile"] == profile]
        if not rows:
            continue
        summary[profile] = {
            "episodes": len(rows),
            "success_rate": sum(r["success"] for r in rows) / len(rows),
            "mean_transfers": sum(r["transfers_completed"] for r in rows) / len(rows),
        }
    report = {
        "checkpoint": str(args.checkpoint),
        "object_tokens": args.object_tokens,
        # Recorded, not inferred later from the checkpoint path: the shuffled
        # control reuses arm B's weights, so the path alone cannot distinguish
        # the two and aggregation would silently merge them into one arm.
        "token_mode": getattr(config, "object_token_mode", None),
        "shuffled_tokens": bool(args.shuffle_tokens),
        "max_steps": args.max_steps,
        "seeds": seeds,
        "summary": summary,
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
