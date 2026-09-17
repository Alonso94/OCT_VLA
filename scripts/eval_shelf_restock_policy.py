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


def joint_vector(joints) -> list[float]:
    """`JointState.to_vector` layout: each arm's joints then its gripper.

    Exactly what the joint control space exports as `observation.state`, so the
    policy is conditioned on the vector it was fit on.
    """
    return [
        *joints.left.positions,
        joints.left.gripper,
        *joints.right.positions,
        joints.right.gripper,
    ]


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


def build_observation(
    obs, *, object_token_spec, torch, np, ranks=None, control_space="cartesian",
    privileged=False, uint8_images=True,
):
    from oct_vla.data.object_tokens import object_token_ranks, object_tokens

    batch = {}
    for camera, feature in CAMERA_FEATURES.items():
        frame = obs.frame(camera)
        image = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
        # Image dtype follows the policy's own VISUAL normalisation, not a fixed
        # convention. pi0.5 declares IDENTITY and takes the uint8 frames the
        # dataloader yields with return_uint8=True; ACT declares MEAN_STD and
        # normalises in place, which overflows a uint8 tensor outright. Reading
        # it from the checkpoint means a new backbone cannot silently be handed
        # the wrong one.
        batch[feature] = tensor if uint8_images else tensor.float().div_(255.0)
    if control_space in ("joint", "joint_delta"):
        if obs.joints is None:
            raise SystemExit(
                "The simulator reported no joint state, but this checkpoint is "
                "joint-space. Its observation.state cannot be built."
            )
        state = joint_vector(obs.joints)
    else:
        state = state_vector(obs.eef)
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32)
    if object_token_spec is not None:
        tokens, mask = object_tokens(obs.scene, obs.context, spec=object_token_spec)
        batch["observation.object_tokens"] = torch.tensor([tokens], dtype=torch.float32)
        batch["observation.object_token_mask"] = torch.tensor(
            [[float(v) for v in mask]], dtype=torch.float32
        )
        if ranks is not None:
            # The same episode-stable ordering the exporter wrote, rebuilt here
            # from this episode's first scene. Training and evaluation must sort
            # the role-stripped arm identically or the model sees a layout it
            # was never fit on.
            batch["observation.object_token_rank"] = torch.tensor(
                [list(object_token_ranks(
                    obs.scene, ranks, spec=object_token_spec, context=obs.context
                ))],
                dtype=torch.float32,
            )
    if privileged:
        # Flattened object tokens, exactly as the privileged export writes them.
        # The vision-free policy reads only this and observation.state; the
        # camera tensors above are ignored because its config declares no image
        # features, so they are left in place rather than special-cased out.
        tokens, _ = object_tokens(obs.scene, obs.context, spec=object_token_spec)
        batch["observation.environment_state"] = torch.tensor(
            [[value for token in tokens for value in token]], dtype=torch.float32
        )
    batch["task"] = [obs.context.instruction]
    return batch


def run_episode(
    client, policy, preprocessor, postprocessor, *,
    seed, profile, max_steps, object_token_spec, torch, np, control_space="cartesian",
    privileged=False, uint8_images=True,
) -> dict:
    observation = client.reset(seed, profile, control_space=control_space)
    policy.reset()
    # Fixed once, from the scene at reset -- exactly as the exporter does.
    ranks = None
    if object_token_spec is not None:
        from oct_vla.data.object_tokens import stable_ranks

        ranks = stable_ranks(observation.scene)
    result = {
        "seed": seed, "profile": profile, "success": False, "steps": 0,
        "transfers_completed": 0, "reason": "step_limit", "detail": "",
        "infeasible_steps": 0, "objects_lifted": 0, "objects_total": 0,
    }
    for step in range(max_steps):
        batch = build_observation(
            observation,
            object_token_spec=object_token_spec,
            torch=torch,
            np=np,
            ranks=ranks,
            control_space=control_space,
            privileged=privileged,
            uint8_images=uint8_images,
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
            detail=outcome.detail or result["detail"],
            infeasible_steps=outcome.infeasible_steps,
            objects_lifted=max(result["objects_lifted"], outcome.objects_lifted),
            objects_total=outcome.objects_total or result["objects_total"],
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
    # Derived from the dataset the checkpoint was fit on, not passed in: a
    # joint-space policy emits absolute joint targets and a Cartesian one emits
    # 14-d increments, and running either through the other's path would be
    # silently wrong rather than an error.
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    action_names = (metadata.features["action"].get("names") or {}).get("motors") or []
    if any("_arm.j" in str(n) for n in action_names):
        # Absolute and incremental joint actions share a layout, so the column
        # names cannot tell them apart. The dataset records which it holds.
        info_path = args.dataset_root / "meta" / "info.json"
        recorded = json.loads(info_path.read_text()).get("control_space")
        if recorded is None:
            raise SystemExit(
                f"{info_path} records no control_space, but this dataset's action "
                "is joint-shaped. Absolute and incremental joint actions look "
                "identical in the schema, and executing one as the other is "
                "silently wrong -- re-export so the dataset says which it is."
            )
        control_space = str(recorded)
    else:
        control_space = "cartesian"
    width = metadata.features["action"]["shape"][0]
    # A vision-free checkpoint declares environment_state and no image features.
    privileged = "observation.environment_state" in metadata.features
    visual_norm = str((config.normalization_mapping or {}).get("VISUAL", "IDENTITY"))
    uint8_images = visual_norm.upper().endswith("IDENTITY")
    print(
        f"control space: {control_space} (action width {width}) "
        f"privileged={privileged} visual_norm={visual_norm} uint8_images={uint8_images}"
    )
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
    spec = ObjectTokenSpec() if (args.object_tokens or privileged) else None

    results = []
    with ShelfRestockEvalClient(args.host, args.port) as client:
        for profile in profiles:
            for seed in seeds:
                outcome = run_episode(
                    client, policy, preprocessor, postprocessor,
                    seed=seed, profile=profile, max_steps=args.max_steps,
                    object_token_spec=spec, torch=torch, np=np,
                    control_space=control_space, privileged=privileged,
                    uint8_images=uint8_images,
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
            "mean_lifted": sum(r["objects_lifted"] for r in rows) / len(rows),
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
