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


def joint_vector(joints, previous: list[float] | None = None) -> list[float]:
    """`JointState.to_vector` layout: each arm's joints then its gripper.

    Exactly what the joint control space exports as `observation.state`, so the
    policy is conditioned on the vector it was fit on.

    With `previous` supplied, the backward difference is appended, mirroring
    `lerobot_export._joint_state_vector` under `state_encoding="position_velocity"`.
    The caller passes the *previous frame's positions*, or None on the first
    step of an episode, where the exporter writes zeros because the first
    recorded frame has no predecessor either.
    """
    positions = [
        *joints.left.positions,
        joints.left.gripper,
        *joints.right.positions,
        joints.right.gripper,
    ]
    if previous is None:
        return positions
    if len(previous) != len(positions):
        raise ValueError(
            f"previous state has {len(previous)} entries but the arm reports "
            f"{len(positions)}; the velocity block would be misaligned"
        )
    return [*positions, *(now - was for now, was in zip(positions, previous, strict=True))]


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
    privileged=False, uint8_images=True, state_encoding="position", previous_joints=None,
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
        if state_encoding == "position_velocity":
            # On the first step of an episode there is no predecessor, so the
            # velocity block is zero -- which is what the exporter writes for a
            # recording's first frame. Passing the current positions as the
            # previous ones produces exactly that, rather than special-casing
            # the width here and risking the two sides disagreeing.
            positions = joint_vector(obs.joints)
            state = joint_vector(
                obs.joints, previous_joints if previous_joints is not None else positions
            )
        else:
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
    privileged=False, uint8_images=True, state_encoding="position", video=None,
    gripper_encoding="measured_aperture",
) -> dict:
    observation = client.reset(seed, profile, control_space=control_space,
                               gripper_encoding=gripper_encoding)
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
    # Carried across steps so the velocity block is a backward difference of
    # consecutive observations, the same quantity the exporter differenced.
    previous_joints = None
    # The opening frame, before any action: without it the video starts one
    # step in and the scene's initial layout is never shown.
    if video is not None:
        video.add(observation, {"step": 0, "transfers": 0, "lifted": 0})
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
            state_encoding=state_encoding,
            previous_joints=previous_joints,
        )
        if observation.joints is not None:
            previous_joints = joint_vector(observation.joints)
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
        if video is not None:
            video.add(observation, {
                "step": step + 1,
                "transfers": result["transfers_completed"],
                "lifted": result["objects_lifted"],
            })
        if outcome.done:
            break
    if video is not None:
        result["video"] = str(video.close() or "")
        result["video_frames"] = video.frames
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
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Record each episode as an mp4 here: head and both wrist views side "
        "by side, captioned. The counters separate 'did nothing' from 'did the "
        "task' but not 'reached and missed' from 'never approached'.",
    )
    parser.add_argument(
        "--video-label",
        default="",
        help="Banner burned into the video, naming the cell it belongs to.",
    )
    parser.add_argument(
        "--video-seeds",
        default="",
        help="Comma-separated subset of --seeds to record. Empty records every "
        "evaluated episode, which is rarely what is wanted: a full sweep would "
        "write one file per episode per profile.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        help="How many actions to execute per forward pass. Inference-only for "
        "chunked policies -- it never enters the training loss -- so this "
        "re-scores an existing checkpoint rather than needing a retrain. 1 is "
        "fully closed-loop; the trained default of 50 is 3.3 s of open loop at "
        "15 Hz.",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        help="Enable ACT temporal ensembling, blending overlapping chunks. "
        "Requires --n-action-steps 1.",
    )
    parser.add_argument(
        "--video-name",
        default="",
        help="Filename stem for the recording, without .mp4. One stable name per "
        "cell means a re-record overwrites rather than accumulates, which is the "
        "difference between a fixed-size directory and one that grows every time "
        "the grid is refreshed. Default names the file after the profile and seed.",
    )
    parser.add_argument("--video-fps", type=float, default=15.0,
                        help="Matches the 15 Hz control rate, so the video runs in real time.")
    parser.add_argument("--video-crf", type=int, default=30,
                        help="x264 quality; higher is smaller. These are committed to the "
                             "repository, so the default trades detail for size.")
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
    info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    # Read the recorded fields once, for every layout. They used to be read per
    # action-name branch, and a dataset the names did not classify -- the 14-d
    # EE-delta view carries no motor names at all -- fell through to a branch
    # that hardcoded `measured_aperture`, silently decoding a {0, 1} command
    # against an aperture threshold.
    recorded_space = info.get("control_space")
    state_encoding = str(info.get("state_encoding", "position"))
    # Safe to default: an absent field can only mean a dataset exported before
    # the encoding existed, and those hold the raw measurement.
    gripper_encoding = str(info.get("gripper_encoding", "measured_aperture"))

    if any("_arm.j" in str(n) for n in action_names):
        # Absolute and incremental joint actions share a layout, so the column
        # names cannot tell them apart. The dataset records which it holds, and
        # guessing wrong is silently wrong rather than loud.
        if recorded_space is None:
            raise SystemExit(
                f"{args.dataset_root / 'meta' / 'info.json'} records no "
                "control_space, but this dataset's action is joint-shaped. "
                "Absolute and incremental joint actions look identical in the "
                "schema, and executing one as the other is silently wrong -- "
                "re-export so the dataset says which it is."
            )
        control_space = str(recorded_space)
    else:
        # End-effector space: a 16-d pose is absolute, the 14-d canonical action
        # is an increment. `cartesian` is the increment and the older default.
        control_space = str(recorded_space or "cartesian")
        state_encoding = "position"
    width = metadata.features["action"]["shape"][0]
    # Whether to build observation.environment_state at all. Not the same as
    # "vision-free": a dataset can carry both the cameras and the flattened
    # object tokens, which is the RGB + objects arm. The builder writes the
    # camera tensors either way and the policy's own config decides what it
    # reads, so this only has to answer "is environment_state expected".
    env_state = "observation.environment_state" in metadata.features
    privileged = env_state
    has_cameras = any(k.startswith("observation.images.") for k in metadata.features)
    visual_norm = str((config.normalization_mapping or {}).get("VISUAL", "IDENTITY"))
    # Never uint8, whatever the normalization mapping says. Training converts
    # every camera key to float32/255 for *every* policy before the processor
    # runs (lerobot_train.py:_preprocess_dataset_batch), so handing a policy
    # uint8 here feeds it inputs 255x the scale it was fit on.
    #
    # This was not hypothetical and it was not loud. A VISUAL=IDENTITY mapping
    # used to select the uint8 path, which covers pi0.5, SmolVLA and GR00T.
    # SmolVLA crashes on it (`upsample_bilinear2d` has no Byte kernel), so it
    # announced itself -- but pi0.5 casts to float without dividing and then
    # computes `img * 2 - 1`, landing in [-1, 509] instead of [-1, 1] and
    # scoring zero without ever complaining. Every pi0.5 rollout recorded
    # before this fix ran on those inputs.
    uint8_images = False
    state_width = metadata.features["observation.state"]["shape"][0]
    print(
        f"control space: {control_space} (action width {width}) "
        f"state: {state_encoding} (width {state_width}) gripper: {gripper_encoding} "
        f"env_state={env_state} cameras={has_cameras} "
        f"visual_norm={visual_norm} uint8_images={uint8_images}"
    )
    # Applied before make_policy, and that ordering is load-bearing: the policy
    # sizes its action queue from n_action_steps in reset(), and only builds a
    # temporal ensembler at all if the coefficient is set at construction time.
    base_n = getattr(config, "n_action_steps", None)
    if args.n_action_steps is not None or args.temporal_ensemble_coeff is not None:
        if base_n is None:
            raise SystemExit(f"{config.type} has no n_action_steps to override")
        chosen = args.n_action_steps if args.n_action_steps is not None else base_n
        # Assigning to the dataclass does not re-run __post_init__, so the
        # policy's own guards never fire. They are restated rather than trusted.
        if not 1 <= chosen <= config.chunk_size:
            raise SystemExit(
                f"--n-action-steps must be within [1, chunk_size={config.chunk_size}]; "
                f"got {chosen}"
            )
        if args.temporal_ensemble_coeff is not None and chosen != 1:
            raise SystemExit(
                "Temporal ensembling needs --n-action-steps 1: the policy has to be "
                f"queried every step to form the ensemble. Got {chosen}."
            )
        config.n_action_steps = chosen
        if args.temporal_ensemble_coeff is not None:
            config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
        print(f"execution horizon: {base_n} -> {config.n_action_steps} "
              f"(temporal_ensemble_coeff={getattr(config, 'temporal_ensemble_coeff', None)})")

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

    record = set(seeds)
    if args.video_dir and args.video_seeds:
        record = {int(v) for v in args.video_seeds.split(",") if v.strip()}
        unknown = record - set(seeds)
        if unknown:
            raise SystemExit(
                f"--video-seeds names {sorted(unknown)}, which --seeds does not "
                "evaluate; nothing would be recorded for them."
            )

    results = []
    with ShelfRestockEvalClient(args.host, args.port) as client:
        for profile in profiles:
            for seed in seeds:
                video = None
                if args.video_dir and seed in record:
                    from oct_vla.serve.video import Caption, RolloutVideo

                    stem = args.video_name or f"{profile}_seed{seed}"
                    video = RolloutVideo(
                        args.video_dir / f"{stem}.mp4",
                        fps=args.video_fps,
                        crf=args.video_crf,
                        caption=Caption(
                            args.video_label or args.checkpoint.parent.parent.parent.name,
                            f"{control_space} | {profile} | seed {seed}",
                        ),
                    )
                outcome = run_episode(
                    client, policy, preprocessor, postprocessor,
                    seed=seed, profile=profile, max_steps=args.max_steps,
                    object_token_spec=spec, torch=torch, np=np,
                    control_space=control_space, privileged=privileged,
                    uint8_images=uint8_images, state_encoding=state_encoding,
                    gripper_encoding=gripper_encoding, video=video,
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
        # Same weights, different inference settings, so aggregation must not
        # merge them. `eval_tag` is empty when nothing was overridden, which
        # keeps a default re-run comparable with reports written before this
        # existed.
        "eval_tag": "_".join(
            part for part in (
                f"n{config.n_action_steps}" if (
                    base_n is not None and config.n_action_steps != base_n
                ) else "",
                f"te{args.temporal_ensemble_coeff:g}"
                if args.temporal_ensemble_coeff is not None else "",
            ) if part
        ),
        "execution": {
            "n_action_steps": getattr(config, "n_action_steps", None),
            "trained_n_action_steps": base_n,
            "chunk_size": getattr(config, "chunk_size", None),
            "temporal_ensemble_coeff": getattr(config, "temporal_ensemble_coeff", None),
        },
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
