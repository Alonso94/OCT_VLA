#!/usr/bin/env python
"""Replay one recorded oracle transfer as JOINT actions through the eval bridge.

The acceptance gate for the joint control space, and the counterpart of
`replay_oracle_episode.py`. It sends the demonstration's own recorded joint
configurations as absolute targets, over the same socket and servo loop a
trained checkpoint uses -- but with no inverse kinematics anywhere in the path.

Absolute targets are why this should track far better than the Cartesian replay
did. There, each action was an increment applied to wherever the arm actually
was, so tracking error compounded with no feedback to correct it. Here every
target states where the arm *should be* at that step, so a shortfall at one
step is corrected by the next rather than carried forward.

Only transfer_index=0 can be replayed from a reset: later clips begin after
earlier objects have moved, which a fresh reset deliberately does not reproduce.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _joint_vector(record: dict) -> list[float]:
    """JointState.to_vector layout, from the stored episode JSON."""
    return [
        *record["left"]["positions"], record["left"]["gripper"],
        *record["right"]["positions"], record["right"]["gripper"],
    ]


def _joint_error(expected: list[float], observed) -> tuple[float, float]:
    """(max arm-joint error in radians, max gripper error), ignoring grippers
    in the first figure because they are a different unit and scale."""
    actual = [
        *observed.left.positions, observed.left.gripper,
        *observed.right.positions, observed.right.gripper,
    ]
    half = len(expected) // 2
    arm_indices = [i for i in range(len(expected)) if i not in (half - 1, 2 * half - 1)]
    grip_indices = (half - 1, 2 * half - 1)
    return (
        max(abs(expected[i] - actual[i]) for i in arm_indices),
        max(abs(expected[i] - actual[i]) for i in grip_indices),
    )


def main() -> int:
    from oct_vla.robots.robotwin.native import RoboTwinNativePort
    from oct_vla.serve.client import ShelfRestockEvalClient
    from oct_vla.serve.server import ShelfRestockEvalServer

    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Radians; the largest per-joint tracking error the gate accepts.",
    )
    args = parser.parse_args()

    episode_file = (
        args.episode if args.episode.name == "episode.json" else args.episode / "episode.json"
    )
    payload = json.loads(episode_file.read_text())
    metadata = payload["metadata"]
    if str(metadata.get("transfer_index")) != "0":
        raise SystemExit(
            "Joint replay requires transfer_index=0 because the bridge resets the scene; "
            f"got {metadata.get('transfer_index')!r}"
        )
    samples = payload["samples"]
    if not samples or samples[0].get("joints") is None:
        raise SystemExit(f"{episode_file} has no recorded joints; re-collect it.")
    profile, seed = str(metadata["task_profile"]), int(payload["seed"])

    def make_port(task_class: str) -> RoboTwinNativePort:
        return RoboTwinNativePort(
            args.robotwin_root,
            task_name=f"oct_vla.tasks.shelf_restock.robotwin_env:{task_class}",
            task_config="demo_clean",
        )

    port = _free_port()
    server = ShelfRestockEvalServer(make_port)
    threading.Thread(target=lambda: server.serve_forever("127.0.0.1", port), daemon=True).start()

    print(f"replaying {len(samples)} joint targets: seed={seed} profile={profile}", flush=True)
    worst_arm = worst_grip = 0.0
    by_phase: dict[str, float] = {}
    with ShelfRestockEvalClient("127.0.0.1", port) as client:
        observation = client.reset(seed, profile, control_space="joint")
        start_arm, _ = _joint_error(_joint_vector(samples[0]["joints"]), observation.joints)
        print(f"initial max joint difference: {start_arm:.5f} rad")
        result = None
        for index, sample in enumerate(samples):
            # The action is the NEXT configuration -- an absolute target, exactly
            # as the exporter writes action.joint_position.
            following = min(index + 1, len(samples) - 1)
            target = _joint_vector(samples[following]["joints"])
            result = client.step(target)
            arm_error, grip_error = _joint_error(target, result.observation.joints)
            phase = str(sample["phase"])
            by_phase[phase] = max(by_phase.get(phase, 0.0), arm_error)
            worst_arm = max(worst_arm, arm_error)
            worst_grip = max(worst_grip, grip_error)
            if result.done:
                break

    assert result is not None
    print("max joint tracking error by phase (rad): "
          + ", ".join(f"{p}={e:.4f}" for p, e in by_phase.items()))
    print(f"worst joint error: {worst_arm:.5f} rad | worst gripper error: {worst_grip:.4f}")
    print(f"infeasible steps: {result.infeasible_steps} (joint space cannot produce any)")
    print(f"transfers completed: {result.transfers_completed} | success: {result.success}")

    if result.infeasible_steps:
        print("JOINT REPLAY FAILED: the joint path must never call IK")
        return 1
    if worst_arm > args.tolerance:
        print(f"JOINT REPLAY FAILED: joint error {worst_arm:.4f} rad exceeds {args.tolerance}")
        return 1
    if result.transfers_completed < 1:
        print("JOINT REPLAY FAILED: the demonstration completed no transfer")
        return 1
    print(f"JOINT REPLAY PASSED: {result.transfers_completed} transfer(s), {len(samples)} actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
