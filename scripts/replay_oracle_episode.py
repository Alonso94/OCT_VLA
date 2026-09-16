#!/usr/bin/env python
"""Replay one recorded oracle transfer through the real evaluation bridge.

This is the acceptance gate between controller work and policy evaluation. It
uses the recorded 15 Hz Cartesian actions, not the oracle's global planner, so
every action traverses the same socket, frame conversion, kinematic IK and
servo loop a trained checkpoint uses.

Only transfer_index=0 can be replayed from a reset: later atomic clips begin
after earlier objects have already been moved and therefore require scene state
that a fresh reset intentionally does not reproduce.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _position_error(recorded: dict, observed) -> float:
    errors = []
    for side in ("left", "right"):
        expected = recorded[side]["pose"]["position"]
        actual = getattr(observed, side).pose.position
        errors.append(math.dist(expected, actual))
    return max(errors)


def _target_position(observation) -> tuple[float, float, float] | None:
    target = observation.scene.get(observation.context.target_track_id)
    return target.pose.position if target is not None else None


def main() -> int:
    from oct_vla.core.action import Action, apply_action
    from oct_vla.core.geometry import inverse, log, multiply
    from oct_vla.robots.robotwin.native import RoboTwinNativePort
    from oct_vla.serve.client import ShelfRestockEvalClient
    from oct_vla.serve.server import ShelfRestockEvalServer

    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument(
        "--episode",
        required=True,
        type=Path,
        help="Canonical episode directory, or its episode.json file",
    )
    args = parser.parse_args()

    episode_file = (
        args.episode if args.episode.name == "episode.json" else args.episode / "episode.json"
    )
    payload = json.loads(episode_file.read_text())
    metadata = payload["metadata"]
    if str(metadata.get("transfer_index")) != "0":
        raise SystemExit(
            "Oracle replay requires transfer_index=0 because the bridge resets the scene; "
            f"got {metadata.get('transfer_index')!r}"
        )
    profile = str(metadata["task_profile"])
    seed = int(payload["seed"])
    samples = payload["samples"]
    if not samples:
        raise SystemExit(f"No samples in {episode_file}")

    def make_port(task_class: str) -> RoboTwinNativePort:
        return RoboTwinNativePort(
            args.robotwin_root,
            task_name=f"oct_vla.tasks.shelf_restock.robotwin_env:{task_class}",
            task_config="demo_clean",
        )

    port = _free_port()
    server = ShelfRestockEvalServer(make_port)
    thread = threading.Thread(
        target=lambda: server.serve_forever("127.0.0.1", port), daemon=True
    )
    thread.start()

    print(
        f"replaying {len(samples)} oracle actions: seed={seed} profile={profile} "
        f"source={episode_file}",
        flush=True,
    )
    try:
        with ShelfRestockEvalClient("127.0.0.1", port) as client:
            observation = client.reset(seed, profile)
            initial_error = _position_error(samples[0]["eef"], observation.eef)
            print(f"initial max EEF position difference: {initial_error * 1000:.2f} mm")
            result = None
            phase_errors: dict[str, float] = {}
            local_errors: dict[str, float] = {}
            rotation_errors: dict[str, float] = {}
            gripper_errors: dict[str, float] = {}
            last_phase = None
            for index, sample in enumerate(samples):
                command = Action.from_vector(sample["action"])
                requested = apply_action(observation.eef, command)
                result = client.step(sample["action"])
                phase = str(sample["phase"])
                local_error = max(
                    math.dist(getattr(requested, side).pose.position,
                              getattr(result.observation.eef, side).pose.position)
                    for side in ("left", "right")
                )
                local_errors[phase] = max(local_errors.get(phase, 0.0), local_error)
                if result.reason == "unreachable_pose":
                    print(f"ORACLE REPLAY FAILED at action {index + 1}: {result.detail}")
                    return 1
                if result.done and not result.success:
                    print(
                        f"ORACLE REPLAY FAILED at action {index + 1}: "
                        f"{result.reason} {result.detail}".rstrip()
                    )
                    return 1
                if index + 1 < len(samples):
                    error = _position_error(samples[index + 1]["eef"], result.observation.eef)
                    phase_errors[phase] = max(phase_errors.get(phase, 0.0), error)
                    expected_eef = samples[index + 1]["eef"]
                    rotation_error = max(
                        math.sqrt(sum(v * v for v in log(multiply(
                            getattr(result.observation.eef, side).pose.orientation,
                            inverse(tuple(expected_eef[side]["pose"]["orientation"])),
                        ))))
                        for side in ("left", "right")
                    )
                    rotation_errors[phase] = max(rotation_errors.get(phase, 0.0), rotation_error)
                    gripper_error = max(
                        abs(getattr(result.observation.eef, side).gripper
                            - float(expected_eef[side]["gripper"]))
                        for side in ("left", "right")
                    )
                    gripper_errors[phase] = max(gripper_errors.get(phase, 0.0), gripper_error)
                if phase != last_phase:
                    position = _target_position(result.observation)
                    print(
                        f"phase {phase!r} at action {index + 1}: "
                        f"target={tuple(round(v, 4) for v in position) if position else None}"
                    )
                    last_phase = phase
                observation = result.observation
                if result.done:
                    break

            assert result is not None
            print(
                "max EEF position error by phase (mm): "
                + ", ".join(
                    f"{phase}={error * 1000:.1f}" for phase, error in phase_errors.items()
                )
            )
            print(
                "max one-step target error by phase (mm): "
                + ", ".join(
                    f"{phase}={error * 1000:.1f}" for phase, error in local_errors.items()
                )
            )
            print(
                "max recorded-orientation error by phase (deg): "
                + ", ".join(
                    f"{phase}={math.degrees(error):.2f}"
                    for phase, error in rotation_errors.items()
                )
            )
            print(
                "max recorded-gripper error by phase: "
                + ", ".join(
                    f"{phase}={error:.3f}" for phase, error in gripper_errors.items()
                )
            )
            print(f"final target position: {_target_position(result.observation)}")
            if result.transfers_completed < 1:
                print(
                    "ORACLE REPLAY FAILED: actions completed without an IK error, "
                    "but the target was not transferred"
                )
                return 1
            print(
                f"ORACLE REPLAY PASSED: {result.transfers_completed} transfer(s), "
                f"{index + 1} actions"
            )
            return 0
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(main())

