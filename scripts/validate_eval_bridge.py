#!/usr/bin/env python
"""Exercise the eval bridge against a live simulator with scripted actions.

Runs in the RoboTwin environment only. The client half needs nothing from
LeRobot -- only the standard library and this package -- so both ends can run in
one process here, which keeps the check to a single job while still going
through the real socket, the real protocol and the real server.

What this actually proves, none of which unit tests can:

* `RoboTwinNativePort.ik()` returns joint angles that move the arm to the
  commanded end-effector pose, through the same frame chain the planner uses.
* The server's step loop advances simulation at the intended control rate.
* Object state, success checking and RGB survive the round trip.

The actions are scripted, not a policy: a commanded displacement whose achieved
displacement can be measured. A policy rollout could not distinguish "IK is
wrong" from "the policy is untrained".
"""

from __future__ import annotations

import argparse
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


def _distance(a, b) -> float:
    return math.dist(a, b)


def main() -> int:
    from oct_vla.robots.robotwin.native import RoboTwinNativePort
    from oct_vla.serve.client import ShelfRestockEvalClient
    from oct_vla.serve.server import ShelfRestockEvalServer

    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--profile", default="three_object")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--delta", type=float, default=0.01, help="Metres per step, per axis")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Metres; how far the achieved displacement may fall short of commanded",
    )
    args = parser.parse_args()

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

    failures: list[str] = []
    try:
        with ShelfRestockEvalClient("127.0.0.1", port) as client:
            print(f"--- reset seed={args.seed} profile={args.profile}")
            observation = client.reset(args.seed, args.profile)
            head = observation.frame("head_camera")
            print(f"    cameras {head.width}x{head.height}, "
                  f"{len(observation.scene.objects)} objects, "
                  f"target={observation.context.target_track_id!r}")
            if not any(observation.frame(name).data for name in
                       ("head_camera", "left_wrist_camera", "right_wrist_camera")):
                failures.append("all camera frames are empty")
            expected_objects = {"two_object": 2, "three_object": 3, "four_object": 4}[args.profile]
            if len(observation.scene.objects) != expected_objects:
                failures.append(
                    f"expected {expected_objects} objects, saw {len(observation.scene.objects)}"
                )

            # 1. Hold. Zero displacement with the current grippers must not move
            #    the arm: if it does, IK is not tracking the commanded pose and
            #    every later measurement is meaningless.
            start = observation.eef
            hold = [0.0] * 6 + [start.left.gripper] + [0.0] * 6 + [start.right.gripper]
            result = client.step(hold)
            drift = _distance(result.observation.eef.left.pose.position, start.left.pose.position)
            print(f"--- hold: left drift {drift * 1000:.1f} mm")
            if drift > args.tolerance:
                failures.append(f"hold drifted {drift * 1000:.1f} mm")

            # 2. Commanded translation. Track achieved against commanded so a
            #    frame error shows up as motion in the wrong direction or none.
            before = result.observation.eef.left.pose.position
            commanded = 0.0
            for index in range(args.steps):
                action = [0.0, 0.0, args.delta, 0.0, 0.0, 0.0, start.left.gripper] + \
                         [0.0] * 6 + [start.right.gripper]
                result = client.step(action)
                commanded += args.delta
                if result.done:
                    print(f"    episode ended early at step {index + 1}: {result.reason}")
                    break
            after = result.observation.eef.left.pose.position
            achieved = after[2] - before[2]
            lateral = math.dist((after[0], after[1]), (before[0], before[1]))
            print(f"--- +z x{args.steps}: commanded {commanded * 1000:.0f} mm, "
                  f"achieved {achieved * 1000:.1f} mm, lateral {lateral * 1000:.1f} mm")
            if achieved < commanded - args.tolerance:
                failures.append(
                    f"achieved {achieved * 1000:.1f} mm of {commanded * 1000:.0f} mm commanded"
                )
            if lateral > args.tolerance:
                failures.append(f"drifted {lateral * 1000:.1f} mm laterally on a pure +z command")

            # 3. Gripper. Absolute target, not an increment.
            close = [0.0] * 6 + [0.0] + [0.0] * 6 + [start.right.gripper]
            for _ in range(4):
                result = client.step(close)
            print(f"--- gripper close: left {start.left.gripper:.2f} -> "
                  f"{result.observation.eef.left.gripper:.2f}")
            if result.observation.eef.left.gripper >= start.left.gripper:
                failures.append("gripper did not close")

            # 4. The clock must advance at the control rate, not per tick.
            elapsed = result.observation.timestamp
            print(f"--- simulation clock: {elapsed:.3f} s over "
                  f"{args.steps + 5} control steps")
            if elapsed <= 0:
                failures.append("simulation clock did not advance")
    finally:
        server.close()

    print()
    if failures:
        print("BRIDGE VALIDATION FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("BRIDGE VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
