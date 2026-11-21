"""Closed loop without a learned executor (plan stage 2b): the frozen PFN drives the wrists.

Per reset: encode one completed support once. Per step: append the *measured* scene and
both wrists to the history (never a prediction), predict the paired window, send the
first ``k`` steps through ``cartesian_absolute`` one per control step, then re-observe.
Gripper commands come from a named, non-learned source:

- ``ObservableGripper``: closes when a wrist's grip point has dwelt within
  ``CLOSE_RADIUS`` of a candidate object for ``DWELL`` steps, opens when the object it
  holds has dwelt resting at the destination height; reads only measured state and the
  window (never target ids, rule, phase or expert timing). Every trigger is logged.
- ``OracleGripper``: the server's per-arm target oracle's gripper command (privileged,
  labelled as such).

Controllers: ``PFNController`` (the frozen prior) and ``ExpertWindowController`` (the
server's per-arm oracle via-points, recomputed from the live state each step, approached
at a bounded speed: the bridge and executor check, privileged). ``HoldController`` keeps
both wrists still.

Nothing here reads ``observation.context``: that is the task manager's oracle target.
"""

from __future__ import annotations

import numpy as np

from oct_vla.bpfn.envs.controllers import PFNController
from oct_vla.bpfn.envs.robotwin.grippers import OracleGripper
from oct_vla.bpfn.envs.robotwin.history import History

ACTION_DIM = 16


def action(poses: np.ndarray, grip: np.ndarray) -> list[float]:
    """[L pos, L quat, L grip, R pos, R quat, R grip] for ``cartesian_absolute``."""
    out = []
    for a in range(2):
        q = poses[a, 3:] / np.linalg.norm(poses[a, 3:])
        out.extend([*map(float, poses[a, :3]), *map(float, q), float(grip[a])])
    assert len(out) == ACTION_DIM
    return out


def run_episode(client, controller, gripper, *, seed: int, profile: str, max_steps: int,
                post_success_steps: int = 0, privileged: bool = False) -> dict:
    """One rollout. ``privileged``: reset with the server's arm targets (needed by the
    oracle gripper and the expert-window controller; recorded in the result)."""
    obs = client.reset(seed, profile=profile, control_space="cartesian_absolute",
                       gripper_encoding="binary_command", max_steps=max_steps,
                       arm_targets=privileged, post_success_steps=post_success_steps)
    hist = History()
    hist.add(obs)
    if hasattr(controller, "reset"):
        controller.reset()
    trace = {"commands": [], "measured": [], "grip": []}
    result = None
    for step in range(max_steps):
        ep = hist.episode()
        poses = (controller(ep) if isinstance(controller, PFNController) else controller(ep, obs))
        grip = gripper(obs, step) if isinstance(gripper, OracleGripper) else gripper(ep, step)
        trace["commands"].append(poses.astype(np.float32))
        trace["grip"].append(grip.astype(np.float32))
        trace["measured"].append(ep.wrist[-1].astype(np.float32))
        result = client.step(action(poses, grip))
        obs = result.observation
        hist.add(obs)
        if result.done:
            break
    final = hist.episode()
    return {"seed": seed, "profile": profile, "success": bool(result.success),
            "steps": step + 1, "reason": result.reason, "success_at": result.success_at,
            "still_restocked": result.still_restocked,
            "distractors_preserved": result.distractors_preserved,
            "transfers_completed": result.transfers_completed,
            "objects_lifted": result.objects_lifted, "infeasible_steps": result.infeasible_steps,
            "events": result.events, "gripper_log": gripper.log, "privileged": privileged,
            "colours": [int(c) for c in final.colour], "tracks": final.track_ids,
            "trace": {k: np.stack(v) for k, v in trace.items()},
            "objects": final.objects.astype(np.float32)}
