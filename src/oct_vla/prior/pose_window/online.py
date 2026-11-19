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

from dataclasses import dataclass, field

import numpy as np

from . import quat as Q
from .metrics import LOWER_Z
from .schema import (FIXTURE, MOVABLE, Episode, category_id, colour_id, query_table,
                     support_table)
from .trajectory import grip_point

ACTION_DIM = 16
CLOSE_RADIUS = 0.02     # m: grip point to object centre
OPEN_HEIGHT = 0.86      # m: an object centre above this has reached the destination band
DWELL = 3               # control steps
MAX_STEP = 0.015        # m per control step for the expert-window tracker
MAX_TURN = 0.08         # rad per control step


def _pose7(pose) -> np.ndarray:
    return np.array([*pose.position, *pose.orientation], float)


@dataclass
class History:
    """Measured rows since reset, as a growing physical ``Episode`` (wrists, objects in
    first-seen track order, shelf decks as fixtures)."""

    times: list = field(default_factory=list)
    wrists: list = field(default_factory=list)
    apertures: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    tracks: tuple = ()
    header: dict = field(default_factory=dict)

    def add(self, obs) -> None:
        if not self.tracks:
            objs = obs.scene.objects
            self.tracks = tuple(o.track_id for o in objs)
            from oct_vla.data.entity_tokens import shelf_support_entities
            from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC

            decks = shelf_support_entities(DEFAULT_SPEC)
            self.header = {
                "kind": [MOVABLE] * len(objs) + [FIXTURE] * len(decks),
                "colour": [colour_id(o.colour) for o in objs] + [0] * len(decks),
                "category": [category_id(_category(o.category)) for o in objs] + [0] * len(decks),
                "size": [tuple(o.size_xyz) for o in objs] + [tuple(d.size_xyz) for d in decks],
                "decks": [_pose7(d.pose) for d in decks],
            }
        by_id = {o.track_id: o for o in obs.scene.objects}
        row = [(_pose7(by_id[t].pose) if t in by_id else self.objects[-1][i])
               for i, t in enumerate(self.tracks)]
        self.objects.append(np.array(row + self.header["decks"]))
        self.wrists.append(np.stack([_pose7(obs.eef.left.pose), _pose7(obs.eef.right.pose)]))
        self.apertures.append([obs.eef.left.gripper, obs.eef.right.gripper])
        self.times.append(float(obs.timestamp))

    def episode(self) -> Episode:
        wr = np.stack(self.wrists)
        wr[..., 3:] = np.stack([Q.continuous(Q.normalize(wr[:, a, 3:])) for a in range(2)], 1)
        ob = np.stack(self.objects)
        ob[..., 3:] = np.stack([Q.continuous(Q.normalize(ob[:, j, 3:])) for j in range(ob.shape[1])], 1)
        h = self.header
        return Episode(np.asarray(self.times), wr, np.asarray(self.apertures, float), ob,
                       np.asarray(h["kind"], np.int8), np.asarray(h["colour"], np.int8),
                       np.asarray(h["category"], np.int8), np.asarray(h["size"], float),
                       self.tracks + ("lower_deck", "upper_deck"), {"source": "live"})


def _category(asset: str | None) -> str | None:
    from .real import CATEGORY_OF_ASSET

    return CATEGORY_OF_ASSET.get(asset)


class ObservableGripper:
    """Close on dwell at an object, open on dwell at the destination height; logged."""

    def __init__(self) -> None:
        self.closed = [False, False]
        self.dwell = [0, 0]
        self.held: list[int | None] = [None, None]
        self.log: list[dict] = []

    def __call__(self, ep: Episode, step: int) -> np.ndarray:
        last = ep.T - 1
        grips = grip_point(ep.wrist[last])
        for a in range(2):
            movable = [j for j in range(len(ep.track_ids)) if ep.kind[j] == MOVABLE]
            if not self.closed[a]:
                near = [j for j in movable if ep.objects[last, j, 2] < LOWER_Z
                        and np.linalg.norm(ep.objects[last, j, :3] - grips[a]) < CLOSE_RADIUS]
                self.dwell[a] = self.dwell[a] + 1 if near else 0
                if self.dwell[a] >= DWELL:
                    self.closed[a], self.held[a], self.dwell[a] = True, near[0], 0
                    self.log.append({"step": step, "arm": a, "event": "close",
                                     "object": ep.track_ids[near[0]]})
            else:
                j = self.held[a]
                resting = (j is not None and ep.objects[last, j, 2] > OPEN_HEIGHT and last > 0
                           and abs(ep.objects[last, j, 2] - ep.objects[last - 1, j, 2]) < 1e-3)
                self.dwell[a] = self.dwell[a] + 1 if resting else 0
                if self.dwell[a] >= DWELL:
                    self.closed[a], self.held[a], self.dwell[a] = False, None, 0
                    self.log.append({"step": step, "arm": a, "event": "open",
                                     "object": ep.track_ids[j]})
        return np.array([0.0 if c else 1.0 for c in self.closed])


class OracleGripper:
    """Privileged: the server's per-arm target oracle's next gripper command."""

    privileged = True

    def __init__(self) -> None:
        self.log: list[dict] = []
        self.last = np.array([1.0, 1.0])

    def __call__(self, obs, step: int) -> np.ndarray:
        if obs.arm_targets is None:
            raise RuntimeError("OracleGripper needs the server's arm targets (reset with arm_targets=True)")
        vias, mask = np.asarray(obs.arm_targets), np.asarray(obs.arm_targets_mask, bool)
        for a in range(2):
            if mask[a].any():
                g = float(vias[a][int(np.flatnonzero(mask[a])[0])][7])
                if g != self.last[a]:
                    self.log.append({"step": step, "arm": a, "event": "close" if g < 0.5 else "open"})
                self.last[a] = g
        return self.last.copy()


class PFNController:
    """The frozen PFN: one support per reset, ``k`` executed steps per prediction."""

    def __init__(self, model, support: Episode, k: int = 1, device: str = "cpu",
                 s_rows: int | None = None, history: int | None = None) -> None:
        from . import batches as Bt

        self.model, self.k, self.device = model, k, device
        self.s_rows = s_rows or Bt.S_ROWS
        self.history = history or Bt.HISTORY
        self.support_tab = support_table(support, Bt.support_rows(support, self.s_rows))
        self._memory = None
        self._queue: list[np.ndarray] = []
        self.windows: list[np.ndarray] = []

    def reset(self) -> None:
        self._memory, self._queue, self.windows = None, [], []

    def __call__(self, ep: Episode) -> np.ndarray:
        import torch

        from . import batches as Bt
        from .model import to_torch

        if not self._queue:
            item = {"support": self.support_tab, "query": query_table(ep, ep.T - 1, self.history),
                    "target": np.zeros((self.model.horizon, 2, 7), np.float32),
                    "target_valid": np.zeros(self.model.horizon, bool), "meta": {}}
            b = to_torch(Bt.pack([item], s_rows=self.s_rows), self.device)
            with torch.no_grad():
                if self._memory is None:
                    self._memory = self.model.encode_support(b)
                window = self.model.predict(self._memory, b)["window"][0].cpu().numpy()
            self.windows.append(window)
            self._queue = [window[h] for h in range(min(self.k, len(window)))]
        return self._queue.pop(0)


class ExpertWindowController:
    """Privileged bridge check: move each wrist toward its live oracle via-point."""

    privileged = True

    def __call__(self, ep: Episode, obs) -> np.ndarray:
        if obs.arm_targets is None:
            raise RuntimeError("needs the server's arm targets (reset with arm_targets=True)")
        vias, mask = np.asarray(obs.arm_targets), np.asarray(obs.arm_targets_mask, bool)
        cur = ep.wrist[-1].copy()
        out = cur.copy()
        for a in range(2):
            if not mask[a].any():
                continue
            tgt = vias[a][int(np.flatnonzero(mask[a])[0])][:7]
            d = tgt[:3] - cur[a, :3]
            n = np.linalg.norm(d)
            out[a, :3] = cur[a, :3] + (d if n <= MAX_STEP else d / n * MAX_STEP)
            ang = float(Q.angle(cur[a, 3:], tgt[3:]))
            s = 1.0 if ang <= MAX_TURN else MAX_TURN / ang
            out[a, 3:] = Q.slerp(cur[a, 3:], tgt[3:], np.array([s]))[0]
        return out


class HoldController:
    def __call__(self, ep: Episode, obs=None) -> np.ndarray:
        return ep.wrist[-1].copy()


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
