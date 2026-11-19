"""The direct PFN loop (stage 2b) against a scripted client: history from measurements,
cartesian_absolute actions, gripper controllers, and no read of the task context."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.core.state import ArmState, EEFState
from oct_vla.prior.pose_window import fixtures as F
from oct_vla.prior.pose_window import online as On
from oct_vla.prior.pose_window.model import PoseWindowPFN
from oct_vla.prior.pose_window.trajectory import TOP_DOWN, wrist_for_grip


def _obs(t, left, right, objects, grips=(0.83, 0.83), arm_targets=None):
    eef = EEFState(ArmState(Pose(tuple(left[:3]), tuple(left[3:])), grips[0]),
                   ArmState(Pose(tuple(right[:3]), tuple(right[3:])), grips[1]))
    scene = ObjectScene(t, tuple(ObjectState(k, Pose(tuple(p), (0, 0, 0, 1)), (0.05, 0.05, 0.07),
                                             1.0, 1.0, colour=c) for k, (p, c) in objects.items()))
    # A context that would leak the answer if read: the loop must never touch it.
    return SimpleNamespace(timestamp=t, eef=eef, scene=scene, context="SECRET",
                           arm_targets=arm_targets, arm_targets_mask=None)


class ScriptedClient:
    """Replays an observation per step and records the actions it receives."""

    def __init__(self, observations):
        self.obs, self.actions, self.reset_kwargs = list(observations), [], None

    def reset(self, seed, **kw):
        self.reset_kwargs = kw
        return self.obs[0]

    def step(self, action):
        self.actions.append(action)
        i = min(len(self.actions), len(self.obs) - 1)
        done = len(self.actions) >= len(self.obs) - 1
        return SimpleNamespace(observation=self.obs[i], success=False, done=done,
                               reason="step_limit" if done else "", transfers_completed=0,
                               objects_lifted=0, infeasible_steps=0, events={},
                               success_at=None, still_restocked=False,
                               distractors_preserved=True)


L0 = np.concatenate([[-0.3, -0.12, 1.2], TOP_DOWN])
R0 = np.concatenate([[0.3, -0.12, 1.2], TOP_DOWN])
OBJ = {"obj_0": ((-0.2, -0.25, 0.78), "red"), "obj_1": ((0.1, -0.25, 0.78), "blue")}


def test_pfn_loop_sends_absolute_actions_and_never_reads_the_context():
    torch.manual_seed(0)
    model = PoseWindowPFN(width=32, heads=2, layers=1).eval()
    support = F.restock_episode([-0.2, 0.1], ["red", "blue"], "red")
    ctrl = On.PFNController(model, support, k=2, s_rows=32)
    client = ScriptedClient([_obs(0.067 * i, L0, R0, OBJ) for i in range(6)])
    res = On.run_episode(client, ctrl, On.ObservableGripper(), seed=0, profile="four_object",
                         max_steps=5)
    assert client.reset_kwargs["control_space"] == "cartesian_absolute"
    assert client.reset_kwargs["gripper_encoding"] == "binary_command"
    assert all(len(a) == On.ACTION_DIM for a in client.actions)
    # k = 2: one prediction serves two control steps.
    assert len(ctrl.windows) == 3 and res["steps"] == 5
    for a in client.actions:
        np.testing.assert_allclose(np.linalg.norm(a[3:7]), 1.0, atol=1e-6)
    assert res["colours"][:2] == [1, 2]                 # red, blue reach the table header


def test_history_builds_tables_from_measurements_only():
    h = On.History()
    h.add(_obs(0.0, L0, R0, OBJ))
    moved = dict(OBJ, obj_0=((-0.2, -0.25, 0.80), "red"))
    h.add(_obs(0.067, L0, R0, moved))
    ep = h.episode()
    assert ep.T == 2 and ep.track_ids[:2] == ("obj_0", "obj_1")
    assert ep.objects[1, 0, 2] == pytest.approx(0.80)
    assert ep.kind.tolist() == [3, 3, 4, 4]           # movables, then the two decks


def test_observable_gripper_closes_on_dwell_and_opens_at_the_destination():
    g = On.ObservableGripper()
    grasp = wrist_for_grip((-0.2, -0.25, 0.78), TOP_DOWN)
    h = On.History()
    for i in range(On.DWELL):
        h.add(_obs(0.067 * i, grasp, R0, OBJ))
        cmd = g(h.episode(), i)
    assert cmd[0] == 0.0 and cmd[1] == 1.0 and g.log[-1]["event"] == "close"
    up = dict(OBJ, obj_0=((-0.1, -0.06, 0.97), "red"))
    for i in range(On.DWELL + 1):
        h.add(_obs(0.3 + 0.067 * i, grasp, R0, up))
        cmd = g(h.episode(), 10 + i)
    assert cmd[0] == 1.0 and g.log[-1]["event"] == "open"
    # It never decided from anything but measured poses: no target id was available.
    assert all("object" in e for e in g.log)


def test_expert_window_controller_moves_toward_the_via_point_at_a_bounded_speed():
    via = np.zeros((2, 6, 8))
    via[0, 0, :7] = np.concatenate([[-0.2, -0.25, 1.0], TOP_DOWN])
    mask = np.zeros((2, 6), bool)
    mask[0, 0] = True
    obs = _obs(0.0, L0, R0, OBJ, arm_targets=via)
    obs.arm_targets_mask = mask
    h = On.History()
    h.add(obs)
    out = On.ExpertWindowController()(h.episode(), obs)
    step = np.linalg.norm(out[0, :3] - L0[:3])
    assert step == pytest.approx(On.MAX_STEP)
    np.testing.assert_allclose(out[1], R0)            # no via-point: the arm holds
