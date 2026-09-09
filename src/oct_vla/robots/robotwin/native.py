"""Concrete NativePort: every RoboTwin/SAPIEN/cuRobo dependency stops here.

This module only imports successfully inside RoboTwin's own Python 3.10
environment. ``RoboTwinBackend`` (backend.py) never imports it directly; a
launcher process constructs a ``RoboTwinNativePort`` and hands it to
``RoboTwinBackend`` there.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from typing import Any

from oct_vla.core.observation import RGBFrame
from oct_vla.robots.robotwin.backend import ArmReading, Contact, Reading, Trajectory
from oct_vla.robots.robotwin.task_config import build_dual_franka_setup

CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")


class NativePortError(RuntimeError):
    """RoboTwin planning or execution failure; callers must not swallow this."""


def _prepare_import_path(root: Path) -> None:
    """Give our own cuRobo/RoboTwin paths priority over a stale editable install.

    cuRobo is installed editable against wherever RoboTwin lived when that
    install ran. Relocating the checkout (as happened here) leaves the venv's
    ``.pth`` pointing at a path that no longer exists; repairing the venv
    itself is out of scope; sys.path in our own process is not.
    """
    for candidate in (str(root / "envs" / "curobo" / "src"), str(root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


@contextmanager
def _chdir(path: Path):
    """RoboTwin task modules resolve asset paths relative to the checkout root."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_task_class(task_name: str) -> Any:
    """Load a RoboTwin-native task (``"stack_blocks_three"``) or an external
    one owned outside the checkout (``"pkg.module:ClassName"``). External
    task classes still subclass RoboTwin's own Base_Task directly; only the
    module's own location is external, since ``envs`` is importable once
    _prepare_import_path/_chdir have run (see e.g. tasks/shelf_restock).
    """
    if ":" in task_name:
        module_name, class_name = task_name.split(":", maxsplit=1)
        try:
            return getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as error:
            raise NativePortError(f"Could not load external task {task_name!r}") from error
    try:
        module = importlib.import_module(f"envs.{task_name}")
        return getattr(module, task_name)
    except (ImportError, AttributeError) as error:
        raise NativePortError(
            f"Could not load RoboTwin task envs.{task_name}.{task_name}"
        ) from error


def _rgb_frame(raw: dict, camera_name: str) -> RGBFrame:
    import numpy as np

    try:
        image = raw["observation"][camera_name]["rgb"]
    except (KeyError, TypeError) as error:
        raise NativePortError(f"RoboTwin observation is missing {camera_name}/rgb") from error
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise NativePortError(f"{camera_name}/rgb must have shape [H, W, 3]; got {array.shape}")
    array = np.ascontiguousarray(array, dtype=np.uint8)
    height, width = array.shape[:2]
    return RGBFrame(width, height, array.tobytes())


class RoboTwinNativePort:
    """Drive one RoboTwin task instance behind the canonical NativePort seam.

    Bypasses RoboTwin's own ``take_action``: that helper silently falls back
    to a fixed-length no-op trajectory when a plan fails, which would hide
    planning failures from the canonical backend. This port raises instead.
    """

    dt = 1.0 / 250.0

    def __init__(
        self,
        robotwin_root: Path,
        *,
        task_name: str,
        task_config: str = "demo_clean",
        plan_attempts: int = 4,
    ) -> None:
        if plan_attempts < 1:
            raise ValueError(f"plan_attempts must be at least 1; got {plan_attempts}")
        self.root = Path(robotwin_root).expanduser().resolve()
        self.task_name = task_name
        self.task_config = task_config
        self.plan_attempts = plan_attempts
        self._task: Any | None = None
        self._episode = 0
        #: track_id of the object this arm is engaging with, excluded from
        #: the planning world for the duration. Set by the oracle via
        #: `move_to`; see NativePort.ignored_object.
        self.ignored_object: str | None = None

    def reset(self, seed: int) -> None:
        _prepare_import_path(self.root)
        setup = build_dual_franka_setup(
            self.root, task_name=self.task_name, task_config=self.task_config
        )
        with _chdir(self.root):
            if self._task is not None:
                self._safe_close()
            self._task = _load_task_class(self.task_name)()
            self._task.setup_demo(now_ep_num=self._episode, seed=seed, is_test=True, **setup)
            self.ignored_object = None
        actual_dt = float(self._task.scene.get_timestep())
        if abs(actual_dt - self.dt) > 1e-9:
            raise NativePortError(
                f"RoboTwin scene timestep is {actual_dt}s; expected {self.dt}s. "
                "Update RoboTwinNativePort.dt to match the configured task_config."
            )
        self._episode += 1

    @property
    def task(self) -> Any:
        """The live RoboTwin task. Simulation-privileged: the oracle and the
        collection glue need its actors to build a ground-truth estimator, and
        reaching through a private attribute to get them would hide that."""
        self._require_task()
        return self._task

    def read(self) -> Reading:
        self._require_task()
        with _chdir(self.root):
            raw = self._task.get_obs()
            left = self._arm_reading("left")
            right = self._arm_reading("right")
        cameras = tuple(_rgb_frame(raw, name) for name in CAMERA_NAMES)
        return Reading(left, right, cameras)

    def plan(
        self,
        side: str,
        pose_wxyz: tuple[float, ...],
        constraint: tuple[float, ...] | None = None,
    ) -> Trajectory:
        self._require_task()
        # `constraint` is cuRobo's hold_vec_weight, 6 values: indices 0-2 are
        # rotation, 3-5 are translation, 1.0 holds that component fixed along
        # the path and 0.0 leaves it free. Six because that is exactly the
        # shape CuroboPlanner.plan_path's constraint_pose expects; anything
        # else can only be a caller mistake, so fail loudly here rather than
        # inside RoboTwin's planner.
        if constraint is not None and (
            len(constraint) != 6 or not all(isfinite(v) for v in constraint)
        ):
            raise NativePortError(
                f"constraint must have exactly 6 finite elements (rotation xyz, "
                f"translation xyz); got {constraint!r}"
            )
        # Refresh the planner's obstacles from the live scene first. Doing it
        # here rather than at each call site is deliberate: a plan made
        # against a stale world is the failure mode that is hardest to notice,
        # since the planner still reports Success while routing through an
        # object it thinks has not moved. Tasks with no movable geometry need
        # no hook and simply do not define one.
        refresh = getattr(self._task, "refresh_planning_world", None)
        if refresh is not None:
            with _chdir(self.root):
                refresh(self.ignored_object)
        plan_path = self._plan_path_fn(side)
        # Retry a bounded number of times before giving up. cuRobo's trajopt is
        # seeded stochastically and re-seeds per call, so a fresh attempt from
        # the identical state genuinely succeeds where the last one failed --
        # measured directly: collecting 32 seeds with a single attempt yielded
        # 14 episodes, every loss a bare `Fail` and split evenly across both
        # arms. This is not swallowing the failure: it still raises once the
        # attempts are spent, and says how many were tried.
        for _ in range(self.plan_attempts):
            with _chdir(self.root):
                if constraint is None:
                    result = plan_path(list(pose_wxyz))
                else:
                    result = plan_path(list(pose_wxyz), constraint_pose=list(constraint))
            if result.get("status") == "Success":
                break
        else:
            raise NativePortError(
                f"{side} arm global plan failed after {self.plan_attempts} attempts: "
                f"{result.get('status')!r}"
            )
        position = tuple(tuple(float(v) for v in row) for row in result["position"])
        velocity = tuple(tuple(float(v) for v in row) for row in result["velocity"])
        return Trajectory(position, velocity)

    def command(
        self, side: str, q: tuple[float, ...], qdot: tuple[float, ...], gripper: float
    ) -> None:
        self._require_task()
        with _chdir(self.root):
            self._task.robot.set_arm_joints(list(q), list(qdot), side)
            self._task.robot.set_gripper(gripper, side)

    def tick(self) -> None:
        self._require_task()
        self._task.scene.step()

    def contacts(self, min_impulse: float = 1e-6) -> tuple[Contact, ...]:
        """Contacts involving a robot link, above a noise threshold.

        Robot links are keyed by their entity's `per_scene_id`, not by link
        name: both arms load the same Panda URDF, so all fourteen link names
        are shared between them and a name cannot say which arm a link is on.
        Reporting an arm-vs-arm collision as a bare
        'panda_rightfinger <-> panda_hand' made a real collision between the
        moving arm and the parked one indistinguishable from a gripper
        closing on itself.
        """
        self._require_task()
        robot = self._task.robot
        owners = {
            link.entity.per_scene_id: (side, link.get_name())
            for side, entity in (("left", robot.left_entity), ("right", robot.right_entity))
            for link in entity.get_links()
        }
        found = []
        for contact in self._task.scene.get_contacts():
            first, second = contact.bodies
            impulse = sum(
                sum(component**2 for component in point.impulse) ** 0.5 for point in contact.points
            )
            if impulse <= min_impulse:
                continue
            for body, counterpart in ((first, second), (second, first)):
                owner = owners.get(body.entity.per_scene_id)
                if owner is None:
                    continue
                side, link = owner
                other = owners.get(counterpart.entity.per_scene_id)
                other_name = f"{other[0]}/{other[1]}" if other else counterpart.entity.name
                found.append(Contact(side, link, other_name, impulse))
                break
        return tuple(found)

    def hold(self) -> None:
        """Freeze both arms at their measured position with zero velocity."""
        self._require_task()
        with _chdir(self.root):
            for side in ("left", "right"):
                q = self._measured_arm_qpos(side)
                self._task.robot.set_arm_joints(list(q), [0.0] * len(q), side)
                self._task.robot.set_gripper(self._measured_gripper(side), side)

    def close(self) -> None:
        if self._task is None:
            return
        self._safe_close()
        self._task = None

    # -- internal --

    def _require_task(self) -> None:
        if self._task is None:
            raise NativePortError("RoboTwinNativePort.reset() has not been called")

    def _safe_close(self) -> None:
        with _chdir(self.root):
            close = getattr(self._task, "close_env", None) or getattr(self._task, "close", None)
            if close is not None:
                close()

    def _plan_path_fn(self, side: str):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        return (
            self._task.robot.left_plan_path if side == "left" else self._task.robot.right_plan_path
        )

    def _arm_reading(self, side: str) -> ArmReading:
        robot = self._task.robot
        pose = robot.get_left_ee_pose() if side == "left" else robot.get_right_ee_pose()
        return ArmReading(
            tuple(float(v) for v in pose),
            self._measured_gripper(side),
            self._measured_arm_qpos(side),
        )

    def _measured_gripper(self, side: str) -> float:
        """Physical joint position, not the drive target ``get_*_gripper_val`` returns."""
        robot = self._task.robot
        entity = robot.left_entity if side == "left" else robot.right_entity
        joints = robot.left_gripper if side == "left" else robot.right_gripper
        scale = robot.left_gripper_scale if side == "left" else robot.right_gripper_scale
        if not joints or None in joints:
            return 0.0
        active = entity.get_active_joints()
        qpos = entity.get_qpos()
        value = (qpos[active.index(joints[0][0])] - scale[0]) / (scale[1] - scale[0])
        return float(min(1.0, max(0.0, value)))

    def _measured_arm_qpos(self, side: str) -> tuple[float, ...]:
        """Physical joint qpos, not the drive target ``get_*_arm_jointState`` returns."""
        robot = self._task.robot
        entity = robot.left_entity if side == "left" else robot.right_entity
        arm_joints = robot.left_arm_joints if side == "left" else robot.right_arm_joints
        active = entity.get_active_joints()
        qpos = entity.get_qpos()
        return tuple(float(qpos[active.index(joint)]) for joint in arm_joints)
