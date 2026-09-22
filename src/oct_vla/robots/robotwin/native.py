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
from collections.abc import Callable
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


class UnreachablePose(NativePortError):
    """The solver found no configuration reaching a commanded pose.

    Split out from its parent because the two mean opposite things during a
    closed-loop evaluation. A `NativePortError` is a broken simulator and must
    stop the run; an `UnreachablePose` is the *policy* asking for somewhere the
    arm cannot go, which is an ordinary -- and for an undertrained policy,
    frequent -- episode outcome. Left merged, a single bad action aborts the
    entire evaluation and discards every episode scored before it.
    """


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
        need_plan: bool = False,
    ) -> None:
        if plan_attempts < 1:
            raise ValueError(f"plan_attempts must be at least 1; got {plan_attempts}")
        self.root = Path(robotwin_root).expanduser().resolve()
        self.task_name = task_name
        self.task_config = task_config
        #: Builds the class to instantiate, overriding what `task_name` would
        #: resolve to. A factory rather than a class: recording a RoboTwin
        #: built-in subclasses it to intercept `_take_picture`, and defining
        #: that subclass needs `envs` importable -- which only happens inside
        #: `reset`, after `_prepare_import_path` and the chdir. Handing over a
        #: class built earlier fails with "Could not load RoboTwin task".
        self._task_factory: Callable[[], type] | None = None
        #: Let RoboTwin plan. Our oracle plans every motion itself and replays
        #: the result, so it wants this off; a built-in task's `play_once`
        #: expects RoboTwin to plan and otherwise indexes an unpopulated
        #: `left_joint_path`, failing with a bare IndexError.
        self.need_plan = need_plan
        self.plan_attempts = plan_attempts
        self._task: Any | None = None
        self._episode = 0
        #: track_id of the object this arm is engaging with, excluded from
        #: the planning world for the duration. Set by the oracle via
        #: `move_to`; see NativePort.ignored_object.
        self.ignored_object: str | None = None

    def use_task_factory(self, factory: Callable[[], type] | None) -> None:
        """Build the task class with `factory` on each reset, not by name.

        Called inside the prepared import path, so the factory may import from
        `envs` and subclass a built-in. `task_name` still configures the setup,
        so the task config, the embodiment override and the step limit are
        unchanged; only the class that gets instantiated differs.
        """
        self._task_factory = factory

    def reset(self, seed: int) -> None:
        _prepare_import_path(self.root)
        setup = build_dual_franka_setup(
            self.root,
            task_name=self.task_name,
            task_config=self.task_config,
            need_plan=self.need_plan,
        )
        with _chdir(self.root):
            if self._task is not None:
                self._safe_close()
            # The factory yields the *class*; instantiating it is this line's
            # job. Conflating the two put a class in `self._task`, and every
            # later call fired unbound -- "close_env() missing 1 required
            # positional argument: 'self'", which names neither the factory nor
            # the cause.
            task_class = (
                self._task_factory() if self._task_factory else _load_task_class(self.task_name)
            )
            self._task = task_class()
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

    def arm_joints(self, side: str) -> tuple[float, ...]:
        """Current positions of one arm's active joints, in the planner's order.

        The same ordering `command` expects and `ik` returns, so a caller can
        difference the two to work out how fast the arm has to move.
        """
        self._require_task()
        robot = self._task.robot
        planner = robot.left_planner if side == "left" else robot.right_planner
        entity = robot.left_entity if side == "left" else robot.right_entity
        qpos = entity.get_qpos()
        return tuple(
            float(qpos[planner.all_joints.index(name)])
            for name in planner.active_joints_name
            if name in planner.all_joints
        )

    def ik(self, side: str, pose_wxyz: tuple[float, ...]) -> tuple[float, ...]:
        """Joint positions reaching `pose_wxyz` (world frame), seeded from now.

        Closed-loop policy rollout needs this instead of `plan`: a policy emits a
        small end-effector increment every control step, and motion planning a
        fresh trajectory per step is both far too slow and wrong in kind -- the
        planner is free to reach the target along any collision-free path, while
        the policy is specifying the step itself.

        This is deliberately *kinematic* IK: environment obstacles are not
        supplied to its solver. A local Cartesian policy command may end at a
        contact pose (most importantly, a grasp), and asking collision-aware IK
        to validate that endpoint rejects the interaction the policy is meant
        to perform. SAPIEN executes the command and remains the authority on
        environment contact. Joint limits and cuRobo's robot self-collision
        constraints remain enabled.

        Everything RoboTwin-specific stays here rather than in the caller. The
        target passes through the same two frame changes `plan` relies on, by
        calling RoboTwin's own helpers rather than reimplementing them:
        gripper-to-endlink on the robot, then world-to-base on the planner, plus
        the same non-aloha `frame_bias` offset `plan_path` applies. Reusing them
        is the point -- a second, drifting copy of these transforms is exactly
        how a policy ends up evaluated against subtly different kinematics than
        the oracle that produced its training data.

        Seeding and regularising on the current configuration makes this IK
        *servoing*: of the many joint solutions reaching a pose, it returns the
        one nearest where the arm already is, so consecutive steps do not jump
        between elbow configurations.
        """
        self._require_task()
        robot = self._task.robot
        if getattr(robot, "communication_flag", False):
            raise NativePortError(
                "IK needs an in-process cuRobo planner, but this embodiment runs its "
                "planners in separate processes (per-arm curobo yml paths differ)"
            )
        planner = robot.left_planner if side == "left" else robot.right_planner
        motion_gen = getattr(planner, "motion_gen", None)
        if motion_gen is None:
            raise NativePortError(f"{side} planner is not a cuRobo planner; IK is unavailable")

        import numpy as np
        import torch
        from curobo.types.math import Pose as CuroboPose
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        # MotionGen's built-in IK solver shares its collision world. That is
        # correct for the oracle's global planner, but wrong for the evaluation
        # servo: the world includes the target object, so a valid grasp pose is
        # classified as infeasible simply because the gripper is touching it.
        # Cache one world-less solver on each live planner. A task reset creates
        # new planners, so this cannot leak across robot instances.
        ik_solver = getattr(planner, "_oct_vla_evaluation_ik_solver", None)
        if ik_solver is None:
            reference = motion_gen.ik_solver
            config = IKSolverConfig.load_from_robot_config(
                planner.yml_path,
                world_model=None,
                tensor_args=reference.tensor_args,
                # A Cartesian servo needs the solution connected to the arm's
                # current configuration, not the globally best of many random
                # elbow configurations. The current q is supplied as the sole
                # seed below; removing world obstacles is what makes that local
                # solve viable even at intentional contact poses.
                num_seeds=1,
                # MotionGen accepts 5 mm of endpoint error. That is tolerable
                # for a one-shot plan but compounds across incremental policy
                # actions. Sub-millimetre convergence also enables cuRobo's
                # high-precision iteration budget.
                position_threshold=0.0005,
                rotation_threshold=0.01,
                use_cuda_graph=reference.use_cuda_graph,
                self_collision_check=True,
                self_collision_opt=True,
                regularization=True,
            )
            ik_solver = IKSolver(config)
            planner._oct_vla_evaluation_ik_solver = ik_solver

        with _chdir(self.root):
            endlink = robot._trans_from_gripper_to_endlink(list(pose_wxyz), arm_tag=side)
            base = np.concatenate(
                [np.array(planner.robot_origion_pose.p), np.array(planner.robot_origion_pose.q)]
            )
            target = np.concatenate([np.array(endlink.p), np.array(endlink.q)])
            position, orientation = planner._trans_from_world_to_base(base, target)
            bias = np.asarray(planner.frame_bias, dtype=float)
            position = np.asarray(position, dtype=float) + bias

            current = self.arm_joints(side)
            seed = torch.tensor(
                [round(value, 5) for value in current], dtype=torch.float32
            ).cuda().reshape(1, -1)
            goal = CuroboPose.from_list(list(position) + list(orientation))
            result = ik_solver.solve_single(
                goal,
                retract_config=seed,
                seed_config=seed.unsqueeze(0),
            )

        if not bool(result.success.any()):
            raise UnreachablePose(
                f"{side} arm kinematic IK failed for world pose "
                f"{tuple(round(float(v), 4) for v in pose_wxyz)}"
            )
        # Select by joint name, never by position. cuRobo solves over the whole
        # robot model, so for a Panda it returns nine values -- seven arm joints
        # plus two gripper fingers -- while `set_arm_joints` wants exactly the
        # planner's active arm joints, in the planner's order. Slicing the first
        # seven would happen to work here and break silently on any embodiment
        # whose model orders joints differently, commanding the arm to a
        # configuration nobody asked for.
        values = result.js_solution.position.view(-1).tolist()
        names = list(getattr(result.js_solution, "joint_names", None) or [])
        if len(names) != len(values):
            raise NativePortError(
                f"IK returned {len(values)} joint values but "
                f"{len(names)} joint names; cannot match them to arm joints"
            )
        by_name = dict(zip(names, values, strict=True))
        missing = [name for name in planner.active_joints_name if name not in by_name]
        if missing:
            raise NativePortError(
                f"IK solution is missing arm joints {missing}; "
                f"it solved for {names}"
            )
        return tuple(float(by_name[name]) for name in planner.active_joints_name)

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
