"""Top-down grasp-candidate generation for box-shaped objects.

RoboTwin's ``create_box`` attaches synthetic "contact points" for grasping,
but for a procedural box they carry only orientation -- their translation is
always ``(0, 0, 0)`` regardless of object size, unlike a real scanned asset
whose contact points land genuinely off-center on its mesh. Reusing them
would buy nothing a direct derivation doesn't already give, so this module
derives top-down grasp poses directly, in the workcell frame, independent of
SAPIEN/RoboTwin -- the same generic/pure-first pattern as backend.py's
world<->workcell conversion and perception/ground_truth.py's estimator.

Two 90-degree-apart candidates are generated (RoboTwin's own 4 box contact
points reduce to the same 2 unique top-down orientations plus their 180-degree
duplicates). Each is scored only by whether its associated horizontal object
dimension fits the gripper's opening; IK/planner-reachability scoring belongs
to runtime code that can actually attempt a plan against a live scene, not
here.
"""

from dataclasses import dataclass
from math import pi

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.geometry import exp, log, multiply
from oct_vla.core.objects import ObjectState

# Gripper pointing straight down: local +X (this frame's approach axis, the
# same convention robot.left_plan_path/right_plan_path expect internally)
# maps to world/workcell (0, 0, -1). Equivalent, up to the usual double-cover
# sign, to RoboTwin's own GRASP_DIRECTION_DIC["top_down"] wxyz constant
# (verified by composing/rotating both through this module's own primitives).
POINTING_DOWN = (-0.5, 0.5, 0.5, 0.5)


@dataclass(frozen=True)
class GraspCandidate:
    grasp_pose: Pose
    pregrasp_pose: Pose
    closing_width: float
    wrist_yaw: float


def _object_yaw(obj: ObjectState) -> float:
    """The object's own rotation about z, assuming it has no roll/pitch.

    Holds by construction for objects spawned via ObjectVariation.sample_pose
    (a pure z-axis rotation); a tilted object would need axis-projection
    logic this does not implement.
    """
    _, _, yaw = log(obj.pose.orientation)
    return yaw


def _top_down_orientation(yaw: float) -> tuple[float, float, float, float]:
    """Point straight down, wrist rotated by `yaw` about the world/workcell z
    axis. Composing a z-rotation on the left of POINTING_DOWN leaves the
    approach axis at (0, 0, -1) unchanged, since it is already antiparallel
    to the rotation axis; only the closing axis spins in the horizontal plane.
    """
    return multiply(exp((0.0, 0.0, yaw)), POINTING_DOWN)


def generate_top_down_grasps(
    obj: ObjectState, *, gripper_max_width: float, standoff: float
) -> tuple[GraspCandidate, ...]:
    """Candidates with the TCP at the object's center (grasp) and `standoff`
    meters directly above it (pregrasp), filtered to those whose associated
    horizontal object dimension fits within the gripper's opening.

    Which object axis each 90-degree offset corresponds to (x for 0, y for
    pi/2) is this module's best-effort mapping to the frame convention
    robot.left_plan_path/right_plan_path use internally; it has not yet been
    confirmed by a live grasp attempt (see docs/architecture.md).
    """
    object_yaw = _object_yaw(obj)
    candidates = []
    for offset_yaw, closing_width in ((0.0, obj.size_xyz[0]), (pi / 2, obj.size_xyz[1])):
        if closing_width > gripper_max_width:
            continue
        orientation = _top_down_orientation(object_yaw + offset_yaw)
        cx, cy, cz = obj.pose.position
        grasp_pose = Pose((cx, cy, cz), orientation, WORKCELL_FRAME)
        pregrasp_pose = Pose((cx, cy, cz + standoff), orientation, WORKCELL_FRAME)
        candidates.append(
            GraspCandidate(grasp_pose, pregrasp_pose, closing_width, object_yaw + offset_yaw)
        )
    return tuple(candidates)
