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
from oct_vla.core.geometry import exp, log, multiply, rotate
from oct_vla.core.objects import ObjectState

# Gripper pointing straight down: local +X (this frame's approach axis, the
# same convention robot.left_plan_path/right_plan_path expect internally)
# maps to world/workcell (0, 0, -1). Equivalent, up to the usual double-cover
# sign, to RoboTwin's own GRASP_DIRECTION_DIC["top_down"] wxyz constant
# (verified by composing/rotating both through this module's own primitives).
POINTING_DOWN = (-0.5, 0.5, 0.5, 0.5)

#: Distance from the pose `robot.get_*_ee_pose()` reports back to the point
#: between the finger pads, along the approach axis. The reported pose is NOT
#: the grasp point: commanding it at an object's centre buries the fingers
#: about this far below the object (confirmed live -- a finger drove into the
#: table). Composed from three independently checkable sources: the reported
#: pose sits 0.0397m above `panda_hand` (measured in-scene), `panda_hand` is
#: 0.0584m from the finger link origin (panda.urdf, panda_finger_joint1), and
#: the finger pad's collision spheres are centred 0.015m and 0.043m along the
#: finger (collision_franka.yml, cuRobo's own model), whose midpoint is
#: 0.029m. As a cross-check the same model puts the finger origin at
#: reported-0.0981m; it measures at reported-0.0976m.
GRASP_TCP_OFFSET = 0.0397 + 0.0584 + 0.029

#: How far panda_hand's lowest collision sphere sits below the commanded pose:
#: 0.0397m to the hand origin, then its outermost sphere centre at 0.045 with
#: radius 0.022 (collision_franka.yml).
_HAND_REACH_BELOW_TCP = 0.0397 + 0.045 + 0.022

#: Deepest the finger pads may sit below an object's top face before the hand
#: body itself reaches the object. Grasping at an object's *centre* fails this
#: for anything taller than about 4cm -- observed live as
#: `panda_hand <-> restock_object_0` while descending onto a 7.5cm object.
MAX_GRASP_DEPTH = GRASP_TCP_OFFSET - _HAND_REACH_BELOW_TCP

#: Default pad depth below the top face, comfortably inside MAX_GRASP_DEPTH.
DEFAULT_GRASP_DEPTH = 0.015


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
    obj: ObjectState,
    *,
    gripper_max_width: float,
    standoff: float,
    grasp_depth: float = DEFAULT_GRASP_DEPTH,
) -> tuple[GraspCandidate, ...]:
    """Candidates that put the finger pads `grasp_depth` below the object's
    top face, and a pregrasp `standoff` further back along the approach axis,
    filtered to those whose horizontal object dimension fits the gripper.

    Poses are offset back along the approach axis by `GRASP_TCP_OFFSET`,
    because the pose the robot reports and accepts is not the point between
    the fingers.

    Grasping near the top rather than at the object's centre is what keeps
    the hand body clear of a tall object; see `MAX_GRASP_DEPTH`.

    Which object axis each 90-degree offset corresponds to (x for 0, y for
    pi/2) is this module's best-effort mapping to the frame convention
    robot.left_plan_path/right_plan_path use internally; it has not yet been
    confirmed by a live grasp attempt (see docs/architecture.md).
    """
    if not 0.0 < grasp_depth < MAX_GRASP_DEPTH:
        raise ValueError(
            f"grasp_depth must be in (0, {MAX_GRASP_DEPTH:.4f}); deeper puts the hand "
            f"body inside the object. Got {grasp_depth}"
        )
    object_yaw = _object_yaw(obj)
    # Distance from the object's centre up to where the pads should sit.
    rise = max(obj.size_xyz[2] / 2.0 - grasp_depth, 0.0)
    candidates = []
    for offset_yaw, closing_width in ((0.0, obj.size_xyz[0]), (pi / 2, obj.size_xyz[1])):
        if closing_width > gripper_max_width:
            continue
        orientation = _top_down_orientation(object_yaw + offset_yaw)
        # Unit vector pointing from the commanded pose toward the object.
        approach = rotate(orientation, (1.0, 0.0, 0.0))
        pads = tuple(c - rise * a for c, a in zip(obj.pose.position, approach, strict=True))
        grasp_position = tuple(
            p - GRASP_TCP_OFFSET * a for p, a in zip(pads, approach, strict=True)
        )
        pregrasp_position = tuple(
            g - standoff * a for g, a in zip(grasp_position, approach, strict=True)
        )
        candidates.append(
            GraspCandidate(
                Pose(grasp_position, orientation, WORKCELL_FRAME),
                Pose(pregrasp_position, orientation, WORKCELL_FRAME),
                closing_width,
                object_yaw + offset_yaw,
            )
        )
    return tuple(candidates)
