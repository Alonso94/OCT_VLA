"""Where a restocked object goes on the upper shelf, and where it ends up
after being compacted toward its previous neighbour.

Pure geometry over the spec and an `ObjectScene`; no SAPIEN, no planner. The
oracle turns these object poses into commanded poses with
`grasps.holding_tcp_pose`, since placing a held object is the same
relationship as grasping it, read in the other direction.

The atomic task is "restock one object, then compact it toward the previous
neighbour if one exists", so placement deliberately leaves a gap wider than
the success threshold: if the object were simply placed at its final
compacted position there would be no compaction to demonstrate.
"""

from dataclasses import dataclass
from math import cos, pi, sin

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.geometry import exp, log, multiply
from oct_vla.core.objects import ObjectState
from oct_vla.tasks.shelf_restock.oracle.arms import Side
from oct_vla.tasks.shelf_restock.oracle.grasps import (
    DEFAULT_GRASP_DEPTH,
    GRASP_TCP_OFFSET,
    POINTING_DOWN,
    backed_off,
    holding_tcp_pose,
)
from oct_vla.tasks.shelf_restock.spec import ShelfRestockSpec
from oct_vla.tasks.shelf_restock.success import horizontal_radius

#: Surface gap left when first setting an object down beside its neighbour.
#: Must clear ShelfRestockSpec.compaction_distance (0.04) by more than the
#: measured placement error (now ~0.0005m, down from the ~0.025m this value
#: was originally sized against) even in the worst case, or a freshly placed
#: object could already land inside the compaction threshold and the
#: compaction step would be vacuous. 0.06 +/- 0.0005 stays above 0.04 with
#: ~0.02 of margin; COMPACTED_GAP (0.005) +/- 0.0005 stays below it -- so the
#: two states remain distinguishable under real placement noise. Keeping the
#: gap this small also keeps a placed row physically shorter, which matters
#: because row length is what pushes objects out of an arm's reach.
PLACEMENT_GAP = 0.06

#: Surface gap after compaction: close enough to read as "pushed together",
#: with a little clearance so the nudge does not rely on interpenetration.
COMPACTED_GAP = 0.005

#: Lateral gap left between the closed gripper's own pushing face and the
#: object's trailing face while descending beside it, so the descent does not
#: scrape the face it is about to push. This is clearance BEYOND the
#: gripper's own half-width (PUSHER_HALF_WIDTH) -- genuine free space between
#: the blade's face and the object's face, not between the commanded pose
#: and the object.
CONTACT_CLEARANCE = 0.01

#: Half the closed gripper's own thickness along the push axis. The gripper
#: is a solid blade, not a point: its pushing face sits this far from the
#: pose the arm is actually commanded to, so it has to be accounted for both
#: when descending clear of the object and when deciding where to stop.
PUSHER_HALF_WIDTH = 0.012


def push_axis_half_extent(obj: ObjectState) -> float:
    """Half the object's extent along the world x axis, given its yaw.

    `horizontal_radius` deliberately returns a yaw-independent circumscribing
    radius -- the right conservative choice for a neighbour-gap test, since
    the true surface gap along the line of centres is never smaller than
    that value. But a push descends beside, and must clear, the object's
    *actual* face along the push axis, not a nominal circle: for a rotated
    box that face can sit well outside the circumscribing radius. A live run
    measured exactly this on a 0.054 x 0.0479 box yawed 0.3265rad --
    `horizontal_radius` assumed 0.027 while the real x half-extent was
    0.0332 -- and the descent jammed on the box, the arm stalling 62mm high
    with only a fingertip in contact.
    """
    size_x, size_y, _ = obj.size_xyz
    yaw = log(obj.pose.orientation)[2]
    return (size_x * abs(cos(yaw)) + size_y * abs(sin(yaw))) / 2.0


class PlacementError(RuntimeError):
    """Raised when no valid placement exists on the shelf."""


@dataclass(frozen=True)
class Placement:
    """Object poses, plus the commanded poses that put the object there.

    Only carries the *target* compacted pose, not a compaction command pose:
    where the object should end up is a placement decision, but how to move
    it there is an execution decision that must be made against the object's
    observed pose at compaction time, not the pose it was predicted to land
    at. The oracle calls `plan_push` with the re-observed object instead.
    """

    object_pose: Pose
    place_pose: Pose
    preplace_pose: Pose
    compacted_object_pose: Pose | None

    @property
    def needs_compaction(self) -> bool:
        return self.compacted_object_pose is not None


def resting_z(spec: ShelfRestockSpec, obj: ObjectState) -> float:
    """Height of an object's centre when it sits on the upper deck."""
    return spec.upper_shelf.top_z + obj.size_xyz[2] / 2.0


def _offset_position(
    neighbour: ObjectState, obj: ObjectState, gap: float, direction: float, z: float
):
    reach = horizontal_radius(neighbour) + horizontal_radius(obj) + gap
    return (neighbour.pose.position[0] + direction * reach, neighbour.pose.position[1], z)


def _placement_direction(
    spec: ShelfRestockSpec, neighbour: ObjectState, obj: ObjectState, side: Side | None
) -> float:
    """Put the object on whichever side of the neighbour the row should grow toward.

    The old "roomier side" heuristic actively breaks a row: once the first
    object sits near one edge, the roomier side is always back toward the
    middle, so a third object would land on top of the first. A chain only
    works if it grows consistently one way, so this prefers the direction
    toward the arm that will compact (`side`) -- +1.0 ("right") or -1.0
    ("left") -- and only falls back to the other side when the preferred one
    has no room. With no `side`, or when the preferred side is full, the
    original roomier-side rule still applies.
    """
    shelf = spec.upper_shelf
    low = shelf.center_xyz[0] - shelf.half_extent_xyz[0]
    high = shelf.center_xyz[0] + shelf.half_extent_xyz[0]
    neighbour_x = neighbour.pose.position[0]
    needed = horizontal_radius(neighbour) + horizontal_radius(obj) + PLACEMENT_GAP
    room_positive = high - (neighbour_x + needed)
    room_negative = (neighbour_x - needed) - low
    if max(room_positive, room_negative) < 0.0:
        raise PlacementError(
            f"no room beside neighbour at x={neighbour_x:.3f} for an object needing "
            f"{needed:.3f}m on a shelf spanning [{low:.3f}, {high:.3f}]"
        )
    if side == "right" and room_positive >= 0.0:
        return 1.0
    if side == "left" and room_negative >= 0.0:
        return -1.0
    return 1.0 if room_positive >= room_negative else -1.0


def plan_placement(
    spec: ShelfRestockSpec,
    obj: ObjectState,
    wrist_yaw: float,
    neighbour: ObjectState | None = None,
    *,
    standoff: float,
    grasp_depth: float = DEFAULT_GRASP_DEPTH,
    side: Side | None = None,
) -> Placement:
    """Where to put `obj`, and how to command the arm holding it there.

    `side` names the arm that will perform the *compaction*, not the one
    placing `obj`, and it biases where the row starts and grows so the chain
    never has to be pushed back across an already-placed object.

    Compaction pulls each newly placed object back toward its neighbour, so
    a row does not march across the deck -- it stays clustered near wherever
    the first object landed. A live run found the right arm could not plan
    to x=-0.124 at the deck's y (the deck is 0.60m wide), even though
    `contains()` accepts that x and `is_cross_body_limited` does not flag it
    (the deck now sits below CROSS_BODY_Y, so that guard never fires here) --
    a real reach limit this codebase has not measured a boundary for. Starting
    a row at the deck edge furthest from the compacting arm parks the whole
    clustered row exactly in that unreachable outer third. So with no
    neighbour, the object instead starts one placement step from the deck
    centre, on the far side from `side`: a short row then sits centred on the
    deck, inside the overlap of both arms' reach, instead of against an edge
    only one arm can get to. (With `side` None, it still starts at the deck
    centre.) With a neighbour, `obj` goes beside it leaving PLACEMENT_GAP,
    preferring the side toward `side`, and `compacted_object_pose` names the
    target that closes that gap to COMPACTED_GAP -- the *where*, not the
    *how*. Getting the object there is `plan_push`'s job, run later against
    the object's re-observed pose rather than a prediction.
    """
    z = resting_z(spec, obj)
    if neighbour is None:
        shelf = spec.upper_shelf
        shelf_centre_x = shelf.center_xyz[0]
        step = 2 * horizontal_radius(obj) + PLACEMENT_GAP
        if side == "right":
            x = shelf_centre_x - step
        elif side == "left":
            x = shelf_centre_x + step
        else:
            x = shelf_centre_x
        position = (x, spec.upper_shelf.center_xyz[1], z)
        compacted = None
    else:
        direction = _placement_direction(spec, neighbour, obj, side)
        position = _offset_position(neighbour, obj, PLACEMENT_GAP, direction, z)
        compacted = _offset_position(neighbour, obj, COMPACTED_GAP, direction, z)

    if not spec.upper_shelf.contains(position):
        raise PlacementError(
            f"placement {tuple(round(v, 3) for v in position)} is not on the upper shelf"
        )

    place_pose = holding_tcp_pose(position, obj.size_xyz[2], wrist_yaw, grasp_depth=grasp_depth)
    return Placement(
        object_pose=Pose(position, obj.pose.orientation, WORKCELL_FRAME),
        place_pose=place_pose,
        preplace_pose=backed_off(place_pose, standoff),
        compacted_object_pose=(
            Pose(compacted, obj.pose.orientation, WORKCELL_FRAME) if compacted else None
        ),
    )


#: cuRobo `hold_vec_weight` for a push, evaluated in the goal (end-effector)
#: frame since `PoseCostConfig.project_distance` defaults True and nothing
#: here overrides it. This lives here, coupled to `plan_push`'s orientation,
#: rather than alongside `motion.STRAIGHT_LINE`: which index means the
#: vertical depends entirely on which local axis maps to world-up under THIS
#: orientation, so the mapping and the constraint have to move together or
#: they will silently drift apart.
#:
#: Under `multiply(exp((0.0, 0.0, pi / 2)), POINTING_DOWN)` the local axes
#: map to world as:
#:   local +X -> (0, 0, -1)   the vertical -- this is the one to hold
#:   local +Y -> (0, -1, 0)
#:   local +Z -> (-1, 0, 0)   the world x travel axis (sign doesn't matter)
#:
#: The fully strict vector (1,1,1,1,1,0) -- holding all but the travel axis
#: -- was tried first and is what we actually want, but cuRobo reproducibly
#: refused to plan it (`right arm global plan failed: 'Fail'`). The likely
#: cause (not isolated directly): `hold_partial_pose` pins held components to
#: the GOAL pose's values for the whole path, but the arm arrives at its
#: contact pose a few millimetres off what was commanded -- measured,
#: commanded (0.0529, -0.0584, 1.1008) vs. achieved (0.0509, -0.0592,
#: 1.1045). With five of six components pinned to the commanded values, the
#: problem is infeasible from the very first waypoint, since the achieved
#: start already violates them.
#:
#: (1,1,1,1,0,0) -- all three rotations plus local +X, i.e. world-vertical --
#: plans and works well: the blade held its height to 0.2mm across the whole
#: push (z 1.1045 -> 1.1043), leaving a neighbour gap of 0.0035m against a
#: 0.04m threshold, the best of any push variant tried. Height is the part
#: that matters because the failure mode this replaced was an arc that let
#: the blade ride up over the object (an earlier run stalled 62mm high and
#: only nudged the box 3mm) -- vertical drift breaks a push; a few
#: millimetres of lateral bow through the horizontal plane does not, and the
#: planner needs that plane free (the two remaining translation entries) to
#: stay feasible at all.
#:
#: `motion.STRAIGHT_LINE` (1,1,1,0,0,0) is not tight enough for a push: it
#: holds the wrist orientation but leaves all three translation axes free, so
#: the path can still bow -- including vertically -- through the contact
#: instead of holding height, which is the arc failure above.
PUSH_PATH_CONSTRAINT = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True)
class Push:
    """Geometry for compacting an object by pushing it, gripper closed
    throughout.

    A push needs no grasp, so nothing can be dropped or missed; the gripper
    stays closed for the whole motion. Contacting the object's trailing face
    means the arm is always behind the object relative to its direction of
    travel, never straddling it.
    """

    approach_pose: Pose
    start_pose: Pose
    end_pose: Pose
    retreat_pose: Pose
    constraint: tuple[float, float, float, float, float, float]


def plan_push(obj: ObjectState, target_x: float, *, standoff: float) -> Push:
    """Push `obj` from its current (re-observed) position to `target_x`.

    Where the object should end up is decided at placement time, but how to
    move it there must be planned against the object's actual observed pose,
    not a prediction -- hence this takes an `ObjectState`, not a target pose.

    A horizontal approach (approach axis along +/-x) was tried first and
    failed to plan on a live run: for a push toward -x it put the approach
    pose at x=0.254 against the right arm's own base at x=0.4, folding the
    arm back on itself, and cuRobo returned a bare `Fail`. Every motion that
    has ever planned reliably in this workcell is top-down, so the push is
    now top-down too: descend beside the object with a closed gripper,
    translate sideways to shove it, lift away. Only x changes between the
    start and end poses; the approach axis stays vertical throughout.
    """
    obj_x, obj_y, obj_z = obj.pose.position
    if target_x == obj_x:
        raise PlacementError(f"nothing to push: object already at x={obj_x:.3f}")
    direction = 1.0 if target_x > obj_x else -1.0
    extent = push_axis_half_extent(obj)

    # Top-down, fingers separated perpendicular to the direction of travel so
    # both pads meet the object's face flat-on and it cannot slip out
    # sideways between them.
    orientation = multiply(exp((0.0, 0.0, pi / 2)), POINTING_DOWN)

    # The approach axis is vertical now, so GRASP_TCP_OFFSET raises z instead
    # of shifting x. Pads at the object's own centre height are what keeps a
    # sideways shove from tipping it rather than sliding it.
    z = obj_z + GRASP_TCP_OFFSET

    # Derivation (checkable by sign): for a push toward -x the gripper sits
    # on the object's +x side, so its pushing face -- PUSHER_HALF_WIDTH
    # toward -x from the commanded pose -- is at `pose_x - PUSHER_HALF_WIDTH`.
    # During descent that face must clear the object's own trailing face at
    # `obj_x + extent` by CONTACT_CLEARANCE, giving
    # `start_pose.x = obj_x + extent + PUSHER_HALF_WIDTH + CONTACT_CLEARANCE`.
    # At the end of the push that same face must land exactly on the
    # object's target face at `target_x + extent`, giving
    # `end_pose.x = target_x + extent + PUSHER_HALF_WIDTH`. Both generalise
    # to either direction by negating with `direction`.
    start_pose = Pose(
        (obj_x - direction * (extent + PUSHER_HALF_WIDTH + CONTACT_CLEARANCE), obj_y, z),
        orientation,
        WORKCELL_FRAME,
    )
    end_pose = Pose(
        (target_x - direction * (extent + PUSHER_HALF_WIDTH), obj_y, z),
        orientation,
        WORKCELL_FRAME,
    )
    return Push(
        approach_pose=backed_off(start_pose, standoff),
        start_pose=start_pose,
        end_pose=end_pose,
        retreat_pose=backed_off(end_pose, standoff),
        constraint=PUSH_PATH_CONSTRAINT,
    )
