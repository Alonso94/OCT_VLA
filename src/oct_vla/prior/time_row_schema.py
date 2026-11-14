"""The time-row data contract (TIME_ROW_V1): rows are observed time steps, labels are the
next event's paired left/right gripper targets.

A support demonstration is a table whose **rows are time steps** (keyframes, optionally
with interpolated states between them). Each row has:
- input column groups: a variable-size entity set with typed attributes;
- **two labelled target columns:** the left and the right gripper at the next event.

A query is the episode's initial and current scene with both target columns masked; it
is predicted from the supports with fixed weights. `paper/5_time_row_pfn_design.md` is
the full rationale. This module only names the fields; `time_row_batches.py` fills them
from the unchanged v1 generator.

**Definitions** (v1 generator, `generator.execute`):

- **Shared boundary τ(t):** the first keyframe strictly after row t. Both arms are read
  at the *same* τ, never at their own next events.
- **Participation** of arm a at τ: a is the arm whose contact mode changes at τ, i.e.
  `keyframe.gripper`:
  - grasp: the grasping arm;
  - regrasp: the receiving holder;
  - release: the giver;
  - place: the holder.

  v1 has exactly one participating arm per event.
- **Engaged** arm a at τ: its gripper is closed at τ (aperture < OPEN). In v1 a closed
  gripper always holds an object. A non-participating but engaged arm still matters to
  coordination: at a regrasp the giver carries the object to the meeting point.
- **Arm event:** the keyframe's event if the arm participates, else HOLD.
- **Arm object:** the acted object's column if the arm participates, else −1.
- **Arm target pose:** that arm's recorded gripper point and aperture at τ (`feats[n+a]`),
  for every arm.
  - An idle arm can still move at τ: the post-place 10 cm lift, the giver's carry to
    the meeting point.
  - Whether an idle arm's pose is supervised is a loss decision; `participation` and
    `engaged` make it explicit.
- **Group fields:** the event type, the acted object, and that object's **final goal
  pose** (`Episode.final_pose`).
- **Terminal:** rows at or after the last keyframe have no next event; `target_valid` = 0
  and every label is a masked default.
- **Coordinates:** workcell metres. There is no orientation: v1 gripper rows carry
  position and aperture only, and object rows a planar yaw. `orientation` and `codes`
  are reserved and always masked in v1.
"""

from __future__ import annotations

from .schema import EVENTS, GRASP, N_FEATURES, PLACE, REGRASP, RELEASE

SCHEMA_VERSION = "time-row-v1"

ARMS = ("left", "right")
N_ARMS = len(ARMS)

#: Input attribute groups of an entity cell: v1's feature columns, named.
ATTRIBUTE_GROUPS = {"position": (0, 3), "yaw": (3, 5), "size": (5, 8), "aperture": (8, 9)}
assert ATTRIBUTE_GROUPS["aperture"][1] == N_FEATURES

#: Per-arm event label: HOLD (0) or the v1 event the arm takes part in. The indices of
#: grasp..release equal v1's `EVENTS` indices, so no remapping is needed.
HOLD = 0
ARM_EVENTS = ("hold",) + EVENTS[1:]
assert ARM_EVENTS.index("grasp") == GRASP and ARM_EVENTS.index("release") == RELEASE
#: Group event label: NONE (0) for a terminal row, else the v1 event.
NO_EVENT = 0
GROUP_EVENTS = ("none",) + EVENTS[1:]
EVENTS_WITH_TWO_ENGAGED = (REGRASP, RELEASE)
assert PLACE == GROUP_EVENTS.index("place")

#: Width of a goal or object pose: x, y, z, sin(yaw), cos(yaw), as v1's targets.
POSE_DIM = 5
#: Reserved fields: present in the schema, always masked in v1.
RESERVED = {"orientation": 4, "codes": 8}

#: --- TIME_ROW_V2: the v2 grammar (programs_v2, generator_v2) ---
#: Adds the push events (`contact`, `push`) and gripper orientation: a 6D rotation
#: (approach axis, closing axis) per gripper, in the inputs and as a per-arm target.
SCHEMA_VERSION_V2 = "time-row-v2"
ROT_DIM = 6

#: Label arrays of one row, with their per-row shape. Support labels carry a [K, T]
#: prefix, query targets none.
ROW_LABELS = {
    "y_pos": (N_ARMS, 3),
    "y_aperture": (N_ARMS,),
    "y_participation": (N_ARMS,),
    "y_engaged": (N_ARMS,),
    "y_event": (N_ARMS,),
    "y_object": (N_ARMS,),
    "g_event": (),
    "g_object": (),
    "g_goal": (POSE_DIM,),
    "target_valid": (),
}


def _v2_events():
    from .generator_v2 import EVENTS_V2

    return EVENTS_V2


ARM_EVENTS_V2 = ("hold",) + _v2_events()[1:]

#: Optional (``TimeRowConfig.waypoints``): each arm's approach waypoint before its next
#: event, ``data/future_keyframes.approach_waypoint`` above the acting arm's target with
#: the other objects of the keyframe as obstacles; the idle arm's is its own target.
WAYPOINT_LABELS = {"y_wp": (2, 3)}
WAYPOINT_SUFFIX = "+wp"
GROUP_EVENTS_V2 = ("none",) + _v2_events()[1:]
ROW_LABELS_V2 = {**ROW_LABELS, "y_rot": (N_ARMS, ROT_DIM)}
