"""JSON encoding for the canonical types that cross the eval bridge.

Only what closed-loop evaluation actually consumes: enough of an `ObjectScene`
and `TaskContext` to rebuild object tokens on the policy side, and the arm state
the policy conditions on. Deliberately narrower than `data/store.py`'s canonical
on-disk schema, which additionally carries segmentation masks and embeddings --
those are recording concerns and would add megabytes per step to a 15 Hz control
loop that never reads them.

Like `protocol`, this is imported by both environments, so it stays pure
standard library and speaks only in the core dataclasses.
"""

from __future__ import annotations

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.core.state import ArmJoints, ArmState, EEFState, JointState


def pose_to_json(pose: Pose) -> dict:
    return {
        "position": list(pose.position),
        "orientation": list(pose.orientation),
        "frame": pose.frame,
    }


def pose_from_json(data: dict) -> Pose:
    return Pose(tuple(data["position"]), tuple(data["orientation"]), data["frame"])


def eef_to_json(eef: EEFState) -> dict:
    return {
        side: {"pose": pose_to_json(arm.pose), "gripper": arm.gripper}
        for side, arm in (("left", eef.left), ("right", eef.right))
    }


def eef_from_json(data: dict) -> EEFState:
    def arm(side: str) -> ArmState:
        return ArmState(pose_from_json(data[side]["pose"]), data[side]["gripper"])

    return EEFState(arm("left"), arm("right"))


def joints_to_json(joints: JointState | None) -> dict | None:
    """Measured joint configuration, or None when the backend reports none.

    Sent alongside the EEF pose rather than instead of it: a joint-space policy
    is conditioned on joints, a Cartesian one on the pose, and the bridge does
    not know which is connected.
    """
    if joints is None:
        return None
    return {
        side: {"positions": list(arm.positions), "gripper": arm.gripper}
        for side, arm in (("left", joints.left), ("right", joints.right))
    }


def joints_from_json(data: dict | None) -> JointState | None:
    if data is None:
        return None
    return JointState(
        *(
            ArmJoints(tuple(data[side]["positions"]), data[side]["gripper"])
            for side in ("left", "right")
        )
    )


def scene_to_json(scene: ObjectScene) -> dict:
    return {
        "timestamp": scene.timestamp,
        "objects": [
            {
                "track_id": obj.track_id,
                "pose": pose_to_json(obj.pose),
                "size_xyz": list(obj.size_xyz),
                "visibility": obj.visibility,
                "confidence": obj.confidence,
                "support_surface": obj.support_surface,
            }
            for obj in scene.objects
        ],
    }


def scene_from_json(data: dict) -> ObjectScene:
    return ObjectScene(
        data["timestamp"],
        tuple(
            ObjectState(
                track_id=obj["track_id"],
                pose=pose_from_json(obj["pose"]),
                size_xyz=tuple(obj["size_xyz"]),
                visibility=obj["visibility"],
                confidence=obj["confidence"],
                support_surface=obj["support_surface"],
            )
            for obj in data["objects"]
        ),
    )


def context_to_json(context: TaskContext) -> dict:
    return {
        "instruction": context.instruction,
        "target_track_id": context.target_track_id,
        "previous_neighbor_track_id": context.previous_neighbor_track_id,
        "phase": context.phase,
    }


def context_from_json(data: dict) -> TaskContext:
    return TaskContext(
        instruction=data["instruction"],
        target_track_id=data["target_track_id"],
        previous_neighbor_track_id=data["previous_neighbor_track_id"],
        phase=data["phase"],
    )
