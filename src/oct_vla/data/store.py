"""Pure-stdlib on-disk store for an Episode: JSON metadata + gzip-raw RGB streams.

Raw concatenated RGB bytes rather than PNG or MP4, because the package takes
no runtime dependencies (`dependencies = []` in pyproject.toml is deliberate:
no numpy/imageio/pillow) and because the project's plan makes a LeRobot
export a downstream adapter that reads FROM this canonical form -- the
stored form only has to be complete and lossless, never itself playable.
gzip is pure stdlib and costs nothing extra to reach for; frames are fixed
size within an episode (validate_episode enforces this), so frame i of a
camera is always a known byte slice and needs no per-frame index.
"""

import gzip
import json
from base64 import b64decode, b64encode
from pathlib import Path

from oct_vla.core.action import Action
from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.core.observation import RGBFrame, RobotObservation
from oct_vla.core.state import ArmJoints, ArmState, EEFState, JointState

from .episode import Episode, Sample

#: Camera name -> RobotObservation attribute, and the on-disk filename stem.
_CAMERA_ATTRS = {
    "head_camera": "head_rgb",
    "left_wrist_camera": "left_wrist_rgb",
    "right_wrist_camera": "right_wrist_rgb",
}


def _pose_to_json(pose: Pose) -> dict:
    return {
        "position": list(pose.position),
        "orientation": list(pose.orientation),
        "frame": pose.frame,
    }


def _pose_from_json(data: dict) -> Pose:
    return Pose(tuple(data["position"]), tuple(data["orientation"]), data["frame"])


def _arm_to_json(arm: ArmState) -> dict:
    return {"pose": _pose_to_json(arm.pose), "gripper": arm.gripper}


def _joints_to_json(joints: ArmJoints) -> dict:
    return {"positions": list(joints.positions), "gripper": joints.gripper}


def _joints_from_json(data: dict) -> ArmJoints:
    return ArmJoints(tuple(data["positions"]), data["gripper"])


def _arm_from_json(data: dict) -> ArmState:
    return ArmState(_pose_from_json(data["pose"]), data["gripper"])


def _object_to_json(obj: ObjectState) -> dict:
    return {
        "track_id": obj.track_id,
        "pose": _pose_to_json(obj.pose),
        "size_xyz": list(obj.size_xyz),
        "visibility": obj.visibility,
        "confidence": obj.confidence,
        "support_surface": obj.support_surface,
        "mask": b64encode(obj.mask).decode("ascii") if obj.mask is not None else None,
        "embedding": list(obj.embedding) if obj.embedding is not None else None,
        "category": obj.category,
    }


def _object_from_json(data: dict) -> ObjectState:
    return ObjectState(
        track_id=data["track_id"],
        pose=_pose_from_json(data["pose"]),
        size_xyz=tuple(data["size_xyz"]),
        visibility=data["visibility"],
        confidence=data["confidence"],
        support_surface=data["support_surface"],
        mask=b64decode(data["mask"]) if data["mask"] is not None else None,
        embedding=tuple(data["embedding"]) if data["embedding"] is not None else None,
        # `.get`, not `[...]`: every recording made before the multi-category
        # change lacks the key, and those corpora must stay readable.
        category=data.get("category"),
    )


def _scene_to_json(scene: ObjectScene) -> dict:
    return {"timestamp": scene.timestamp, "objects": [_object_to_json(o) for o in scene.objects]}


def _scene_from_json(data: dict) -> ObjectScene:
    return ObjectScene(data["timestamp"], tuple(_object_from_json(o) for o in data["objects"]))


def _context_to_json(context: TaskContext) -> dict:
    return {
        "instruction": context.instruction,
        "target_track_id": context.target_track_id,
        "previous_neighbor_track_id": context.previous_neighbor_track_id,
        "phase": context.phase,
    }


def _context_from_json(data: dict) -> TaskContext:
    return TaskContext(
        instruction=data["instruction"],
        target_track_id=data["target_track_id"],
        previous_neighbor_track_id=data["previous_neighbor_track_id"],
        phase=data["phase"],
    )


def write_episode(episode: Episode, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    camera_sizes: dict[str, dict[str, int]] = {}
    camera_bytes: dict[str, bytearray] = {name: bytearray() for name in _CAMERA_ATTRS}
    sample_records = []
    for sample in episode.samples:
        observation = sample.observation
        for name, attr in _CAMERA_ATTRS.items():
            frame: RGBFrame = getattr(observation, attr)
            # Fixed-size-per-episode is a data-contract invariant
            # (validate_episode checks it); the first frame seen sets the
            # size every later frame is assumed to share.
            camera_sizes.setdefault(name, {"width": frame.width, "height": frame.height})
            camera_bytes[name].extend(frame.data)
        sample_records.append(
            {
                "timestamp": sample.timestamp,
                "phase": sample.phase,
                "context": _context_to_json(sample.context),
                "action": list(sample.action.to_vector()),
                "eef": {
                    "left": _arm_to_json(observation.eef.left),
                    "right": _arm_to_json(observation.eef.right),
                },
                # Optional: recordings made before joint capture have no
                # "joints" key, and must stay readable.
                **(
                    {}
                    if observation.joints is None
                    else {
                        "joints": {
                            "left": _joints_to_json(observation.joints.left),
                            "right": _joints_to_json(observation.joints.right),
                        }
                    }
                ),
                "scene": _scene_to_json(sample.scene),
            }
        )

    payload = {
        "seed": episode.seed,
        "instruction": episode.instruction,
        "success": episode.success,
        "metadata": dict(episode.metadata),
        "cameras": camera_sizes,
        "samples": sample_records,
    }
    (directory / "episode.json").write_text(json.dumps(payload, indent=2))
    for name in _CAMERA_ATTRS:
        with gzip.open(directory / f"{name}.rgb.gz", "wb") as handle:
            handle.write(bytes(camera_bytes[name]))
    return directory


def read_episode(directory: str | Path) -> Episode:
    directory = Path(directory)
    payload = json.loads((directory / "episode.json").read_text())

    raw_cameras: dict[str, bytes] = {}
    for name in _CAMERA_ATTRS:
        with gzip.open(directory / f"{name}.rgb.gz", "rb") as handle:
            raw_cameras[name] = handle.read()

    samples = []
    for index, record in enumerate(payload["samples"]):
        frames = {}
        for name in _CAMERA_ATTRS:
            size = payload["cameras"][name]
            frame_bytes = size["width"] * size["height"] * 3
            start = index * frame_bytes
            frames[name] = RGBFrame(
                size["width"], size["height"], raw_cameras[name][start : start + frame_bytes]
            )
        eef = EEFState(
            _arm_from_json(record["eef"]["left"]), _arm_from_json(record["eef"]["right"])
        )
        # sample.timestamp doubles as the observation's own timestamp: both
        # are derived from the same tick clock at capture time (see
        # EpisodeRecorder), so there is nothing to reconcile between them.
        raw_joints = record.get("joints")
        joints = (
            None
            if raw_joints is None
            else JointState(
                _joints_from_json(raw_joints["left"]), _joints_from_json(raw_joints["right"])
            )
        )
        observation = RobotObservation(
            record["timestamp"],
            eef,
            frames["head_camera"],
            frames["left_wrist_camera"],
            frames["right_wrist_camera"],
            joints=joints,
        )
        samples.append(
            Sample(
                timestamp=record["timestamp"],
                observation=observation,
                scene=_scene_from_json(record["scene"]),
                context=_context_from_json(record["context"]),
                action=Action.from_vector(record["action"]),
                phase=record["phase"],
            )
        )

    return Episode(
        seed=payload["seed"],
        instruction=payload["instruction"],
        samples=tuple(samples),
        success=payload["success"],
        metadata=payload["metadata"],
    )
