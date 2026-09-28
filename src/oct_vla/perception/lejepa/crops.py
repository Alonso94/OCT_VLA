"""Per-object crops of the head image, cut by projecting each recorded box.

The head camera never moves in the shelf-restock scene -- `demo_clean` sets
`random_head_camera_dis: 0` and the pose is our fixed override -- so one
intrinsic matrix and one world-to-camera transform serve every frame of every
episode. Neither was recorded with the corpus, so both are rebuilt here the way
RoboTwin builds them (`envs/camera/camera.py`):

* the pose matrix has columns (forward, left, up = forward x left), which is
  SAPIEN's camera-link convention (x forward, y left, z up);
* `get_extrinsic_matrix()` is world -> OpenCV camera (x right, y down,
  z forward), i.e. the inverse pose followed by that axis relabelling;
* `get_intrinsic_matrix()` for a camera created from `fovy` has square pixels,
  `fy = (h / 2) / tan(fovy / 2)` and the principal point at `(w / 2, h / 2)`.

The pinned reference values in the tests were read off SAPIEN 3.0.0b1 itself
(a `RenderCameraComponent` needs no GPU for its matrices), so a change to any
of the three conventions above fails loudly instead of shifting every crop.

Workcell == SAPIEN world for this task (`WORLD_TO_WORKCELL` is the identity in
tasks/shelf_restock/collect.py), and a recorded ObjectState's pose is already
the *centred upright* box: position at the box centre, orientation about which
`size_xyz` is measured. So the eight corners are exact, no mesh offset needed.
"""

from __future__ import annotations

import ast
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oct_vla.core.objects import ObjectScene, ObjectState

#: RoboTwin's D435 entry (env_cfg/task_config/_camera_config.yml), which
#: demo_clean selects as head_camera_type. Pinned so the camera model works
#: without a RoboTwin checkout; `head_camera_from_robotwin` re-reads the file
#: and refuses to disagree with these.
D435_FOVY_DEG = 37.0
D435_WIDTH = 320
D435_HEIGHT = 240
#: SAPIEN's near plane in camera.py. A corner behind it would project through
#: the camera centre and produce a box that looks plausible and is wrong.
NEAR_PLANE = 0.1

#: world axes of an OpenCV camera, written in SAPIEN camera-link axes:
#: x_cv = -y_link (right), y_cv = -z_link (down), z_cv = x_link (forward).
_LINK_TO_CV = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

_ROBOTWIN_ENV = Path(__file__).resolve().parents[2] / "tasks" / "shelf_restock" / "robotwin_env.py"

CROP_SIZE = 64
DEFAULT_PADDING = 0.15
#: Smallest square side, in source pixels, worth cutting. Objects on the far
#: shelf span ~30 px; a box much smaller than this is a projection that is
#: mostly off-image or edge-on, and upsampling it to 64 px invents texture.
MIN_SIDE_PX = 12
#: Fraction of the unpadded projected box that must lie inside the image.
MIN_VISIBLE_FRACTION = 0.6


@dataclass(frozen=True)
class PinholeCamera:
    """An OpenCV-convention pinhole camera: `intrinsic` K and world->camera."""

    intrinsic: np.ndarray  # (3, 3)
    extrinsic: np.ndarray  # (3, 4)
    width: int
    height: int

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (N, 2) and camera-frame depth (N,) of world points."""
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        camera = points @ self.extrinsic[:, :3].T + self.extrinsic[:, 3]
        depth = camera[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = (camera @ self.intrinsic.T)[:, :2] / depth[:, None]
        return pixels, depth


def intrinsic_from_fovy(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """SAPIEN's K for a camera created with `fovy` (square pixels, centred)."""
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


def robotwin_camera_pose(position, forward, left) -> np.ndarray:
    """camera.py's `mat44`: camera-link -> world, columns forward/left/up.

    RoboTwin normalises forward and left but never orthogonalises them, so a
    non-orthogonal pair would reach SAPIEN as a non-rotation and be silently
    projected onto one. Refused here instead.
    """
    forward = np.asarray(forward, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    left = left / np.linalg.norm(left)
    if abs(float(forward @ left)) > 1e-6:
        raise ValueError(f"camera forward {forward} and left {left} are not orthogonal")
    up = np.cross(forward, left)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([forward, left, up], axis=1)
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    return pose


def extrinsic_cv_from_pose(camera_to_world: np.ndarray) -> np.ndarray:
    """SAPIEN's `get_extrinsic_matrix()`: world -> OpenCV camera, (3, 4)."""
    rotation = camera_to_world[:3, :3]
    position = camera_to_world[:3, 3]
    world_to_link = rotation.T
    extrinsic = np.empty((3, 4))
    extrinsic[:, :3] = _LINK_TO_CV @ world_to_link
    extrinsic[:, 3] = -extrinsic[:, :3] @ position
    return extrinsic


def camera_from_placement(
    placement: dict[str, Any],
    *,
    fovy_deg: float = D435_FOVY_DEG,
    width: int = D435_WIDTH,
    height: int = D435_HEIGHT,
) -> PinholeCamera:
    pose = robotwin_camera_pose(placement["position"], placement["forward"], placement["left"])
    return PinholeCamera(
        intrinsic_from_fovy(fovy_deg, width, height), extrinsic_cv_from_pose(pose), width, height
    )


def head_camera_override(source: Path = _ROBOTWIN_ENV) -> dict[str, list[float]]:
    """`HEAD_CAMERA_OVERRIDE`, read from robotwin_env.py's source.

    That module imports RoboTwin (`envs`) at load time and so cannot be
    imported by the policy environment. Parsing its literal rather than
    copying the numbers means the crops cannot drift from the camera the
    corpus was actually recorded with.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "HEAD_CAMERA_OVERRIDE"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if set(value) != {"position", "forward", "left"}:
                raise ValueError(f"unexpected HEAD_CAMERA_OVERRIDE keys {sorted(value)}")
            return {key: [float(v) for v in value[key]] for key in value}
    raise ValueError(f"No HEAD_CAMERA_OVERRIDE assignment in {source}")


def default_head_camera() -> PinholeCamera:
    """The shelf-restock head camera from the pinned D435 entry + our override."""
    return camera_from_placement(head_camera_override())


def head_camera_from_robotwin(root: str | Path) -> PinholeCamera:
    """The head camera exactly as RoboTwin would configure it from `root`.

    Walks the same path collection does -- demo_clean via
    `build_dual_franka_setup`, then `_override_head_camera` -- and checks the
    result against the pinned camera, so a RoboTwin checkout that disagrees
    (another fovy, a randomised head camera) stops the crop build.
    """
    from oct_vla.robots.robotwin.task_config import (
        _load_yaml,
        build_dual_franka_setup,
        resolve_task_config_dir,
    )

    root = Path(root)
    setup = build_dual_franka_setup(root, task_name="shelf_restock", task_config="demo_clean")
    randomization = setup.get("domain_randomization") or {}
    if "random_head_camera_dis" not in randomization:
        raise ValueError("demo_clean does not say whether the head camera is randomised")
    if float(randomization["random_head_camera_dis"]) != 0.0:
        raise ValueError("demo_clean randomises the head camera; one static camera does not apply")
    head_type = setup["camera"]["head_camera_type"]
    camera_types = _load_yaml(resolve_task_config_dir(root) / "_camera_config.yml")
    spec = camera_types[head_type]
    cameras = setup["left_embodiment_config"]["static_camera_list"]
    heads = [camera for camera in cameras if camera.get("name") == "head_camera"]
    if len(heads) != 1:
        raise ValueError(f"expected one head_camera in the embodiment config, found {len(heads)}")
    placement = dict(heads[0])
    placement.update(head_camera_override())
    camera = camera_from_placement(
        placement, fovy_deg=float(spec["fovy"]), width=int(spec["w"]), height=int(spec["h"])
    )
    pinned = default_head_camera()
    if not (
        np.allclose(camera.intrinsic, pinned.intrinsic)
        and np.allclose(camera.extrinsic, pinned.extrinsic)
        and (camera.width, camera.height) == (pinned.width, pinned.height)
    ):
        raise ValueError(f"RoboTwin at {root} configures a head camera other than the pinned one")
    return camera


def rotation_matrix(quaternion_xyzw) -> np.ndarray:
    """Active rotation of an xyzw unit quaternion (core/geometry.py's convention)."""
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def box_corners(position, quaternion_xyzw, size_xyz) -> np.ndarray:
    """The eight world-frame corners (8, 3) of a posed, sized box."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local = signs * (np.asarray(size_xyz, dtype=np.float64) / 2.0)
    return local @ rotation_matrix(quaternion_xyzw).T + np.asarray(position, dtype=np.float64)


def object_corners(obj: ObjectState) -> np.ndarray:
    return box_corners(obj.pose.position, obj.pose.orientation, obj.size_xyz)


def projected_bbox(camera: PinholeCamera, corners: np.ndarray) -> tuple[float, ...] | None:
    """Pixel bbox (x0, y0, x1, y1) of projected corners; None if any is behind."""
    pixels, depth = camera.project(corners)
    if not np.all(depth > NEAR_PLANE):
        return None
    return (
        float(pixels[:, 0].min()),
        float(pixels[:, 1].min()),
        float(pixels[:, 0].max()),
        float(pixels[:, 1].max()),
    )


def square_crop_box(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    padding: float = DEFAULT_PADDING,
    min_side: int = MIN_SIDE_PX,
    min_visible: float = MIN_VISIBLE_FRACTION,
) -> tuple[int, int, int] | None:
    """An in-image square (x0, y0, side) around `bbox`, or None.

    `padding` is added on every side as a fraction of the box's longer edge.
    A square that would cross the border is shifted inside rather than
    truncated, so every crop keeps its aspect and is resized uniformly.
    """
    x0, y0, x1, y1 = bbox
    area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    if area <= 0.0:
        return None
    inside = max(min(x1, width) - max(x0, 0.0), 0.0) * max(min(y1, height) - max(y0, 0.0), 0.0)
    if inside / area < min_visible:
        return None
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * padding)
    side = int(round(min(side, width, height)))
    if side < min_side:
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = min(max(left, 0), width - side)
    top = min(max(top, 0), height - side)
    return left, top, side


def resize_square(patch: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    import cv2

    # INTER_AREA when shrinking, which averages instead of aliasing; the far
    # shelf's crops are *upsampled* and get bilinear.
    interpolation = cv2.INTER_AREA if patch.shape[0] >= size else cv2.INTER_LINEAR
    out = cv2.resize(patch, (size, size), interpolation=interpolation)
    return np.ascontiguousarray(out, dtype=np.uint8)


def crop_object(
    image: np.ndarray,
    camera: PinholeCamera,
    obj: ObjectState,
    *,
    size: int = CROP_SIZE,
    padding: float = DEFAULT_PADDING,
    min_side: int = MIN_SIDE_PX,
    min_visible: float = MIN_VISIBLE_FRACTION,
) -> tuple[np.ndarray, tuple[int, int, int]] | None:
    """The (size, size, 3) uint8 crop of `obj` and its source square, or None."""
    if image.dtype != np.uint8 or image.shape != (camera.height, camera.width, 3):
        raise ValueError(
            f"expected a ({camera.height}, {camera.width}, 3) uint8 image, got "
            f"{image.shape} {image.dtype}"
        )
    bbox = projected_bbox(camera, object_corners(obj))
    if bbox is None:
        return None
    square = square_crop_box(
        bbox,
        camera.width,
        camera.height,
        padding=padding,
        min_side=min_side,
        min_visible=min_visible,
    )
    if square is None:
        return None
    left, top, side = square
    return resize_square(image[top : top + side, left : left + side], size), square


# --------------------------------------------------------------------------- episodes


@dataclass(frozen=True)
class HeadEpisode:
    """One recorded episode's head frames and per-frame object scenes."""

    directory: Path
    seed: int
    metadata: dict[str, str]
    frames: np.ndarray  # (T, H, W, 3) uint8
    scenes: tuple[ObjectScene, ...]
    #: (T, 2, 3) left and right end-effector positions. Kept so a crop can be
    #: tagged with how close a gripper is: a held object's crop is mostly arm.
    eef_positions: np.ndarray


def read_head_episode(directory: str | Path) -> HeadEpisode:
    """Read only the head stream and scenes of a canonical episode.

    `data.store.read_episode` decodes all three cameras and builds a full
    Sample per frame; crops need neither wrist camera, and at ~100 MB of raw
    video per camera per episode that is most of the cost. The per-object
    parsing is still store's own, so the pose convention cannot diverge.
    """
    from oct_vla.data.store import _scene_from_json

    directory = Path(directory)
    payload = json.loads((directory / "episode.json").read_text())
    size = payload["cameras"]["head_camera"]
    width, height = int(size["width"]), int(size["height"])
    with gzip.open(directory / "head_camera.rgb.gz", "rb") as handle:
        raw = handle.read()
    samples = payload["samples"]
    expected = len(samples) * width * height * 3
    if len(raw) != expected:
        raise ValueError(
            f"{directory}: head stream holds {len(raw)} bytes, {len(samples)} samples "
            f"need {expected}"
        )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(len(samples), height, width, 3)
    scenes = tuple(_scene_from_json(sample["scene"]) for sample in samples)
    eef = np.array(
        [
            [sample["eef"][arm]["pose"]["position"] for arm in ("left", "right")]
            for sample in samples
        ],
        dtype=np.float64,
    )
    return HeadEpisode(
        directory, int(payload["seed"]), dict(payload.get("metadata", {})), frames, scenes, eef
    )


def model_id_by_size(
    model: str, assets_root: str | Path, precision: int = 4
) -> dict[tuple[float, ...], int]:
    """size_key -> mesh variant id, from the asset's own metadata.

    Episodes record `model_ids` only as the set spawned in the scene, not per
    object; the variants of 113_coffee-box all differ in size, so size
    recovers the per-object id. Refuses an asset whose variants collide.
    """
    from oct_vla.robots.robotwin.assets import available_model_ids, upright_size

    root = Path(assets_root)
    table: dict[tuple[float, ...], int] = {}
    for model_id in available_model_ids(model, root):
        key = tuple(round(float(v), precision) for v in upright_size(model, model_id, root))
        if key in table:
            raise ValueError(f"{model} variants {table[key]} and {model_id} share size {key}")
        table[key] = int(model_id)
    return table


# --------------------------------------------------------------------------- overlay


def draw_overlay(image: np.ndarray, camera: PinholeCamera, scene: ObjectScene) -> np.ndarray:
    """The frame with each object's projected box edges and crop square drawn.

    The check that the camera model is right: edges must sit on the objects.
    """
    import cv2

    canvas = np.ascontiguousarray(image.copy())
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    colours = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 0, 255)]
    for index, obj in enumerate(scene.objects):
        colour = colours[index % len(colours)]
        pixels, depth = camera.project(object_corners(obj))
        if np.all(depth > NEAR_PLANE):
            for a, b in edges:
                cv2.line(
                    canvas,
                    tuple(int(round(v)) for v in pixels[a]),
                    tuple(int(round(v)) for v in pixels[b]),
                    colour,
                    1,
                    cv2.LINE_AA,
                )
        bbox = projected_bbox(camera, object_corners(obj))
        square = None if bbox is None else square_crop_box(bbox, camera.width, camera.height)
        if square is not None:
            left, top, side = square
            cv2.rectangle(canvas, (left, top), (left + side - 1, top + side - 1), (255, 255, 0), 1)
    return canvas
