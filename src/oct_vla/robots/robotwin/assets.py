"""Read RoboTwin object-asset metadata: which variants exist, and how big they are.

RoboTwin's mesh assets are authored y-up and are spawned upright by a fixed
base quaternion (`qpos=[0.5, 0.5, 0.5, 0.5]` wxyz in RoboTwin's own tasks,
which maps model +y to world +z). Everything here converts that mesh-frame
metadata into the upright, world-aligned terms the canonical `ObjectState`
schema is defined in, so no consumer of an object's pose or size has to know
which asset it came from.

Sizes are read rather than chosen: `create_actor` overwrites its own `scale`
argument with the value inside `model_data<N>.json`, so an asset's real
dimensions are a property of the asset, not something the task can pick.
"""

import json
from pathlib import Path

from oct_vla.core.geometry import Quaternion, Vector3, inverse, multiply, rotate

#: Mesh-frame to upright rotation (xyzw), equivalent to RoboTwin's own
#: `qpos=[0.5, 0.5, 0.5, 0.5]` wxyz. Maps model (x, y, z) to world (y, z, x),
#: so the model's +y becomes world up.
UPRIGHT_ROTATION: Quaternion = (0.5, 0.5, 0.5, 0.5)

#: Highest `model_data<N>.json` index worth probing for. RoboTwin ships a
#: handful of variants per asset and numbers them from zero with no manifest.
_MAX_MODEL_ID = 32


def robotwin_assets_root() -> Path:
    """The `assets/` directory of the RoboTwin checkout currently imported.

    Located from the imported `envs` package rather than from the working
    directory or an env var: by the time this is called RoboTwin is already
    loaded, and that is the checkout whose meshes will actually be built.
    (`create_actor` itself resolves `assets/objects` relative to the process
    working directory, which is why it must not be the source of truth here.)
    """
    import envs

    return Path(envs.__file__).resolve().parent.parent / "assets"


def _model_data_path(modelname: str, model_id: int, root: Path | None = None) -> Path:
    base = root if root is not None else robotwin_assets_root()
    return base / "objects" / modelname / f"model_data{model_id}.json"


def available_model_ids(modelname: str, root: Path | None = None) -> tuple[int, ...]:
    """Variant ids of `modelname` that have usable metadata, lowest first."""
    found = tuple(
        model_id
        for model_id in range(_MAX_MODEL_ID)
        if _model_data_path(modelname, model_id, root).is_file()
    )
    if not found:
        base = root if root is not None else robotwin_assets_root()
        raise FileNotFoundError(f"No model_data<N>.json for {modelname!r} under {base / 'objects'}")
    return found


def upright_size(
    modelname: str,
    model_id: int,
    root: Path | None = None,
    *,
    upright_rotation: Quaternion = UPRIGHT_ROTATION,
) -> Vector3:
    """An asset variant's world-axis-aligned size when standing upright.

    `extents` are the mesh's own bounding box in mesh axes and `scale` is
    per-axis, so the scaled extents are rotated into the upright frame and
    taken as absolute magnitudes. For the axis-permuting rotation RoboTwin
    uses this is an exact relabelling of the three dimensions; the general
    form is used so an asset with some other base orientation still yields a
    bounding box rather than a silently wrong one.
    """
    path = _model_data_path(modelname, model_id, root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        extents, scale = data["extents"], data["scale"]
    except FileNotFoundError as error:
        raise FileNotFoundError(f"No metadata for {modelname!r} model_id={model_id}") from error
    except KeyError as error:
        raise ValueError(f"{path} has no {error.args[0]!r} field") from error

    scaled = tuple(float(e) * float(s) for e, s in zip(extents, scale, strict=True))
    rotated = rotate(upright_rotation, scaled)
    return tuple(abs(value) for value in rotated)


def centered_upright_pose(
    position: Vector3,
    orientation: Quaternion,
    upright_rotation: Quaternion,
    center_offset: Vector3,
) -> tuple[Vector3, Quaternion]:
    """Convert a measured mesh pose into the object's own centred, upright one.

    The single definition of what an asset's pose *means*, shared by the
    perception evidence source and by the planner's world model -- both
    describe the same box, and a discrepancy between them would put cuRobo's
    obstacle somewhere the oracle does not think the object is.
    """
    upright = multiply(orientation, inverse(upright_rotation))
    offset = rotate(upright, center_offset)
    return tuple(p + o for p, o in zip(position, offset, strict=True)), upright


def upright_center_offset(
    modelname: str,
    model_id: int,
    root: Path | None = None,
    *,
    upright_rotation: Quaternion = UPRIGHT_ROTATION,
) -> Vector3:
    """Where the bounding-box centre sits relative to the mesh's own origin,
    in the upright frame.

    These assets are authored with their origin at the object's *base*, not
    its centre: for every `113_coffee-box` variant this offset along world z
    matches the variant's half-height to within a millimetre. Anything that
    treats the actor's reported position as a centre -- `holding_tcp_pose`
    computes an object's top face as `position + height / 2` -- therefore
    aims half an object too low, which was observed live as
    `left/panda_hand <-> restock_object_0` while descending to grasp.
    """
    path = _model_data_path(modelname, model_id, root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        center, scale = data["center"], data["scale"]
    except FileNotFoundError as error:
        raise FileNotFoundError(f"No metadata for {modelname!r} model_id={model_id}") from error
    except KeyError as error:
        raise ValueError(f"{path} has no {error.args[0]!r} field") from error

    scaled = tuple(float(c) * float(s) for c, s in zip(center, scale, strict=True))
    return rotate(upright_rotation, scaled)
