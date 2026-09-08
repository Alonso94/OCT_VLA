import json

import pytest

from oct_vla.core.geometry import exp, multiply, rotate
from oct_vla.robots.robotwin.assets import (
    UPRIGHT_ROTATION,
    available_model_ids,
    centered_upright_pose,
    upright_center_offset,
    upright_size,
)


def write_model(root, modelname, model_id, extents, scale):
    directory = root / "objects" / modelname
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"model_data{model_id}.json").write_text(
        json.dumps({"extents": list(extents), "scale": list(scale)}), encoding="utf-8"
    )


def test_upright_rotation_stands_the_mesh_up():
    """RoboTwin's assets are y-up; the base rotation must send model +y to
    world +z, or every object would be spawned lying on its side."""
    assert rotate(UPRIGHT_ROTATION, (0.0, 1.0, 0.0)) == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)


def test_size_is_the_scaled_extents_relabelled_by_that_rotation(tmp_path):
    write_model(tmp_path, "asset", 0, (1.0, 2.0, 3.0), (0.1, 0.1, 0.1))
    # model x -> world y, model y -> world z, model z -> world x.
    assert upright_size("asset", 0, tmp_path) == pytest.approx((0.3, 0.1, 0.2))


def test_per_axis_scale_is_applied_before_rotating(tmp_path):
    write_model(tmp_path, "asset", 0, (1.0, 1.0, 1.0), (0.1, 0.2, 0.3))
    assert upright_size("asset", 0, tmp_path) == pytest.approx((0.3, 0.1, 0.2))


def test_size_is_never_negative_whatever_the_rotation(tmp_path):
    write_model(tmp_path, "asset", 0, (1.0, 2.0, 3.0), (0.1, 0.1, 0.1))
    size = upright_size("asset", 0, tmp_path, upright_rotation=exp((0.0, 0.0, 3.14159265358979)))
    assert all(value > 0 for value in size)


def test_available_model_ids_finds_every_variant_in_order(tmp_path):
    for model_id in (2, 0, 5):
        write_model(tmp_path, "asset", model_id, (1.0, 1.0, 1.0), (0.1, 0.1, 0.1))
    assert available_model_ids("asset", tmp_path) == (0, 2, 5)


def test_an_asset_with_no_metadata_is_an_error_not_an_empty_list(tmp_path):
    (tmp_path / "objects" / "asset").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No model_data"):
        available_model_ids("asset", tmp_path)


def test_missing_fields_name_the_file_and_the_field(tmp_path):
    directory = tmp_path / "objects" / "asset"
    directory.mkdir(parents=True)
    (directory / "model_data0.json").write_text(json.dumps({"scale": [1, 1, 1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="extents"):
        upright_size("asset", 0, tmp_path)


def test_a_yaw_applied_on_top_of_the_base_rotation_stays_a_yaw():
    """How the task spawns objects: yaw about world z composed onto the base
    rotation. Dividing the base back out must leave exactly that yaw, which
    is what makes the canonical pose meaningful to yaw-only consumers."""
    from oct_vla.core.geometry import inverse, log

    yaw = 0.37
    spawned = multiply(exp((0.0, 0.0, yaw)), UPRIGHT_ROTATION)
    recovered = multiply(spawned, inverse(UPRIGHT_ROTATION))
    assert log(recovered) == pytest.approx((0.0, 0.0, yaw), abs=1e-9)


def test_center_offset_is_read_and_rotated_into_the_upright_frame(tmp_path):
    directory = tmp_path / "objects" / "asset"
    directory.mkdir(parents=True)
    (directory / "model_data0.json").write_text(
        json.dumps({"center": [0.0, 1.0, 0.0], "scale": [0.1, 0.1, 0.1]}), encoding="utf-8"
    )
    # model +y is world up, so a centre 1.0 up the mesh's y is +0.1 in world z.
    assert upright_center_offset("asset", 0, tmp_path) == pytest.approx((0.0, 0.0, 0.1))


def test_centered_upright_pose_undoes_exactly_how_the_task_spawns_an_object():
    """The scene places the mesh origin so the object's centre lands where it
    wants it; reading the pose back must recover that centre, or the two
    disagree about where the object is."""
    from oct_vla.core.geometry import rotate as _rotate

    center, yaw, offset = (0.2, -0.25, 0.79), 0.6, (0.0, 0.0, 0.038)
    orientation = exp((0.0, 0.0, yaw))
    origin = tuple(c - r for c, r in zip(center, _rotate(orientation, offset), strict=True))

    recovered, recovered_orientation = centered_upright_pose(
        origin, multiply(orientation, UPRIGHT_ROTATION), UPRIGHT_ROTATION, offset
    )
    assert recovered == pytest.approx(center)
    assert recovered_orientation == pytest.approx(orientation)


def test_an_asset_whose_origin_is_already_its_centre_is_left_alone():
    position, orientation = (0.1, 0.2, 0.3), exp((0.0, 0.0, 0.0))
    recovered, _ = centered_upright_pose(position, orientation, orientation, (0.0, 0.0, 0.0))
    assert recovered == pytest.approx(position)
