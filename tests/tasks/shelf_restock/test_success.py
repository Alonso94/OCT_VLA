import pytest

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.tasks.shelf_restock.spec import ObjectVariation, ShelfRegion, ShelfRestockSpec
from oct_vla.tasks.shelf_restock.success import check_placement


def spec(compaction_distance=0.03) -> ShelfRestockSpec:
    return ShelfRestockSpec(
        lower_shelf=ShelfRegion("lower_shelf", (0.0, -0.15, 0.74), (0.2, 0.1, 0.01)),
        upper_shelf=ShelfRegion("upper_shelf", (0.0, 0.10, 1.05), (0.2, 0.1, 0.015)),
        object_variation=ObjectVariation(
            (-0.1, 0.1), (-0.2, -0.1), (-0.2, 0.2), ((0.03, 0.05),) * 3
        ),
        compaction_distance=compaction_distance,
    )


def obj(track_id, position) -> ObjectState:
    return ObjectState(track_id, Pose(position, (0, 0, 0, 1)), (0.04, 0.04, 0.04), 1.0, 1.0)


def test_check_placement_true_without_a_previous_neighbor():
    scene = ObjectScene(0.0, (obj("target", (0.0, 0.10, 1.05)),))
    context = TaskContext(instruction="restock", target_track_id="target")

    result = check_placement(scene, context, spec())

    assert result.target_on_upper_shelf is True
    assert result.target_to_neighbor_gap is None
    assert result.compacted_to_neighbor is None
    assert result.success is True


def test_check_placement_false_when_target_still_on_lower_shelf():
    scene = ObjectScene(0.0, (obj("target", (0.0, -0.15, 0.74)),))
    context = TaskContext(instruction="restock", target_track_id="target")

    result = check_placement(scene, context, spec())

    assert result.target_on_upper_shelf is False
    assert result.success is False


def test_check_placement_requires_compaction_when_neighbor_present():
    scene = ObjectScene(
        0.0,
        (
            obj("target", (0.0, 0.10, 1.05)),
            obj("neighbor", (0.10, 0.10, 1.05)),
        ),
    )
    context = TaskContext(
        instruction="restock", target_track_id="target", previous_neighbor_track_id="neighbor"
    )

    close = check_placement(scene, context, spec(compaction_distance=0.07))
    assert close.compacted_to_neighbor is True
    assert close.success is True

    far = check_placement(scene, context, spec(compaction_distance=0.03))
    assert far.compacted_to_neighbor is False
    assert far.success is False
    # 0.10 between centres, minus each object's 0.02 horizontal radius.
    assert far.target_to_neighbor_gap == pytest.approx(0.06)


def test_check_placement_raises_for_unknown_target():
    scene = ObjectScene(0.0, ())
    context = TaskContext(instruction="restock", target_track_id="missing")
    with pytest.raises(ValueError):
        check_placement(scene, context, spec())


def test_check_placement_raises_for_unknown_neighbor():
    scene = ObjectScene(0.0, (obj("target", (0.0, 0.10, 1.05)),))
    context = TaskContext(
        instruction="restock", target_track_id="target", previous_neighbor_track_id="missing"
    )
    with pytest.raises(ValueError):
        check_placement(scene, context, spec())


def test_gap_is_measured_between_surfaces_so_touching_objects_can_pass():
    """Centre-to-centre could never express "compacted": two touching objects
    are already (width_a + width_b) / 2 apart."""
    big = ObjectState("target", Pose((0.0, 0.0, 0.95), (0, 0, 0, 1)), (0.06, 0.06, 0.04), 1.0, 1.0)
    neighbour = ObjectState(
        "neighbor", Pose((0.062, 0.0, 0.95), (0, 0, 0, 1)), (0.06, 0.06, 0.04), 1.0, 1.0
    )
    scene = ObjectScene(0.0, (big, neighbour))
    context = TaskContext(
        instruction="restock", target_track_id="target", previous_neighbor_track_id="neighbor"
    )
    result = check_placement(scene, context, spec(compaction_distance=0.01))
    # Centres are 0.062 apart but the surfaces are nearly touching.
    assert result.target_to_neighbor_gap == pytest.approx(0.002)
    assert result.compacted_to_neighbor is True


def test_gap_scales_with_object_size_not_just_centre_distance():
    """Same centre distance, different sizes -> different gap."""
    small = ObjectState(
        "target", Pose((0.0, 0.0, 0.95), (0, 0, 0, 1)), (0.03, 0.03, 0.04), 1.0, 1.0
    )
    neighbour = ObjectState(
        "neighbor", Pose((0.10, 0.0, 0.95), (0, 0, 0, 1)), (0.03, 0.03, 0.04), 1.0, 1.0
    )
    context = TaskContext(
        instruction="restock", target_track_id="target", previous_neighbor_track_id="neighbor"
    )
    small_gap = check_placement(
        ObjectScene(0.0, (small, neighbour)), context, spec()
    ).target_to_neighbor_gap
    assert small_gap == pytest.approx(0.07)
