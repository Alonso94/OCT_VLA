from math import pi

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.geometry import exp
from oct_vla.core.objects import ObjectRole, ObjectScene, ObjectState, TaskContext, role_of


def pose() -> Pose:
    return Pose((0.1, 0.2, 0.3), exp((0, 0, pi / 4)))


def state(track_id="a", **overrides) -> ObjectState:
    fields = dict(
        track_id=track_id,
        pose=pose(),
        size_xyz=(0.05, 0.05, 0.1),
        visibility=1.0,
        confidence=1.0,
    )
    fields.update(overrides)
    return ObjectState(**fields)


def test_object_state_requires_workcell_pose():
    with pytest.raises(ValueError):
        ObjectState(
            track_id="a",
            pose=Pose((0, 0, 0), (0, 0, 0, 1), "world"),
            size_xyz=(0.1, 0.1, 0.1),
            visibility=1.0,
            confidence=1.0,
        )


def test_object_state_default_pose_frame_is_workcell():
    assert state().pose.frame == WORKCELL_FRAME


@pytest.mark.parametrize("size", [(0, 0.1, 0.1), (-0.1, 0.1, 0.1), (0.1, float("nan"), 0.1)])
def test_object_state_rejects_nonpositive_or_nonfinite_size(size):
    with pytest.raises(ValueError):
        state(size_xyz=size)


@pytest.mark.parametrize("field", ["visibility", "confidence"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_object_state_rejects_out_of_range_visibility_confidence(field, value):
    with pytest.raises(ValueError):
        state(**{field: value})


def test_object_state_rejects_empty_track_id():
    with pytest.raises(ValueError):
        state(track_id="")


def test_object_state_optional_fields_default_to_none():
    o = state()
    assert o.support_surface is None
    assert o.mask is None
    assert o.embedding is None


def test_object_state_embedding_must_be_finite():
    with pytest.raises(ValueError):
        state(embedding=(1.0, float("nan")))
    assert state(embedding=(1.0, -2.0)).embedding == (1.0, -2.0)


def test_object_scene_rejects_duplicate_track_ids():
    with pytest.raises(ValueError):
        ObjectScene(0.0, (state("a"), state("a")))


def test_object_scene_rejects_invalid_timestamp():
    with pytest.raises(ValueError):
        ObjectScene(-1.0, (state("a"),))


def test_object_scene_get_by_track_id():
    scene = ObjectScene(0.0, (state("a"), state("b")))
    assert scene.get("b").track_id == "b"
    assert scene.get("missing") is None


def test_task_context_requires_nonempty_instruction_and_target():
    with pytest.raises(ValueError):
        TaskContext(instruction="", target_track_id="a")
    with pytest.raises(ValueError):
        TaskContext(instruction="restock", target_track_id="")


def test_task_context_rejects_target_equal_to_previous_neighbor():
    with pytest.raises(ValueError):
        TaskContext(instruction="restock", target_track_id="a", previous_neighbor_track_id="a")


def test_role_of_derives_role_from_context_not_identity():
    context = TaskContext(
        instruction="restock a", target_track_id="a", previous_neighbor_track_id="b"
    )
    assert role_of(context, "a") is ObjectRole.TARGET
    assert role_of(context, "b") is ObjectRole.PREVIOUS_NEIGHBOR
    assert role_of(context, "c") is ObjectRole.OTHER


def test_role_of_without_previous_neighbor():
    context = TaskContext(instruction="restock a", target_track_id="a")
    assert role_of(context, "b") is ObjectRole.OTHER
