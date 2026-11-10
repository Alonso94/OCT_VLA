import numpy as np
import pytest

from oct_vla.core.action import Action, ArmAction
from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.core.observation import RGBFrame, RobotObservation
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM, ENTITY_TYPES
from oct_vla.data.episode import Episode, Sample
from oct_vla.data.future_keyframes import (
    FUTURE_DIM,
    FUTURE_EVENTS,
    FUTURE_ROLES,
    interaction_events,
    oracle_future_keyframes,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)
_FRAME = RGBFrame(2, 2, bytes(12))
_ACTION = Action(ArmAction((0, 0, 0), (0, 0, 0), 0.0), ArmAction((0, 0, 0), (0, 0, 0), 1.0))


def _pose(x, y=0.0, z=0.0):
    return Pose((x, y, z), IDENTITY, WORKCELL_FRAME)


def _sample(t, phase, target, objects, left=(-0.5, 0, 0.1), right=(0.5, 0, 0.1), grip=1.0):
    eef = EEFState(ArmState(_pose(*left), grip), ArmState(_pose(*right), 1.0))
    scene = ObjectScene(0.0, tuple(ObjectState(k, _pose(*p), (0.05, 0.05, 0.1), 1.0, 1.0)
                                   for k, p in objects.items()))
    return Sample(t * 0.1, RobotObservation(t * 0.1, eef, _FRAME, _FRAME, _FRAME), scene,
                  TaskContext("restock", target), _ACTION, phase)


def _episode():
    """Left arm moves a to the shelf, then b; phases as the oracle labels them."""
    a, b = (-0.5, 0.0, 0.0), (-0.4, 0.1, 0.0)
    a_shelf = (-0.2, 0.3, 0.4)
    script = [
        ("pregrasp", "a", {"a": a, "b": b}, (-0.5, 0, 0.3)),
        ("close", "a", {"a": a, "b": b}, (-0.5, 0, 0.14)),   # 1
        ("close", "a", {"a": a, "b": b}, (-0.5, 0, 0.14)),   # 2: grasp a
        ("lift", "a", {"a": a, "b": b}, (-0.5, 0, 0.3)),
        ("release", "a", {"a": a_shelf, "b": b}, (-0.2, 0.3, 0.54)),  # 4: place a
        ("open", "b", {"a": a_shelf, "b": b}, (-0.4, 0.1, 0.3)),
        ("compact close", "b", {"a": a_shelf, "b": b}, (-0.4, 0.1, 0.3)),  # not an event
        ("close", "b", {"a": a_shelf, "b": b}, (-0.4, 0.1, 0.14)),  # 7: grasp b
        ("release", "b", {"a": a_shelf, "b": (0.0, 0.3, 0.4)}, (0.0, 0.3, 0.54)),  # 8
        ("retreat", "b", {"a": a_shelf, "b": (0.0, 0.3, 0.4)}, (0.0, 0.3, 0.7)),
    ]
    samples = [_sample(t, phase, target, objects, left=left)
               for t, (phase, target, objects, left) in enumerate(script)]
    return Episode(seed=1, instruction="restock", samples=samples, success=True, metadata={})


def test_events_are_run_ends_of_close_and_release_only():
    events = interaction_events(_episode())
    assert [(e.frame, e.kind, e.track_id, e.side) for e in events] == [
        (2, "grasp", "a", "left"), (4, "place", "a", "left"),
        (7, "grasp", "b", "left"), (8, "place", "b", "left")]


def test_each_frame_sees_the_next_event_and_the_end_is_empty():
    keyframes, mask = oracle_future_keyframes(_episode())
    assert keyframes.shape == (10, 3, FUTURE_DIM) and mask.shape == (10, 3)
    event_col = ENTITY_TOKEN_DIM + len(FUTURE_ROLES)
    kinds = [FUTURE_EVENTS[int(np.argmax(k[0, event_col:]))] if m.all() else None
             for k, m in zip(keyframes, mask)]
    # Frames 0-1 precede the grasp at 2; at frame 2 itself the next event is the place.
    assert kinds == ["grasp", "grasp", "place", "place", "grasp", "grasp", "grasp", "place",
                     None, None]
    assert not keyframes[8:].any()


def test_rows_carry_object_gripper_and_goal():
    keyframes, _ = oracle_future_keyframes(_episode())
    obj, grip, goal = keyframes[0]
    np.testing.assert_allclose(obj[:3], (-0.5, 0.0, 0.0))          # a where it is grasped
    np.testing.assert_allclose(grip[:3], (-0.5, 0.0, 0.14))        # the acting left gripper
    np.testing.assert_allclose(goal[:3], (-0.2, 0.3, 0.4))         # a where it is placed
    type_col = ENTITY_TOKEN_DIM - len(ENTITY_TYPES)
    assert ENTITY_TYPES[int(np.argmax(grip[type_col:ENTITY_TOKEN_DIM]))] == "left_gripper"
    roles = keyframes[0][:, ENTITY_TOKEN_DIM:ENTITY_TOKEN_DIM + len(FUTURE_ROLES)]
    np.testing.assert_array_equal(roles, np.eye(3))


def test_ambiguous_acting_arm_is_refused():
    episode = _episode()
    samples = list(episode.samples)
    # Both grippers equally near a at its grasp: no clear acting arm.
    samples[2] = _sample(2, "close", "a", {"a": (0.0, 0, 0), "b": (-0.4, 0.1, 0)},
                         left=(-0.1, 0, 0.1), right=(0.1, 0, 0.1))
    bad = Episode(seed=2, instruction="restock", samples=samples, success=True, metadata={})
    with pytest.raises(ValueError, match="no clear acting arm"):
        interaction_events(bad)


def test_place_without_grasp_is_refused():
    samples = [s for i, s in enumerate(_episode().samples) if i not in (1, 2)]
    bad = Episode(seed=3, instruction="restock", samples=samples, success=True, metadata={})
    with pytest.raises(ValueError, match="without a grasp"):
        interaction_events(bad)
