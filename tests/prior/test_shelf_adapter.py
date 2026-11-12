"""The learned future-keyframe source's adapter (prior/shelf_adapter.py)."""

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oct_vla.core.action import Action, ArmAction  # noqa: E402
from oct_vla.core.frames import WORKCELL_FRAME, Pose  # noqa: E402
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext  # noqa: E402
from oct_vla.core.observation import RGBFrame, RobotObservation  # noqa: E402
from oct_vla.core.state import ArmState, EEFState  # noqa: E402
from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM, shelf_support_entities  # noqa: E402
from oct_vla.data.episode import Episode, Sample  # noqa: E402
from oct_vla.data.future_keyframes import FUTURE_DIM, FUTURE_EVENTS, FUTURE_ROLES  # noqa: E402
from oct_vla.prior import shelf_adapter as A  # noqa: E402
from oct_vla.prior.generator import N_MAX  # noqa: E402
from oct_vla.prior.model import KeyframePFN  # noqa: E402
from oct_vla.prior.schema import GRASP, INITIAL, LEFT_GRIPPER, MOVABLE, PLACE, SUPPORT  # noqa: E402
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC  # noqa: E402

IDENTITY = (0.0, 0.0, 0.0, 1.0)
SUPPORTS = shelf_support_entities(DEFAULT_SPEC)
_FRAME = RGBFrame(2, 2, bytes(12))
_ACTION = Action(ArmAction((0, 0, 0), (0, 0, 0), 0.0), ArmAction((0, 0, 0), (0, 0, 0), 1.0))
EVENT_COL = ENTITY_TOKEN_DIM + len(FUTURE_ROLES)


def _scene(**positions):
    return ObjectScene(0.0, tuple(
        ObjectState(k, Pose(p, IDENTITY, WORKCELL_FRAME), (0.05, 0.05, 0.08), 1.0, 1.0)
        for k, p in positions.items()))


def _eef(left=(-0.4, -0.2, 1.1), right=(0.4, -0.2, 1.1), grip=1.0):
    return EEFState(ArmState(Pose(left, IDENTITY, WORKCELL_FRAME), grip),
                    ArmState(Pose(right, IDENTITY, WORKCELL_FRAME), 1.0))


def test_the_query_takes_no_task_context():
    params = inspect.signature(A.query).parameters
    assert list(params) == ["initial_scene", "initial_eef", "scene", "eef", "supports"]
    assert not any("context" in p or "target" in p for p in params)


def test_columns_follow_tracks_not_geometry():
    start = _scene(b=(-0.3, -0.25, 0.79), a=(0.1, -0.25, 0.79))
    # a moves to the far left: geometric sorting would swap the two columns.
    now = _scene(b=(-0.3, -0.25, 0.79), a=(-0.5, -0.25, 0.79))
    q = A.query(start, _eef(), now, _eef(), SUPPORTS)
    assert q.order == ("a", "b")
    np.testing.assert_allclose(q.feats[0, 0, :3], (0.1, -0.25, 0.79), atol=1e-6)
    np.testing.assert_allclose(q.feats[1, 0, :3], (-0.5, -0.25, 0.79), atol=1e-6)
    assert list(q.types[:6]) == [MOVABLE, MOVABLE, LEFT_GRIPPER, LEFT_GRIPPER + 1, SUPPORT,
                                 SUPPORT]
    assert (q.types[6:] == -1).all()


def test_gripper_heights_move_to_the_priors_reference_point():
    q = A.query(_scene(a=(0, -0.25, 0.79)), _eef(), _scene(a=(0, -0.25, 0.79)), _eef(), SUPPORTS)
    assert q.feats[1, 1, 2] == pytest.approx(1.1 - A.EEF_TO_GRIP)


def test_a_track_leaving_the_scene_is_refused():
    with pytest.raises(ValueError, match="left the scene"):
        A.query(_scene(a=(0, -0.25, 0.79), b=(0.2, -0.25, 0.79)), _eef(),
                _scene(a=(0, -0.25, 0.79)), _eef(), SUPPORTS)


def _sample(t, phase, target, objects, left):
    scene = _scene(**objects)
    return Sample(t * 0.1, RobotObservation(t * 0.1, _eef(left=left), _FRAME, _FRAME, _FRAME),
                  scene, TaskContext("restock", target), _ACTION, phase)


def _demo():
    a, b = (-0.3, -0.25, 0.79), (0.1, -0.25, 0.79)
    a_up = (-0.1, -0.06, 0.975)
    script = [("open", "a", {"a": a, "b": b}, (-0.3, -0.25, 1.0)),
              ("close", "a", {"a": a, "b": b}, (-0.3, -0.25, 0.94)),        # 1 grasp a
              ("release", "a", {"a": a_up, "b": b}, (-0.1, -0.06, 1.12)),   # 2 place a
              ("close", "b", {"a": a_up, "b": b}, (0.1, -0.25, 0.94)),       # 3 grasp b
              ("release", "b", {"a": a_up, "b": (0.0, -0.06, 0.975)}, (0.0, -0.06, 1.12))]
    samples = [_sample(t, *step) for t, step in enumerate(script)]
    return Episode(seed=4, instruction="restock", samples=samples, success=True, metadata={})


def test_a_support_demo_is_the_initial_frame_and_every_grasp_and_place():
    demo = A.support_demo(_demo(), SUPPORTS)
    assert demo.rows.sum() == 5
    assert list(demo.event[:5]) == [INITIAL, GRASP, PLACE, GRASP, PLACE]
    assert list(demo.gripper[:5]) == [-1, 0, 0, 0, 0]
    assert not demo.flag[0].any()
    # Row 1 grasps a (column 0) with the left gripper (column 2); row 3 grasps b.
    assert set(np.flatnonzero(demo.flag[1])) == {0, 2}
    assert set(np.flatnonzero(demo.flag[3])) == {1, 2}
    np.testing.assert_allclose(demo.feats[2, 0, :3], (-0.1, -0.06, 0.975), atol=1e-6)


def _out(pointer_col, event="place", side=1, goal_only=False):
    pointer = np.full((1, N_MAX), -1e9, np.float32)
    pointer[0, pointer_col] = 5.0
    out = {"pointer": pointer, "final": np.array([[0.05, -0.06, 0.97, 0.0, 1.0]], np.float32)}
    if goal_only:
        return out
    ev = np.zeros((1, 4), np.float32)
    ev[0, FUTURE_EVENTS.index(event)] = 1.0
    grip = np.zeros((1, 2), np.float32)
    grip[0, side] = 1.0
    return {**out, "event": ev, "gripper": grip,
            "obj": np.array([[0.0, -0.1, 1.0, 0.0, 1.0]], np.float32),
            "grip_pos": np.array([[0.0, -0.1, 1.02]], np.float32),
            "aperture": np.array([0.4], np.float32)}


def test_rows_decode_the_prediction_in_the_eef_convention():
    q = A.query(_scene(a=(0, -0.25, 0.79), b=(0.3, -0.25, 0.79)), _eef(),
                _scene(a=(0, -0.25, 0.79), b=(0.3, -0.25, 0.79)), _eef(), SUPPORTS)
    rows, mask = A.keyframe_rows(_out(1), 0, q)
    assert rows.shape == (3, FUTURE_DIM) and mask.all()
    assert A.predicted_track(_out(1), 0, q) == "b"
    np.testing.assert_allclose(rows[0, :3], (0.0, -0.1, 1.0), atol=1e-6)
    np.testing.assert_allclose(rows[1, :3], (0.0, -0.1, 1.02 + A.EEF_TO_GRIP), atol=1e-6)
    np.testing.assert_allclose(rows[2, :3], (0.05, -0.06, 0.97), atol=1e-6)
    np.testing.assert_allclose(rows[0, 9:12], (0.05, 0.05, 0.08), atol=1e-6)  # b's size
    assert rows[1, 12] == pytest.approx(0.4)
    assert rows[1, ENTITY_TOKEN_DIM - 2] == 1.0  # right_gripper type
    assert (rows[:, EVENT_COL + FUTURE_EVENTS.index("place")] == 1.0).all()


def test_a_goal_only_prior_masks_the_gripper_row_and_names_no_event():
    q = A.query(_scene(a=(0, -0.25, 0.79)), _eef(), _scene(a=(0, -0.25, 0.79)), _eef(), SUPPORTS)
    rows, mask = A.keyframe_rows(_out(0, goal_only=True), 0, q)
    assert list(mask) == [True, False, True]
    assert not rows[1].any() and not rows[:, EVENT_COL:].any()
    np.testing.assert_allclose(rows[0, :3], (0, -0.25, 0.79), atol=1e-6)  # where it is now


def test_a_real_prior_runs_on_the_adapter_and_returns_latent_cells():
    torch.manual_seed(0)
    model = KeyframePFN(width=16, heads=2, layers=1)
    demo = A.support_demo(_demo(), SUPPORTS)
    start = _scene(a=(-0.3, -0.25, 0.79), b=(0.1, -0.25, 0.79))
    queries = [A.query(start, _eef(), start, _eef(), SUPPORTS)] * 2
    batch = A.pack([demo], queries, lang=False)
    out = A.infer(model, batch, None, latent=True)
    assert out["latent"].shape == (2, N_MAX + 1, 16)
    assert out["latent_ok"][0].tolist() == [True] * 6 + [False] * (N_MAX - 6) + [True]
    rows, mask = A.keyframe_rows(out, 0, queries[0])
    assert np.isfinite(rows).all() and mask.all()
    # The support reaches the prediction: without it the cells differ.
    bare = A.infer(model, A.pack([], queries, lang=False), None, latent=True)
    assert not np.allclose(out["latent"], bare["latent"])
