"""Semantic colour rules: assignment, eligibility, distractor preservation, plumbing, and
the evaluation server's success test (BimanualPFN stage 0.1)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.serve import codec
from oct_vla.serve.server import _Episode
from oct_vla.tasks.shelf_restock import colours
from oct_vla.tasks.shelf_restock.manager import (ShelfRestockManager, distractors_preserved,
                                                 eligible)
from oct_vla.tasks.shelf_restock.spec import spec_for_rule

_h = importlib.util.spec_from_file_location(
    "server_harness", Path(__file__).resolve().parents[2] / "serve" / "test_server_unreachable.py")
H = importlib.util.module_from_spec(_h)
_h.loader.exec_module(H)

LOWER = (0.0, -0.15, 0.76)
UPPER = (0.0, 0.10, 1.08)


def obj(track, x, shelf, colour, dz=0.0):
    pos = (x, shelf[1], shelf[2] + dz)
    return ObjectState(track, Pose(pos, (0, 0, 0, 1)), (0.04,) * 3, 1.0, 1.0, colour=colour)


def red_spec():
    return spec_for_rule("restock_red", H.spec())


# --- assignment -------------------------------------------------------------------------

def test_colour_assignment_is_deterministic_balanced_and_swappable():
    a = colours.assign(7, 4, ("red", "blue"))
    assert a == colours.assign(7, 4, ("red", "blue"))
    assert sorted(a) == ["blue", "blue", "red", "red"]
    assert sorted(colours.assign(3, 3, ("red", "blue"))) in (["blue", "blue", "red"],
                                                            ["blue", "red", "red"])
    swapped = colours.assign(7, 4, ("red", "blue"), swap=True)
    assert swapped == ["blue" if c == "red" else "red" for c in a]
    assert any(colours.assign(s, 4, ("red", "blue")) != a for s in range(8, 20))


def test_colour_counts_build_a_support_with_three_eligible_and_one_distractor():
    a = colours.assign(4, 4, ("red", "blue"), counts=(3, 1))
    assert sorted(a) == ["blue", "red", "red", "red"]
    with pytest.raises(ValueError, match="do not split"):
        colours.assign(4, 4, ("red", "blue"), counts=(2, 1))


def test_colour_assignment_never_touches_the_layout_stream():
    import random

    layout = random.Random(5)
    before = layout.getstate()
    colours.assign(5, 4, ("red", "blue"))
    assert layout.getstate() == before


@pytest.mark.parametrize("bad", [("purple",), ()])
def test_colour_assignment_refuses_bad_palettes(bad):
    with pytest.raises(ValueError):
        colours.assign(0, 4, bad)
    with pytest.raises(ValueError, match="cannot show"):
        colours.assign(0, 1, ("red", "blue"))


def test_colour_env_parsing(monkeypatch):
    monkeypatch.setenv("OCTVLA_COLOURS", " red, blue ")
    monkeypatch.setenv("OCTVLA_COLOUR_SWAP", "1")
    assert colours.from_env() == (("red", "blue"), True, None)
    monkeypatch.setenv("OCTVLA_COLOUR_COUNTS", "3,1")
    assert colours.from_env()[2] == (3, 1)
    monkeypatch.delenv("OCTVLA_COLOURS")
    assert colours.from_env()[0] == ()


# --- the object state carries its colour ------------------------------------------------

def test_object_state_validates_and_round_trips_colour():
    with pytest.raises(ValueError, match="colour"):
        obj("a", 0.0, LOWER, "purple")
    scene = ObjectScene(0.0, (obj("a", 0.0, LOWER, "red"), obj("b", 0.1, LOWER, None)))
    back = codec.scene_from_json(codec.scene_to_json(scene))
    assert [o.colour for o in back.objects] == ["red", None]
    legacy = codec.scene_to_json(scene)
    for o in legacy["objects"]:
        o.pop("colour")
    assert all(o.colour is None for o in codec.scene_from_json(legacy).objects)


def test_store_round_trips_colour(tmp_path):
    from oct_vla.data import store

    o = obj("a", 0.0, LOWER, "blue")
    assert store._object_from_json(store._object_to_json(o)).colour == "blue"
    raw = store._object_to_json(o)
    raw.pop("colour")
    assert store._object_from_json(raw).colour is None


# --- the manager ------------------------------------------------------------------------

def test_manager_selects_only_eligible_objects_and_finishes_with_distractors_left():
    spec = red_spec()
    assert spec.eligible_colour == "red" and spec_for_rule("leftmost", H.spec()).eligible_colour is None
    m = ShelfRestockManager(spec)
    scene = ObjectScene(0.0, (obj("b1", -0.15, LOWER, "blue"), obj("r1", -0.05, LOWER, "red"),
                              obj("r2", 0.05, LOWER, "red")))
    assert eligible(scene, spec) == ("r1", "r2")
    assert m.next_context(scene).target_track_id == "r1"      # leftmost *red*, not b1
    done = ObjectScene(1.0, (obj("b1", -0.15, LOWER, "blue"), obj("r1", -0.05, UPPER, "red"),
                             obj("r2", 0.05, UPPER, "red")))
    assert m.next_context(done) is None and m.is_done(done) and m.is_restocked(done)
    assert distractors_preserved(done, scene, spec)


def test_moving_a_distractor_fails_preservation_and_legacy_rules_are_unchanged():
    spec = red_spec()
    start = ObjectScene(0.0, (obj("b1", -0.15, LOWER, "blue"), obj("r1", 0.05, LOWER, "red")))
    nudged = ObjectScene(1.0, (obj("b1", -0.12, LOWER, "blue"), obj("r1", 0.05, UPPER, "red")))
    assert not distractors_preserved(nudged, start, spec)
    legacy = ShelfRestockManager(spec_for_rule("leftmost", H.spec()))
    half = ObjectScene(1.0, (obj("b1", -0.15, LOWER, "blue"), obj("r1", 0.05, UPPER, "red")))
    assert not legacy.is_restocked(half) and legacy.next_context(half).target_track_id == "b1"


# --- the evaluation server --------------------------------------------------------------

class ScriptedEstimator:
    def __init__(self, scenes):
        self.scenes = list(scenes)

    def estimate(self, observation):
        return self.scenes.pop(0) if len(self.scenes) > 1 else self.scenes[0]


def _server(scenes, post_success_steps=0):
    port = H.FakePort(RuntimeError("unused"))
    server = H.ShelfRestockEvalServer(lambda task: port, spec=red_spec())
    server._port = port
    est = ScriptedEstimator(scenes)
    server._episode = _Episode(spec=red_spec(), manager=ShelfRestockManager(red_spec()),
                               estimator=est, max_steps=50,
                               post_success_steps=post_success_steps)
    server._episode.initial_scene = scenes[0]
    return server


START = ObjectScene(0.0, (obj("b1", -0.15, LOWER, "blue"), obj("r1", 0.05, LOWER, "red")))
DONE = ObjectScene(1.0, (obj("b1", -0.15, LOWER, "blue"), obj("r1", 0.05, UPPER, "red")))
DISTURBED = ObjectScene(1.0, (obj("b1", -0.05, LOWER, "blue"), obj("r1", 0.05, UPPER, "red")))


def test_server_success_needs_the_distractor_left_in_place():
    s = _server([START, DONE])
    header, _ = s._report(s._episode)                 # reset frame: nothing placed yet
    assert not header["success"] and not header["done"]
    header, _ = s._report(s._episode)
    assert header["success"] and header["done"] and header["distractors_preserved"]
    s = _server([START, DISTURBED])
    s._report(s._episode)
    header, _ = s._report(s._episode)
    assert not header["success"] and not header["done"]
    assert header["distractors_preserved"] is False


def test_post_success_dwell_keeps_the_episode_running_and_rescored():
    s = _server([START, DONE], post_success_steps=3)
    s._report(s._episode)
    for step in range(1, 4):
        s._episode.steps = step
        header, _ = s._report(s._episode)
        assert not header["done"] and header["success_at"] == 1
    s._episode.steps = 4
    header, _ = s._report(s._episode)
    assert header["done"] and header["success"]
    # Knocking the distractor during the dwell turns the success into a failure.
    s = _server([START, DONE, DISTURBED], post_success_steps=1)
    s._report(s._episode)
    s._episode.steps = 1
    s._report(s._episode)
    s._episode.steps = 2
    header, _ = s._report(s._episode)
    assert header["done"] and not header["success"] and header["success_at"] == 1
