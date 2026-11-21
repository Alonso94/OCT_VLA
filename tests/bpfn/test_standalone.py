"""``oct_vla.bpfn`` is standalone: it imports nothing else from ``oct_vla``.

Where it keeps its own copy of something the rest of the repo also defines (the recording
format, the shelf decks, the let-go rule), the copy is held equal to the original here,
outside the package, so neither can drift silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from oct_vla.bpfn.benchmark_tools.timelines import let_go
from oct_vla.bpfn.data.real_data.robotwin import SHELF_DECKS, read_recording

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "oct_vla" / "bpfn"


def _imports(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_bpfn_imports_nothing_else_from_oct_vla():
    outside = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for name in _imports(path):
            if name.startswith("oct_vla") and not name.startswith("oct_vla.bpfn"):
                outside.append(f"{path.relative_to(PACKAGE)}: {name}")
    assert not outside, outside


def test_shelf_decks_equal_the_task_spec():
    from oct_vla.data.entity_tokens import shelf_support_entities
    from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC

    decks = shelf_support_entities(DEFAULT_SPEC)
    assert len(decks) == len(SHELF_DECKS)
    for deck, (_, centre, size) in zip(decks, SHELF_DECKS):
        np.testing.assert_allclose(deck.pose.position, centre)
        np.testing.assert_allclose(deck.pose.orientation, (0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(deck.size_xyz, size)


def _recorded_episode(tmp_path):
    """A small run written by the repository's own recorder."""
    from oct_vla.core.action import Action, ArmAction
    from oct_vla.core.frames import WORKCELL_FRAME, Pose
    from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
    from oct_vla.core.observation import RGBFrame, RobotObservation
    from oct_vla.core.state import ArmState, EEFState
    from oct_vla.data.episode import Episode, Sample
    from oct_vla.data.store import write_episode

    rng = np.random.default_rng(0)

    def pose(position):
        q = rng.normal(size=4)
        return Pose(tuple(position), tuple(q / np.linalg.norm(q)), WORKCELL_FRAME)

    frame = RGBFrame(2, 2, bytes(12))
    samples = []
    for t in range(6):
        eef = EEFState(ArmState(pose((-0.3, -0.2 + 0.01 * t, 1.0)), 0.8 - 0.1 * t),
                       ArmState(pose((0.3, -0.2, 1.0)), 0.5))
        objects = (ObjectState("obj_0", pose((-0.1, -0.25, 0.77 + 0.01 * t)), (0.05, 0.06, 0.07),
                               1.0, 1.0, category="071_can", colour="red"),
                   ObjectState("obj_1", pose((0.1, -0.25, 0.77)), (0.04, 0.04, 0.09), 1.0, 1.0,
                               category="113_coffee-box", colour="blue"))
        samples.append(Sample(timestamp=0.07 * t,
                              observation=RobotObservation(0.07 * t, eef, frame, frame, frame),
                              scene=ObjectScene(0.07 * t, objects),
                              context=TaskContext("restock", "obj_0", previous_neighbor_track_id=None,
                                                  phase="lift"),
                              action=Action(ArmAction((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.5),
                                            ArmAction((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.5)),
                              phase="lift"))
    episode = Episode(seed=7, instruction="restock", samples=tuple(samples), success=True,
                      metadata={"task": "ShelfRestockFourObjectTask", "rule": "restock_red"})
    return write_episode(episode, tmp_path / "episode_0"), episode


def test_recording_reader_matches_the_repository_store(tmp_path):
    from oct_vla.data.store import read_states

    directory, _ = _recorded_episode(tmp_path)
    states = read_states(directory)
    ep = read_recording(directory)
    T = len(states)
    assert ep.T == T and ep.track_ids == ("obj_0", "obj_1", "lower_deck", "upper_deck")
    np.testing.assert_allclose(ep.time, states.timestamps)
    for t in range(T):
        for a, side in enumerate(("left", "right")):
            arm = getattr(states.eef[t], side)
            np.testing.assert_allclose(ep.wrist[t, a, :3], arm.pose.position)
            assert abs(abs(float(np.dot(ep.wrist[t, a, 3:], arm.pose.orientation))) - 1) < 1e-9
            assert ep.aperture[t, a] == pytest.approx(arm.gripper)
        for j, obj in enumerate(states.scenes[t].objects):
            np.testing.assert_allclose(ep.objects[t, j, :3], obj.pose.position)
    assert list(ep.colour[:2]) == [1, 2]                  # red, blue
    assert list(ep.size[0]) == pytest.approx([0.05, 0.06, 0.07])
    assert ep.annotations["rule"] == "restock_red"


def test_let_go_matches_the_effects_timeline():
    from oct_vla.data.effects import object_timeline

    rng = np.random.default_rng(1)
    T = 120
    found = 0
    for trial in range(20):
        p = np.zeros((T, 3))
        p[:, 2] = 0.77
        lift, place = int(rng.integers(10, 40)), int(rng.integers(60, 90))
        p[lift:place, 2] = np.linspace(0.78, 0.92, place - lift)
        p[place:, 2] = 0.92
        p[:, 0] = np.where(np.arange(T) >= lift, 0.1 + 0.001 * (np.arange(T) - lift), 0.1)
        eef = np.zeros((T, 2, 3))
        eef[:, 0] = p + 0.01
        leave = place + int(rng.integers(3, 15))
        eef[leave:, 0, 1] -= np.linspace(0, 0.4, T - leave)
        eef[:, 1] = (0.5, 0.0, 1.0)
        aperture = np.full((T, 2), 0.8)
        aperture[lift - 3:leave, 0] = 0.2
        upper = p[:, 2] > 0.86
        expected = object_timeline(p, eef, aperture, upper).get("let_go")
        assert let_go(p, eef, aperture, upper) == expected, trial
        found += expected is not None
    assert found >= 15          # the trials do reach a let-go (not None == None throughout)
