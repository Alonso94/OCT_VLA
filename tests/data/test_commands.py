"""The applied-command log (data/commands.py) and its place in the episode store."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oct_vla.data.commands import COMMANDS_FILE, from_motions, read_commands
from oct_vla.data.store import read_episode, read_states, write_episode
from oct_vla.tasks.shelf_restock.oracle.motion import ExecutedMotion

_spec = importlib.util.spec_from_file_location(
    "store_fixtures", Path(__file__).resolve().parent / "test_store.py")
_store = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_store)
_episode = _store._episode


def _motion(side, ticks, start, held, held_grip=0.8):
    q = tuple(tuple(start + i + 0.1 * j for j in range(7)) for i in range(ticks))
    return ExecutedMotion(side, q, q, tuple(0.1 * i for i in range(ticks)),
                          tuple(held + 0.1 * j for j in range(7)), held_grip)


def test_log_places_moving_and_held_arms_per_tick():
    log = from_motions([_motion("left", 3, 0.0, 9.0), _motion("right", 2, 5.0, 1.0, 0.3)],
                       [0, 2, 4], dt=0.004)
    assert log.ticks == 5 and log.active.tolist() == [0, 0, 0, 1, 1]
    np.testing.assert_allclose(log.q[:3, 0, 0], [0, 1, 2])       # left moves
    np.testing.assert_allclose(log.q[:3, 1, 0], [9, 9, 9])       # right held
    np.testing.assert_allclose(log.q[3:, 1, 0], [5, 6])          # right moves
    np.testing.assert_allclose(log.q[3:, 0, 0], [1, 1])          # left held
    np.testing.assert_allclose(log.gripper[3:, 0], [0.3, 0.3])
    assert not log.qdot[:3, 1].any()                             # a held arm has no velocity
    q, _ = log.at_frames()
    np.testing.assert_allclose(q[:, 0, 0], [0, 2, 1])            # ticks 0, 2, 4


def test_log_refuses_motions_without_their_held_reference():
    bare = ExecutedMotion("left", ((0.0,) * 7,), ((0.0,) * 7,), (0.0,))
    with pytest.raises(ValueError, match="held arm"):
        from_motions([bare], [0], dt=0.004)


def test_log_refuses_frames_beyond_the_commanded_ticks():
    with pytest.raises(ValueError, match="outside"):
        from_motions([_motion("left", 2, 0.0, 1.0)], [0, 5], dt=0.004)


def test_store_writes_and_reads_the_log_beside_the_episode(tmp_path):
    episode = _episode()
    n = len(episode.samples)
    log = from_motions([_motion("left", 2 * n, 0.0, 3.0)], range(0, 2 * n, 2), dt=0.004)
    write_episode(replace(episode, commands=log), tmp_path / "ep")
    back = read_commands(tmp_path / "ep")
    np.testing.assert_allclose(back.q, log.q, atol=1e-5)
    np.testing.assert_array_equal(back.frame_ticks, log.frame_ticks)
    back_ep = read_episode(tmp_path / "ep")
    assert [s.timestamp for s in back_ep.samples] == [s.timestamp for s in episode.samples]
    assert back_ep.commands is None  # read_episode never loads the log; read_commands does
    assert len(read_states(tmp_path / "ep")) == n
    # A legacy recording has no log, and that is reported as None, not an error.
    write_episode(episode, tmp_path / "legacy")
    assert not (tmp_path / "legacy" / COMMANDS_FILE).exists()
    assert read_commands(tmp_path / "legacy") is None


def test_store_refuses_a_log_that_does_not_match_the_samples(tmp_path):
    episode = _episode()
    log = from_motions([_motion("left", 20, 0.0, 3.0)], [0, 1], dt=0.004)
    with pytest.raises(ValueError, match="frame ticks"):
        write_episode(replace(episode, commands=log), tmp_path / "ep")
