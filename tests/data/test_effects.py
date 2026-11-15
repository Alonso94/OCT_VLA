"""Hindsight object effects (data/effects.py), on synthetic tracks with known events."""

from __future__ import annotations

import numpy as np
import pytest

from oct_vla.data import effects as E
from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM

T = 200
HOLD_AP, OPEN_AP = 0.78, 0.833


def _transfer(lift=20, place=50, release=55, leave=60, push=(100, 120), pushed=True):
    """One object carried by the left arm onto the shelf, released, then (optionally)
    pushed 5 cm by the right arm. Returns p [T,3], eef [T,2,3], aperture [T,2], upper [T]."""
    p = np.zeros((T, 3))
    p[:, :] = (0.2, -0.25, 0.80)
    rest = np.array([-0.10, -0.06, 0.97])
    for t in range(lift, place):  # carried, descending at the end
        a = (t - lift) / (place - lift)
        p[t] = (1 - a) * p[0] + a * rest + (0, 0, 0.10 * np.sin(np.pi * a))
    p[place:] = rest
    if pushed:
        for t in range(push[0], push[1]):
            p[t] = rest + (0.05 * (t - push[0]) / (push[1] - push[0]), 0, 0)
        p[push[1]:] = rest + (0.05, 0, 0)
    eef = np.zeros((T, 2, 3))
    eef[:, 0] = p + (0, 0, 0.13)          # left holds it until it leaves
    for t in range(leave, T):
        eef[t, 0] = rest + (0, 0, 0.13 + 0.01 * (t - leave))   # rises 1 cm/frame
    eef[:, 1] = (0.5, -0.1, 1.2)           # right parked far away
    if pushed:
        for t in range(push[0] - 10, push[1] + 5):
            eef[t, 1] = p[t] + (0.06, 0, 0.13)
    eef[:lift - 5, 0] = p[0] + (0, 0, 0.30)
    for t in range(lift - 5, lift):
        eef[t, 0] = p[0] + (0, 0, 0.13)
    ap = np.full((T, 2), OPEN_AP)
    ap[lift - 4:release, 0] = HOLD_AP
    for t in range(release, min(release + 15, T)):
        ap[t, 0] = HOLD_AP + (OPEN_AP - HOLD_AP) * (t - release + 1) / 15
    upper = (p[:, 2] > 0.9) & (np.abs(p[:, 0] + 0.06) < 0.30)
    return p, eef, ap, upper


def test_release_distance_is_the_oracles():
    from oct_vla.tasks.shelf_restock.keyframes import RELEASE_DISTANCE

    assert E.RELEASE_DISTANCE == RELEASE_DISTANCE


def test_timeline_finds_each_event_in_order():
    p, eef, ap, upper = _transfer()
    line = E.object_timeline(p, eef, ap, upper)
    assert line["mover"] == "left" and line["lift"] is not None
    assert line["grasp"] <= line["lift"] < line["place"] <= line["release"] <= line["let_go"]
    # The aperture starts rising at 55 and passes OPENING above its hold within 3 frames.
    assert 55 <= line["release"] <= 58
    # Let go once the mover is RELEASE_DISTANCE away: it starts 0.13 m off, rises 1 cm/frame.
    assert line["let_go"] == 60 + 8
    assert line["push_start"] >= 100 - 1 and line["push_end"] <= 120
    assert line["pusher"] == "right" and line["settled"] > line["push_end"]


def test_stillness_alone_never_ends_a_segment():
    """The object sits still between release and push for ~45 frames: contract B must
    keep it as the effect through that wait and the push, not switch at the pause."""
    p, eef, ap, upper = _transfer()
    line = E.object_timeline(p, eef, ap, upper)
    (_, end_a, goal_a), = E.segments({"o": line}, "A")
    (_, end_b, goal_b), = E.segments({"o": line}, "B")
    assert end_a == goal_a == line["let_go"]
    assert goal_b == line["push_end"] and end_b == line["settled"] > line["push_end"]
    np.testing.assert_allclose(p[goal_b] - p[goal_a], (0.05, 0, 0), atol=3e-3)


def test_unpushed_object_contracts_agree():
    line = E.object_timeline(*_transfer(pushed=False))
    assert "push_start" not in line
    assert E.segments({"o": line}, "A") == E.segments({"o": line}, "B")


def test_a_neighbours_later_push_is_not_the_objects_own():
    """The first-placed object is nudged by the second's compaction after the second's
    lift: under contract B it must still end at its own let-go."""
    first = E.object_timeline(*_transfer(pushed=True, push=(150, 170)))
    second = dict(first, lift=first["let_go"] + 20, let_go=first["let_go"] + 60,
                  push_start=None, push_end=None)
    lines = {"a": first, "b": second}
    segs = dict((k, (e, g)) for k, e, g in E.segments(lines, "B"))
    assert segs["a"] == (first["let_go"], first["let_go"])


def test_segments_refuse_an_object_never_let_go_and_unknown_contracts():
    line = {"lift": 10, "place": 30, "release": None}
    with pytest.raises(ValueError, match="never let go"):
        E.segments({"o": line}, "A")
    with pytest.raises(ValueError, match="contract"):
        E.segments({}, "C")


def test_rows_switch_at_segment_ends_and_match_the_goal_prior_format():
    from oct_vla.prior.shelf_adapter import _yaw_pose
    from oct_vla.data.entity_tokens import _token

    p, eef, ap, upper = _transfer()
    yaw = 0.4
    quat = np.tile([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)], (T, 1))
    tracks = E.Tracks(("o",), p[:, None], quat[:, None], np.array([[0.05, 0.06, 0.08]]),
                      eef, ap, upper[:, None])
    lines = E.timelines(tracks)
    rows, mask, target = E.effect_rows(tracks, lines, "A")
    let_go = lines["o"]["let_go"]
    assert target[:let_go] == ["o"] * let_go and set(target[let_go:]) == {None}
    assert mask[0].tolist() == [True, False, True] and not mask[let_go:].any()
    # Row 2 is the goal at let-go, in exactly the token a goal-only prior writes.
    expected = _token(_yaw_pose(p[let_go], (np.sin(yaw), np.cos(yaw))), (0.05, 0.06, 0.08),
                      0.0, "movable")
    np.testing.assert_allclose(rows[0, 2, :ENTITY_TOKEN_DIM], expected, atol=1e-6)
    assert rows[0, 2, ENTITY_TOKEN_DIM:].tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert rows[0, 0, ENTITY_TOKEN_DIM:].tolist() == [1, 0, 0, 0, 0, 0, 0]
    # Row 0 follows the object; the goal row stays fixed within the segment.
    np.testing.assert_allclose(rows[30, 0, :3], p[30], atol=1e-6)
    np.testing.assert_array_equal(rows[10, 2], rows[40, 2])
    rows_b, _, target_b = E.effect_rows(tracks, lines, "B")
    assert target_b[let_go] == "o" and target_b[lines["o"]["settled"]] is None
