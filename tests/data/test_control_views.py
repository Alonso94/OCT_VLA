"""The view table is the contract between the unified export and the projection.
If they disagree about which column carries what, a run trains on the wrong
encoding while its metadata says otherwise -- and nothing downstream notices."""

from __future__ import annotations

import pytest

from oct_vla.data.control_views import ALT_PREFIX, VIEWS, alt_columns, view_for


def test_the_canonical_view_needs_no_remapping():
    """Absolute joint is stored under the names LeRobot expects, so a stock
    reader that knows nothing about views still gets a usable dataset."""
    view = view_for("joint", "position")
    assert view.is_canonical
    assert (view.state_column, view.action_column) == ("observation.state", "action")


def test_every_other_view_reads_at_least_one_alt_column():
    for key, view in VIEWS.items():
        if key == ("joint", "position"):
            continue
        assert view.state_column.startswith(ALT_PREFIX) or \
            view.action_column.startswith(ALT_PREFIX), key


def test_no_alternative_is_named_like_a_policy_feature():
    """The trap this prefix exists to avoid: LeRobot types any column starting
    with `observation` or `action` as a policy feature, so `action.joint_delta`
    would become a second ACTION feature beside the real one."""
    for column in alt_columns():
        assert not column.startswith("observation")
        assert not column.startswith("action")
        assert column.startswith(ALT_PREFIX)


def test_joint_delta_reuses_the_canonical_state():
    """An increment is defined against the configuration, which is already
    observation.state -- so only the action column moves."""
    view = view_for("joint_delta", "position")
    assert view.state_column == "observation.state"
    assert view.action_column == f"{ALT_PREFIX}joint_action_delta"


def test_end_effector_views_move_both_columns():
    """Task space changes the observation too: a joint configuration is not an
    end-effector pose."""
    for space in ("cartesian", "cartesian_absolute"):
        view = view_for(space, "position")
        assert view.state_column == f"{ALT_PREFIX}eef_state"
        assert view.action_column.startswith(ALT_PREFIX)
    assert view_for("cartesian", "position").action_column != \
        view_for("cartesian_absolute", "position").action_column


def test_slugs_are_unique_so_views_cannot_overwrite_each_other():
    slugs = [v.slug for v in VIEWS.values()]
    assert len(slugs) == len(set(slugs))


def test_an_unknown_combination_is_refused_with_the_alternatives():
    """Defaulting to the canonical pair would train on absolute joints while the
    run metadata claimed something else."""
    with pytest.raises(ValueError, match="No view for"):
        view_for("cartesian", "position_velocity")
    with pytest.raises(ValueError, match="No view for"):
        view_for("not_a_space")
