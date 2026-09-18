"""One dataset, several control spaces, selected by a view.

Until now each action encoding needed its own dataset directory, because LeRobot
binds training to the single column named `action` and `observation.state`. That
gave eight near-identical directories whose only real difference was a few
megabytes of parquet.

The way out is that `dataset_to_policy_features` types a column by its *prefix*:
`observation.*` becomes STATE, `action*` becomes ACTION, and **everything else is
skipped entirely**. So a column named `alt.eef_state` is stored, published and
copied like any other, and is simply invisible to policy construction. A unified
dataset carries the canonical pair under the names LeRobot expects and every
alternative under `alt.`, and a view promotes the pair a given run wants.

That prefix rule is also a trap, and this module exists partly to keep it
visible: an earlier export wrote `action.joint_position`, which starts with
"action", and LeRobot duly reported two ACTION features where every policy
expects one.

The canonical pair is absolute joint targets, for three reasons. It is the only
encoding that has produced a non-zero closed-loop result here; it is what
RoboTwin's own `joint_action` format uses; and joint increments are recoverable
from it exactly, which is not true in the other direction without the state.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Written into meta/info.json so a reader can tell a unified dataset from a
#: single-encoding one without inspecting columns.
UNIFIED_CONTROL_SPACE = "unified"

#: Prefix for alternative encodings. Chosen because LeRobot's feature typing
#: ignores anything that is not `observation*` or `action*`; renaming these to
#: `action.something` would silently create a second ACTION feature.
ALT_PREFIX = "alt."


@dataclass(frozen=True)
class ControlView:
    """Which stored columns a given control space reads as state and action."""

    control_space: str
    state_encoding: str
    state_column: str
    action_column: str
    #: Human-readable, used for the derived directory name and in reports.
    slug: str

    @property
    def is_canonical(self) -> bool:
        """True when the view needs no remapping at all."""
        return self.state_column == "observation.state" and self.action_column == "action"


def _view(control_space, state_encoding, state_column, action_column, slug):
    return ControlView(control_space, state_encoding, state_column, action_column, slug)


#: Every view a unified dataset can serve, keyed by (control_space, state_encoding).
VIEWS: dict[tuple[str, str], ControlView] = {
    ("joint", "position"): _view(
        "joint", "position", "observation.state", "action", "abs"),
    ("joint_delta", "position"): _view(
        "joint_delta", "position", "observation.state", f"{ALT_PREFIX}joint_action_delta",
        "delta"),
    ("joint_delta", "position_velocity"): _view(
        "joint_delta", "position_velocity", f"{ALT_PREFIX}joint_state_velocity",
        f"{ALT_PREFIX}joint_action_delta", "delta_velo"),
    ("joint", "position_velocity"): _view(
        "joint", "position_velocity", f"{ALT_PREFIX}joint_state_velocity", "action",
        "abs_velo"),
    ("cartesian", "position"): _view(
        "cartesian", "position", f"{ALT_PREFIX}eef_state", f"{ALT_PREFIX}eef_action_delta",
        "eedelta"),
    ("cartesian_absolute", "position"): _view(
        "cartesian_absolute", "position", f"{ALT_PREFIX}eef_state",
        f"{ALT_PREFIX}eef_action_abs", "eeabs"),
}


def view_for(control_space: str, state_encoding: str = "position") -> ControlView:
    """The view for a control space, or a listing of what exists.

    Raises rather than defaulting: silently falling back to the canonical pair
    would train a policy on absolute joint targets while its run metadata said
    something else, and nothing downstream would notice.
    """
    try:
        return VIEWS[(control_space, state_encoding)]
    except KeyError:
        raise ValueError(
            f"No view for control_space={control_space!r} "
            f"state_encoding={state_encoding!r}; have "
            f"{sorted(VIEWS)}"
        ) from None


def alt_columns() -> tuple[str, ...]:
    """Every `alt.` column a unified export must write, deduplicated."""
    columns = set()
    for view in VIEWS.values():
        for column in (view.state_column, view.action_column):
            if column.startswith(ALT_PREFIX):
                columns.add(column)
    return tuple(sorted(columns))
