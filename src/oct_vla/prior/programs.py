"""Task programs of the synthetic interaction prior: selector × destination × arrangement × arm.

A program says which object moves next, where it goes, how the placed objects are arranged,
and which arm(s) do it. It never names a complete benchmark task. The complete shelf-restock
program family (`SHELF_FAMILY`) is excluded from pretraining by construction, so a
shelf result measures transfer, not memorisation. Every generated episode records its
program id as provenance.

The selector orders objects by the *initial* layout. A query therefore carries the
episode's initial frame alongside the current one (generator.py), and any order is
recoverable from the observation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

SELECTORS = (
    "left_to_right",
    "right_to_left",
    "front_to_back",
    "back_to_front",
    "middle_out",  # median x first, then outward alternating left, right (odd counts only)
    "outside_in",  # leftmost, rightmost, leftmost of the rest, ...
    "near_dest_first",  # by distance to the destination support's centre
    "far_dest_first",
    "small_to_large",  # by volume; the scene must have distinct volumes
    "large_to_small",
)
DESTINATIONS = ("higher", "lower", "left_side", "right_side")
ARRANGEMENTS = ("row_left", "row_right", "stack")
ARMS = ("left", "right", "nearest", "handover")

_SELECTOR_TEXT = {
    "left_to_right": "from left to right",
    "right_to_left": "from right to left",
    "front_to_back": "from front to back",
    "back_to_front": "from back to front",
    "middle_out": "starting with the middle one and working outward",
    "outside_in": "from the outside in",
    "near_dest_first": "nearest to the destination first",
    "far_dest_first": "farthest from the destination first",
    "small_to_large": "smallest first",
    "large_to_small": "largest first",
}
_DEST_TEXT = {
    "higher": "onto the higher surface",
    "lower": "onto the lower surface",
    "left_side": "onto the surface on the left",
    "right_side": "onto the surface on the right",
}
_ARR_TEXT = {
    "row_left": "lining them up from the left end",
    "row_right": "lining them up from the right end",
    "stack": "stacking each on the previous one",
}
_ARM_TEXT = {
    "left": "with the left arm",
    "right": "with the right arm",
    "nearest": "with whichever arm is nearer",
    "handover": "handing each object from one arm to the other",
}
AMBIGUOUS_INSTRUCTIONS = (
    "Do what the demonstrations show.",
    "Rearrange the objects the way the examples do.",
    "Repeat the demonstrated task on this scene.",
)


@dataclass(frozen=True)
class Program:
    selector: str
    destination: str
    arrangement: str
    arm: str

    def __post_init__(self) -> None:
        for value, allowed in (
            (self.selector, SELECTORS),
            (self.destination, DESTINATIONS),
            (self.arrangement, ARRANGEMENTS),
            (self.arm, ARMS),
        ):
            if value not in allowed:
                raise ValueError(f"{value!r} is not one of {allowed}")

    @property
    def id(self) -> int:
        return ALL_PROGRAMS.index(self)

    def instructions(self) -> tuple[str, str]:
        """Two explicit paraphrases naming every component of the program."""
        sel, dest = _SELECTOR_TEXT[self.selector], _DEST_TEXT[self.destination]
        arr, arm = _ARR_TEXT[self.arrangement], _ARM_TEXT[self.arm]
        return (
            f"Move the objects {sel} {dest}, {arr}, {arm}.",
            f"{arm[0].upper() + arm[1:]}, take the objects {sel} and put them {dest[5:]}, {arr}.",
        )

    def differs_in(self, other: Program) -> tuple[str, ...]:
        return tuple(
            f
            for f in ("selector", "destination", "arrangement", "arm")
            if getattr(self, f) != getattr(other, f)
        )


ALL_PROGRAMS: tuple[Program, ...] = tuple(
    Program(*combo) for combo in itertools.product(SELECTORS, DESTINATIONS, ARRANGEMENTS, ARMS)
)

#: The shelf-restock family, never seen in pretraining: transfer in left-to-right,
#: right-to-left or middle-out order onto the higher surface, in a row, by the left
#: or the nearer arm. The OCT-VLA evaluation rules (leftmost, rightmost, middle-first)
#: are all inside it.
SHELF_FAMILY: frozenset[Program] = frozenset(
    p
    for p in ALL_PROGRAMS
    if p.selector in ("left_to_right", "right_to_left", "middle_out")
    and p.destination == "higher"
    and p.arrangement in ("row_left", "row_right")
    and p.arm in ("left", "nearest")
)

#: Held-out *compositions*: each component is trained with other partners, never
#: in these conjunctions. A secondary test (GEMIL's pair splits failed it).
COMPOSITION_HOLDOUT: frozenset[Program] = frozenset(
    p
    for p in ALL_PROGRAMS
    if (p.selector == "outside_in" and p.arrangement == "stack")
    or (p.selector == "far_dest_first" and p.arm == "handover")
)

TRAIN_PROGRAMS: tuple[Program, ...] = tuple(
    p for p in ALL_PROGRAMS if p not in SHELF_FAMILY and p not in COMPOSITION_HOLDOUT
)

#: The finite set of strings the language encoder ever sees, in a fixed order.
ALL_INSTRUCTIONS: tuple[str, ...] = tuple(
    dict.fromkeys([*AMBIGUOUS_INSTRUCTIONS, *(s for p in ALL_PROGRAMS for s in p.instructions())])
)


def program_split(name: str) -> tuple[Program, ...]:
    if name == "train":
        return TRAIN_PROGRAMS
    if name == "composition":
        return tuple(sorted(COMPOSITION_HOLDOUT, key=ALL_PROGRAMS.index))
    if name == "shelf_family":
        return tuple(sorted(SHELF_FAMILY, key=ALL_PROGRAMS.index))
    raise ValueError(f"unknown program split {name!r}")
