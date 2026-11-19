"""Object colours for semantic shelf tasks: which object gets which colour (pure Python).

The draw uses its own named stream, ``random.Random(f"oct_vla/colour/v1/{seed}")``, so
switching colour on never changes a seed's layout (that stream is the task's
``random.Random(seed)``). Each scene holds at least one object of every palette colour
(as balanced as the count allows), and ``swap`` exchanges the first two palette colours:
the same layout with the colours trading places, the matched-state control.

Configured by environment for the RoboTwin task process (``robotwin_env``), like the
asset and variant overrides: ``OCTVLA_COLOURS=red,blue`` and ``OCTVLA_COLOUR_SWAP=1``.
Unset means no tint and the old behaviour.
"""

from __future__ import annotations

import os
import random

from oct_vla.core.objects import COLOURS

#: Rendered base colours (linear RGB). Saturated and far apart, so a head-camera crop
#: separates them under the task's lights; the canary checks that on real renders.
RGB = {
    "red": (0.80, 0.08, 0.08),
    "blue": (0.08, 0.20, 0.85),
    "green": (0.10, 0.65, 0.15),
    "yellow": (0.90, 0.78, 0.08),
}
STREAM = "oct_vla/colour/v1"


def assign(seed: int, n: int, palette: tuple[str, ...], swap: bool = False,
           counts: tuple[int, ...] | None = None) -> list[str]:
    """Colour names for objects 0..n-1 (spawn order). ``counts``: exactly this many of each
    palette colour (e.g. (3, 1): a colour support with three eligible and one distractor);
    default: as balanced as ``n`` allows."""
    if not palette:
        raise ValueError("empty palette")
    unknown = [c for c in palette if c not in COLOURS]
    if unknown:
        raise ValueError(f"unknown colours {unknown}; choose from {COLOURS}")
    if n < len(palette):
        raise ValueError(f"{n} objects cannot show all {len(palette)} palette colours")
    rng = random.Random(f"{STREAM}/{seed}")
    if counts is not None:
        if len(counts) != len(palette) or sum(counts) != n or min(counts) < 1:
            raise ValueError(f"counts {counts} do not split {n} objects over {palette}")
        base = [c for c, k in zip(palette, counts) for _ in range(k)]
    else:
        base = [palette[i % len(palette)] for i in range(n)]
    rng.shuffle(base)
    if swap:
        if len(palette) < 2:
            raise ValueError("a colour swap needs two palette colours")
        a, b = palette[0], palette[1]
        base = [b if c == a else a if c == b else c for c in base]
    return base


def from_env() -> tuple[tuple[str, ...], bool, tuple[int, ...] | None]:
    """(palette, swap, counts) from OCTVLA_COLOURS / OCTVLA_COLOUR_SWAP /
    OCTVLA_COLOUR_COUNTS; an empty palette = off."""
    raw = os.environ.get("OCTVLA_COLOURS", "").strip()
    palette = tuple(c.strip() for c in raw.split(",") if c.strip())
    swap = os.environ.get("OCTVLA_COLOUR_SWAP", "0").strip() in ("1", "true", "yes")
    raw_counts = os.environ.get("OCTVLA_COLOUR_COUNTS", "").strip()
    counts = tuple(int(k) for k in raw_counts.split(",")) if raw_counts else None
    return palette, swap, counts
