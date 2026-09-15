"""Mapping this dataset's feature names onto a pretrained policy's expected ones.

pi0.5 was pretrained through openpi, whose camera keys are `base_0_rgb`,
`left_wrist_0_rgb` and `right_wrist_0_rgb`. This project's canonical schema names
the same three views `head`, `left_wrist` and `right_wrist` (see
`data/lerobot_export.py`). LeRobot bridges the two with a rename map.

The map lives here, in one place, because training and evaluation must use the
*same* one. Training fails loudly on a mismatch -- `make_policy` refuses to guess
which view is the base view -- but inference does not: every camera has identical
shape and dtype, so a wrong mapping silently feeds the policy a wrist view where
it expects the scene view, and shows up only as an unexplained drop in success
rate. Two copies of this dict is precisely how that happens.
"""

from __future__ import annotations

#: Dataset feature name -> name pi0.5 was pretrained with.
PI05_RENAME_MAP: dict[str, str] = {
    "observation.images.head": "observation.images.base_0_rgb",
    "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
}
