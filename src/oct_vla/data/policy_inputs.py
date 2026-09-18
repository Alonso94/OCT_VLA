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

#: Dataset feature name -> name SmolVLA was pretrained with.
#:
#: `lerobot/smolvla_base` declares `camera1`, `camera2`, `camera3`, and the
#: mapping is needed even though SmolVLA builds its image features from
#: whatever the dataset provides. The reason is subtle and cost a smoke run to
#: find: the RGB arm loads the base checkpoint with `--policy.path`, which
#: brings the checkpoint's own `input_features` and hard-fails on our names,
#: while the object arm passes `--policy.type` and builds features from the
#: dataset instead. Without this map the two arms of one backbone would train
#: on differently *named* cameras -- the RGB arm refusing to start, and, had it
#: not refused, the comparison silently confounded.
#:
#: camera1 takes the head view, mirroring `head -> base_0_rgb` for pi0.5: it is
#: the scene view in both cases, and the wrist views follow left then right.
SMOLVLA_RENAME_MAP: dict[str, str] = {
    "observation.images.head": "observation.images.camera1",
    "observation.images.left_wrist": "observation.images.camera2",
    "observation.images.right_wrist": "observation.images.camera3",
}

#: ACT is trained from scratch here, so it has no pretrained camera names to
#: match and simply adopts whatever the dataset declares. Registered explicitly
#: rather than left to a default: `rename_map_for` refusing an unknown backbone
#: is what stops a new policy silently inheriting pi0.5's openpi names.
ACT_RENAME_MAP: dict[str, str] = {}

#: Dataset feature name -> name FlashVLA's RoboTwin checkpoint was trained with.
#:
#: RoboTwin's own camera convention, not openpi's: `cam_high` for the scene view
#: and `cam_{left,right}_wrist` for the grippers. Distinct from pi0.5's map even
#: though FlashVLA is a pi0.5 finetune, because the finetuning corpus renamed
#: them. Getting this wrong is silent -- all three views share a shape and dtype
#: -- so it is registered here rather than defaulted.
FLASHVLA_RENAME_MAP: dict[str, str] = {
    "observation.images.head": "observation.images.cam_high",
    "observation.images.left_wrist": "observation.images.cam_left_wrist",
    "observation.images.right_wrist": "observation.images.cam_right_wrist",
}

_RENAME_MAPS: dict[str, dict[str, str]] = {
    "act": ACT_RENAME_MAP,
    "pi05": PI05_RENAME_MAP,
    "control_pi05": PI05_RENAME_MAP,
    "masked_pi05": PI05_RENAME_MAP,
    "pi05-flashvla": FLASHVLA_RENAME_MAP,
    "pi0-flashvla": FLASHVLA_RENAME_MAP,
    "smolvla": SMOLVLA_RENAME_MAP,
    "control_smolvla": SMOLVLA_RENAME_MAP,
}


def rename_map_for(policy_type: str) -> dict[str, str]:
    """The rename map a policy type was pretrained against.

    Keyed on the policy type rather than passed by the caller: evaluation reads
    the type back from the checkpoint, so the map can never drift from the one
    training used.
    """
    try:
        return _RENAME_MAPS[policy_type]
    except KeyError:
        raise ValueError(
            f"No rename map registered for policy type {policy_type!r}; "
            f"known types are {sorted(_RENAME_MAPS)}"
        ) from None
