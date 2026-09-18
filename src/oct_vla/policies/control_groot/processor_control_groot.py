"""Processors for the object-conditioned GR00T.

Pure delegation, as in the other plugins. GR00T is the one policy with real
camera-key expectations: `processor_groot.py` matches the checkpoint's
video-modality keys against the dataset's cameras and, on no match, falls back
to *alphabetical* order with only a warning. So the rename map matters more here
than anywhere else -- a mismatch feeds the wrong view and shows up only as a
success rate.
"""

from __future__ import annotations

from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

from .configuration_control_groot import ControlGrootConfig


def make_control_groot_pre_post_processors(
    config: ControlGrootConfig, dataset_stats: dict | None = None
):
    return make_groot_pre_post_processors(config, dataset_stats)
