"""Processors for the object-conditioned X-VLA.

Pure delegation to X-VLA's own, as the π0.5 and SmolVLA plugins do: the object
tokens ride through the pipeline untouched, and every image, state and language
step must stay identical to the unconditioned arm or the comparison is
confounded.

The module exists mainly so it cannot be forgotten. A plugin without one trains
fine and then fails at `make_pre_post_processors`, which is what evaluation
calls -- a failure that surfaces only after the GPU hours are spent.
"""

from __future__ import annotations

from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

from .configuration_control_xvla import ControlXVLAConfig


def make_control_xvla_pre_post_processors(
    config: ControlXVLAConfig, dataset_stats: dict | None = None
):
    return make_xvla_pre_post_processors(config, dataset_stats)
