"""Processors for the object-conditioned VLA-JEPA.

Pure delegation, as in the other plugins: the object tokens ride through the
pipeline untouched, and every other step must stay identical to the
unconditioned arm or the comparison is confounded. The module exists mainly so
it cannot be forgotten -- a plugin without one trains fine and then fails at
`make_pre_post_processors`, which is what evaluation calls.
"""

from __future__ import annotations

from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors

from .configuration_control_vla_jepa import ControlVLAJEPAConfig


def make_control_vla_jepa_pre_post_processors(
    config: ControlVLAJEPAConfig, dataset_stats: dict | None = None
):
    return make_vla_jepa_pre_post_processors(config, dataset_stats)
