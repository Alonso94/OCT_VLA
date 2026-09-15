"""Use π0.5's standard processors for the object-conditioned residual."""

from __future__ import annotations

import torch
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

from .configuration_control_pi05 import ControlPI05Config


def make_control_pi05_pre_post_processors(
    config: ControlPI05Config, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
):
    return make_pi05_pre_post_processors(config, dataset_stats)
