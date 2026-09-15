"""Use SmolVLA's standard processors for the object-conditioned residual.

LeRobot resolves processors by naming convention -- `make_{policy_type}_…` in
the `processor_*` module beside the config -- so a plugin without this file
loads for training but fails at `make_pre_post_processors`, which is what
evaluation calls. The object tokens need no processing of their own: they are
already canonical, and the policy applies any ablation itself.
"""

from __future__ import annotations

import torch
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from .configuration_control_smolvla import ControlSmolVLAConfig


def make_control_smolvla_pre_post_processors(
    config: ControlSmolVLAConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
):
    return make_smolvla_pre_post_processors(config, dataset_stats)
