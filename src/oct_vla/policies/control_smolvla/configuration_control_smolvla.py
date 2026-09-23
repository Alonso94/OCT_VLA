"""SmolVLA conditioned on the entity set (see oct_vla.policies.conditioning)."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from oct_vla.policies.object_conditioning import ObjectConditioningConfig


@PreTrainedConfig.register_subclass("control_smolvla")
@dataclass
class ControlSmolVLAConfig(ObjectConditioningConfig, SmolVLAConfig):
    """Stock SmolVLA plus the arm named by ``object_conditioning``."""
