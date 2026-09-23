"""ACT conditioned on the entity set (see oct_vla.policies.conditioning)."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

from oct_vla.policies.object_conditioning import ObjectConditioningConfig


@PreTrainedConfig.register_subclass("control_act")
@dataclass
class ControlACTConfig(ObjectConditioningConfig, ACTConfig):
    """Stock ACT plus the arm named by ``object_conditioning``."""
