"""VLA-JEPA conditioned on the entity set (see oct_vla.policies.conditioning).

A V-JEPA2 video encoder and a Qwen3-VL backbone feeding a DiT action head. The
one world model in the set that fits a 4090 (~2.45 B, ~10.8 GB to fine-tune); it
already masks `action_is_pad`, and its `reinit_modules` handles the action-width
mismatch between its pretraining and our 16-d Franka.

KV only: its DiT uses diffusers attention, which `install_diffusers_kv` hooks.
The AdaLN and token arms are not ported to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from oct_vla.policies.adapted_vla_jepa.configuration_adapted_vla_jepa import AdaptedVLAJEPAConfig
from oct_vla.policies.object_conditioning import ObjectConditioningConfig


@PreTrainedConfig.register_subclass("control_vla_jepa")
@dataclass
class ControlVLAJEPAConfig(ObjectConditioningConfig, AdaptedVLAJEPAConfig):
    """VLA-JEPA plus KV conditioning."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.object_conditioning != "kv":
            raise ValueError("control_vla_jepa implements object_conditioning='kv' only")
