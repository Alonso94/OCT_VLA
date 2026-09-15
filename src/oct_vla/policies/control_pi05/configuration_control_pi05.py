"""Configuration for the offline-object-conditioned π0.5 policy."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("control_pi05")
@dataclass
class ControlPI05Config(PI05Config):
    """π0.5 with a zero-initialized residual from precomputed scene tokens."""

    object_token_key: str = "observation.object_tokens"
    object_token_mask_key: str = "observation.object_token_mask"
    object_token_dim: int = 15
    object_queries: int = 4
    object_attention_heads: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.object_token_dim <= 0:
            raise ValueError("object_token_dim must be positive")
        if self.object_queries <= 0:
            raise ValueError("object_queries must be positive")
        if self.object_attention_heads <= 0:
            raise ValueError("object_attention_heads must be positive")
