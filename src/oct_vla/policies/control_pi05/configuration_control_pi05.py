"""Configuration for the offline-object-conditioned π0.5 policy."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from oct_vla.policies.object_conditioning import ObjectTokenConfigMixin


@PreTrainedConfig.register_subclass("control_pi05")
@dataclass
class ControlPI05Config(ObjectTokenConfigMixin, PI05Config):
    """π0.5 with a zero-initialized residual from precomputed scene tokens."""

    object_token_key: str = "observation.object_tokens"
    object_token_mask_key: str = "observation.object_token_mask"
    #: Episode-stable slot ordering, used only by the role-stripped mode.
    object_token_rank_key: str = "observation.object_token_rank"
    #: Width of the tokens *as stored in the dataset*, before any ablation.
    object_token_dim: int = 15
    #: "full", or "role_stripped" to drop the role one-hot and the
    #: target-first ordering. See oct_vla.data.token_transforms.
    object_token_mode: str = "full"
    #: Evaluation-only control: permute tokens across objects. Training with
    #: this on would teach the policy to ignore the tokens, which is the
    #: opposite of what the control is meant to detect.
    object_token_shuffle: bool = False
    object_queries: int = 4
    object_attention_heads: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_object_tokens()
