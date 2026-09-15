"""Configuration for the offline-object-conditioned π0.5 policy."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from oct_vla.data.token_transforms import TOKEN_MODES, token_dim_for_mode


@PreTrainedConfig.register_subclass("control_pi05")
@dataclass
class ControlPI05Config(PI05Config):
    """π0.5 with a zero-initialized residual from precomputed scene tokens."""

    object_token_key: str = "observation.object_tokens"
    object_token_mask_key: str = "observation.object_token_mask"
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

    @property
    def effective_object_token_dim(self) -> int:
        """Width the ObjectExpert actually projects, after the mode's ablation.

        Derived rather than configured: a hand-set width that disagreed with
        the mode would surface as a shape error deep inside cross-attention,
        long after the run started.
        """
        return token_dim_for_mode(self.object_token_dim, self.object_token_mode)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.object_token_mode not in TOKEN_MODES:
            raise ValueError(
                f"object_token_mode must be one of {TOKEN_MODES}, got {self.object_token_mode!r}"
            )
        if self.object_token_dim <= 0:
            raise ValueError("object_token_dim must be positive")
        if self.object_queries <= 0:
            raise ValueError("object_queries must be positive")
        if self.object_attention_heads <= 0:
            raise ValueError("object_attention_heads must be positive")
