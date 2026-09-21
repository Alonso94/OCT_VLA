"""VLA-JEPA with ControlVLA object conditioning.

The world model of the set: a V-JEPA2 video encoder and a Qwen3-VL backbone
feeding a DiT action head. Chosen over LingBot-VA and FastWAM because it is the
only one that fits a 4090 (~2.45 B, ~10.8 GB to finetune), already masks
`action_is_pad`, and carries `reinit_modules` -- an explicit cross-embodiment
mechanism for the action-dimension mismatch between its pretraining and our
16-d Franka.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from oct_vla.policies.adapted_vla_jepa.configuration_adapted_vla_jepa import AdaptedVLAJEPAConfig

from oct_vla.policies.object_conditioning import ObjectTokenConfigMixin


@PreTrainedConfig.register_subclass("control_vla_jepa")
@dataclass
class ControlVLAJEPAConfig(ObjectTokenConfigMixin, AdaptedVLAJEPAConfig):
    """VLA-JEPA with a zero-initialized residual from precomputed scene tokens."""

    object_representation: str = "legacy"
    object_entity_normalizer: dict | None = None

    object_token_key: str = "observation.object_tokens"
    object_token_mask_key: str = "observation.object_token_mask"
    #: Episode-stable slot ordering, used only by the role-stripped mode.
    object_token_rank_key: str = "observation.object_token_rank"
    #: Width of the tokens *as stored in the dataset*, before any ablation.
    # Legacy records remain the default. The 17-wide entity schema is explicit.
    object_token_dim: int = 15
    #: "full", or "role_stripped" to drop the role one-hot and the
    #: target-first ordering. See oct_vla.data.token_transforms.
    object_token_mode: str = "full"
    #: Evaluation-only control: permute tokens across objects.
    object_token_shuffle: bool = False
    #: "controlvla" for the published dual-attention injection over the
    #: unpooled object set; "pooled" for the weaker broadcast residual every
    #: result recorded before 2026-09-20 was measured with.
    object_injection_mode: str = "controlvla"
    #: Learned code across entity slots. Off keeps the encoder
    #: permutation-invariant; on makes it order-sensitive, which is what the
    #: shuffled-token control needs in order to measure anything.
    object_entity_positional: bool = False
    object_max_entities: int = 16
    #: Used by the "pooled" mode only.
    object_queries: int = 4
    object_attention_heads: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_object_tokens()
