"""VLA-JEPA with ControlVLA object conditioning.

The world model of the set: a V-JEPA2 video encoder and a Qwen3-VL backbone
feeding a DiT action head. Chosen over LingBot-VA and FastWAM because it is the
only one that fits a 4090 (~2.45 B, ~10.8 GB to finetune), already masks
`action_is_pad`, and carries `reinit_modules` -- an explicit cross-embodiment
mechanism for the action-dimension mismatch between its pretraining and our
16-d Franka.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

from oct_vla.policies.object_conditioning import ObjectTokenConfigMixin


@PreTrainedConfig.register_subclass("control_vla_jepa")
@dataclass
class ControlVLAJEPAConfig(ObjectTokenConfigMixin, VLAJEPAConfig):
    """VLA-JEPA with a zero-initialized residual from precomputed scene tokens."""

    object_token_key: str = "observation.object_tokens"
    object_token_mask_key: str = "observation.object_token_mask"
    #: Episode-stable slot ordering, used only by the role-stripped mode.
    object_token_rank_key: str = "observation.object_token_rank"
    #: Width of the tokens *as stored in the dataset*, before any ablation.
    object_token_dim: int = 15
    #: "full", or "role_stripped" to drop the role one-hot and the
    #: target-first ordering. See oct_vla.data.token_transforms.
    object_token_mode: str = "full"
    #: Evaluation-only control: permute tokens across objects.
    object_token_shuffle: bool = False
    object_queries: int = 4
    object_attention_heads: int = 8

    #: The action/state projections must be rebuilt for a 16-d embodiment; the
    #: rest of the network keeps its pretrained weights. Declared here so the
    #: mismatch is handled by the mechanism VLA-JEPA provides rather than by a
    #: strict-load failure.
    reinit_modules: list[str] | None = field(
        default_factory=lambda: [
            "model.action_model.action_encoder",
            "model.action_model.state_encoder",
            "model.action_model.action_decoder",
        ]
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_object_tokens()
