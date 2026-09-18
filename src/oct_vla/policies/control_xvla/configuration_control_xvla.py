"""X-VLA with ControlVLA object conditioning.

X-VLA is the cheapest VLA that fits the 4090 budget (0.88 B, ~3.9 GB to
finetune), and its soft-prompt design -- separate learnable embeddings per
embodiment -- is aimed at exactly our situation: pretrained on other robots,
finetuned on a Franka dual-arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig

from oct_vla.policies.object_conditioning import ObjectTokenConfigMixin


@PreTrainedConfig.register_subclass("control_xvla")
@dataclass
class ControlXVLAConfig(ObjectTokenConfigMixin, XVLAConfig):
    """X-VLA with a zero-initialized residual from precomputed scene tokens."""

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
    #: PEFT targets, which X-VLA does not define. Without an explicit list LoRA
    #: attaches to nothing and the run trains only the object path.
    lora_target_modules: str = field(
        default=r"model\.transformer\.blocks\..*\.attn\.(q|v)"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_object_tokens()
