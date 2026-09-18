"""GR00T N1.7 with ControlVLA object conditioning.

Replaces X-VLA in the backbone set. X-VLA's published weights
(`2toINF/X-VLA-Pt`) are a transformers `AutoModel` repo carrying `model_type`
rather than LeRobot's `type`, so LeRobot's own `xvla` implementation and the
released checkpoint do not meet without a conversion.

GR00T does not have that problem: its checkpoint is named by a *config field*
(`base_model_path`, defaulting to `nvidia/GR00T-N1.7-3B`) and the policy loads
the transformers-format weights itself, so it is built by `--policy.type` with
no `--policy.path`.

It also fits the 4090 budget at 3.14 B (~13.8 GB to finetune) and carries real
cross-embodiment machinery -- category-specific projections indexed by
embodiment id, and `max_state_dim`/`max_action_dim` of 132, so our 16-d action
needs no padding trick.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.groot.configuration_groot import GrootConfig

from oct_vla.policies.object_conditioning import ObjectTokenConfigMixin


@PreTrainedConfig.register_subclass("control_groot")
@dataclass
class ControlGrootConfig(ObjectTokenConfigMixin, GrootConfig):
    """GR00T N1.7 with a zero-initialized residual from precomputed scene tokens."""

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

    #: GR00T defines no default PEFT targets, and LoRA with no targets attaches
    #: to nothing while still training the object path -- which looks like a
    #: working run. The diffusion head's attention and its action projections
    #: are the analogue of what pi0.5 targets.
    lora_target_modules: str = field(
        default=(
            r"(action_head\.model\.transformer_blocks\.\d+\.attn1\.to_(q|v)"
            r"|action_head\.(action_encoder|state_encoder|action_decoder)\..*)"
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_object_tokens()
