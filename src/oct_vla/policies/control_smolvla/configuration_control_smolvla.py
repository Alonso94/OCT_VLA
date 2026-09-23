"""Configuration for the offline-object-conditioned SmolVLA policy.

Deliberately the same knobs, names and defaults as the π0.5 arm: the sweep
compares backbones, so any difference here would be a confound rather than a
result.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from oct_vla.policies.object_conditioning import ObjectTokenConfigMixin


@PreTrainedConfig.register_subclass("control_smolvla")
@dataclass
class ControlSmolVLAConfig(ObjectTokenConfigMixin, SmolVLAConfig):
    """SmolVLA with a zero-initialized residual from precomputed scene tokens."""

    object_representation: str = "legacy"
    object_entity_normalizer: dict | None = None

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
    # Encoder-side arms (LPWM, arXiv:2603.04553), each on top of layerwise so
    # an arm isolates one addition, as for control_act. AdaLN: the pooled
    # scene is added, through a zero-initialised projection, to the vector the
    # action model already modulates its norms by (exact identity at init).
    object_adaln: bool = False
    # In-context: entity tokens prepended to the action-model sequence and
    # dropped from its output. Not identity at init (they enter attention).
    object_incontext: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_object_tokens()
