"""ACT with layerwise conditioning on an unordered ground-truth entity set."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

from oct_vla.policies.object_conditioning import ObjectTokenConfigMixin


@PreTrainedConfig.register_subclass("control_act")
@dataclass
class ControlACTConfig(ObjectTokenConfigMixin, ACTConfig):
    object_representation: str = "entity_v2"
    object_entity_normalizer: dict | None = None
    object_token_key: str = "observation.entity_tokens"
    object_token_mask_key: str = "observation.entity_mask"
    object_token_rank_key: str = "observation.object_token_rank"
    object_token_dim: int = 17
    object_token_mode: str = "full"
    object_token_shuffle: bool = False
    object_injection_mode: str = "layerwise"
    object_entity_positional: bool = False
    object_max_entities: int = 16
    object_queries: int = 4
    object_attention_heads: int = 8
    # Encoder-side additions (LPWM, arXiv:2603.04553), each on top of the
    # layerwise decoder branch so that an arm isolates exactly one of them.
    # AdaLN-Zero modulation of every encoder and decoder block; exact identity at init.
    object_adaln: bool = False
    # Entity tokens appended to the encoder sequence behind a learned logit gate.
    object_incontext: bool = False
    object_incontext_gate_init: float = -4.0

    def __post_init__(self):
        super().__post_init__()
        if self.object_injection_mode != "layerwise":
            raise ValueError("control_act supports layerwise injection only")
        self.validate_object_tokens()
