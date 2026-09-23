"""GR00T N1.7 conditioned on the entity set (see oct_vla.policies.conditioning).

GR00T's checkpoint is named by a *config field* (`base_model_path`, default
`nvidia/GR00T-N1.7-3B`) and the policy loads the transformers-format weights
itself, so it is built with `--policy.type` and no `--policy.path`. It fits a
4090 (3.14 B, ~13.8 GB to fine-tune), and its `max_action_dim` of 132 takes our
16-d action with no padding trick. It trains without LoRA: the VLM is frozen and
the projector and diffusion head are tuned in full.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.groot.configuration_groot import GrootConfig

from oct_vla.policies.object_conditioning import ObjectConditioningConfig


@PreTrainedConfig.register_subclass("control_groot")
@dataclass
class ControlGrootConfig(ObjectConditioningConfig, GrootConfig):
    """Stock GR00T plus the arm named by ``object_conditioning``."""
