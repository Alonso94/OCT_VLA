"""See configuration_slim_groot: stock GR00T with a slim, verified checkpoint."""

from __future__ import annotations

from lerobot.policies.groot.modeling_groot import GrootPolicy

from oct_vla.policies.stage_loading import SlimCheckpointMixin, VerifiedLoadMixin

from .configuration_slim_groot import SlimGrootConfig

#: GR00T's frozen VLM, rebuilt from `base_model_path` on construction.
GROOT_BACKBONE = ("_groot_model.backbone.",)


class SlimGrootPolicy(SlimCheckpointMixin, VerifiedLoadMixin, GrootPolicy):
    config_class = SlimGrootConfig
    name = "slim_groot"
    rebuilt_prefixes = GROOT_BACKBONE
