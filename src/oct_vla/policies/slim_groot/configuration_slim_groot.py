"""Stock GR00T N1.7 whose checkpoints leave out the frozen backbone.

The RGB arm and its budget control (stage 1, and `rgb_cont`) for GR00T. The
network, loss and training are exactly `groot`'s; only the checkpoint changes.
A stock GR00T save writes the whole 3.15 B-parameter model in fp32 plus its
optimiser state -- ~49 GB on disk per checkpoint here, which is how a stage-1
run once died writing its final step -- though its 6.1 GB VLM backbone is
frozen and was measured bit-identical to `nvidia/GR00T-N1.7-3B`. This type
saves the trained head only and rebuilds the backbone from the base on load,
verified (`oct_vla.policies.stage_loading`).
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.groot.configuration_groot import GrootConfig


@PreTrainedConfig.register_subclass("slim_groot")
@dataclass
class SlimGrootConfig(GrootConfig):
    """Identical to GrootConfig; the type name selects the slim checkpoint."""
