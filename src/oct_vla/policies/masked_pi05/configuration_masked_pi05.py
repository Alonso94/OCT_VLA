"""Stock pi0.5, with padded action targets excluded from the loss.

The RGB arm cannot use `control_pi05`: that would give it an object expert and
a zero-initialised injection it has no tokens to feed, which is an
architectural difference where the experiment intends none. But it needs the
same padding correction as the object arms, or the comparison confounds
"conditioning" with "which arm fit repeated terminal actions".

So this changes exactly one thing -- the loss reduction -- and nothing about
the network. Weights remain interchangeable with stock pi0.5 in both
directions.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("masked_pi05")
@dataclass
class MaskedPI05Config(PI05Config):
    """Identical to PI05Config; the type name is what selects the masked loss."""
