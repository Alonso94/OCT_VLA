"""ACT predicting each chunk's end-effector positions relative to the chunk's start.

The grasp diagnostic (research_questions.md §4.8) found the policy missing its
own *training* grasps by 7.6 mm. Absolute EE targets are normalised over the
whole workspace -- x has a 16.5 cm standard deviation -- so a centimetre is
0.06 sigma of the target. Here the positions in `relative_dims` become offsets
from the measured end-effector position at the chunk's first step (UMI's
relative trajectory), normalised by their own statistics; orientation and
gripper stay absolute. At rollout the offsets are added back to the position
measured when the chunk is queried and executed as absolute targets, so no
error integrates across chunks -- the failure of the delta regimes.

State and action normalisation move into the policy, because the relative
target needs the *raw* state and the processor would hand over a normalised
one. The statistics are fitted on the training split only
(scripts/relative_action_stats.py) and passed as config fields, so they travel
in the checkpoint.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("rel_act")
@dataclass
class RelACTConfig(ACTConfig):
    #: Action/state indices made chunk-relative: both arms' x, y, z.
    relative_dims: list[int] = field(default_factory=lambda: [0, 1, 2, 8, 9, 10])
    state_mean: list[float] = field(default_factory=list)
    state_std: list[float] = field(default_factory=list)
    #: Of the transformed target (relative in `relative_dims`, absolute
    #: elsewhere), **per chunk step**, flattened [chunk_size x action dim]. Per
    #: step because the spread grows with the horizon: over a 50-step chunk
    #: (3.3 s) the arm travels up to 40 cm in transport, so one statistic per
    #: dimension would scale a near-term centimetre as small as the absolute
    #: target does. The executed steps are the near ones.
    target_mean: list[float] = field(default_factory=list)
    target_std: list[float] = field(default_factory=list)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )
