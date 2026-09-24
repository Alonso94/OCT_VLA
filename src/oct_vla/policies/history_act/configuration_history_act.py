"""ACT that sees the current observation and one (or more) earlier ones.

With a single frame, ACT cannot tell an arm moving towards the shelf from one
moving away, nor a grasp closing from one opening -- the state is a position,
not a velocity. `history_steps` frames, `history_stride` control steps apart,
give it that motion. LeRobot's ACT refuses `n_obs_steps > 1`, so the history is
requested from the dataset here, through `observation_delta_indices`, and
`n_obs_steps` stays 1.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("history_act")
@dataclass
class HistoryACTConfig(ACTConfig):
    """Stock ACT with `history_steps` observation frames."""

    history_steps: int = 2
    #: Control steps between consecutive frames. At 15 Hz, 3 is 0.2 s: far
    #: enough apart that the frames differ, near enough to be about now.
    history_stride: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.history_steps < 2:
            raise ValueError("history_act needs history_steps >= 2; use plain act for one frame")
        if self.history_stride < 1:
            raise ValueError("history_stride must be at least 1")

    @property
    def observation_delta_indices(self) -> list[int]:
        """Oldest first, ending at the current frame."""
        span = (self.history_steps - 1) * self.history_stride
        return list(range(-span, 1, self.history_stride))
