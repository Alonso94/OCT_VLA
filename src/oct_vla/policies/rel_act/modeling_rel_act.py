"""ACT over chunk-relative end-effector targets (see configuration_rel_act)."""

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import Tensor

from .configuration_rel_act import RelACTConfig


class RelACTPolicy(ACTPolicy):
    config_class = RelACTConfig
    name = "rel_act"

    def __init__(self, config: RelACTConfig, **kwargs):
        super().__init__(config, **kwargs)
        state_width = config.robot_state_feature.shape[0]
        action_width = config.action_feature.shape[0]
        steps = config.chunk_size
        for name, width in (("state_mean", state_width), ("state_std", state_width),
                            ("target_mean", steps * action_width),
                            ("target_std", steps * action_width)):
            values = getattr(config, name)
            if len(values) != width:
                raise ValueError(f"rel_act needs {name} of width {width} "
                                 f"(scripts/relative_action_stats.py); got {len(values)}")
        if state_width != action_width:
            raise ValueError("rel_act offsets actions by the state, so their layouts must match")
        for name in ("state_mean", "state_std"):
            self.register_buffer(f"_{name}", torch.tensor(getattr(config, name), dtype=torch.float32))
        for name in ("target_mean", "target_std"):
            values = torch.tensor(getattr(config, name), dtype=torch.float32)
            self.register_buffer(f"_{name}", values.view(steps, action_width))
        self.register_buffer("_relative", torch.zeros(action_width, dtype=torch.bool))
        self._relative[list(config.relative_dims)] = True

    # ------------------------------------------------------------ transforms

    def _normalised_state(self, state: Tensor) -> Tensor:
        return (state - self._state_mean) / self._state_std

    def _target(self, action: Tensor, state: Tensor) -> Tensor:
        """Raw absolute chunk [B, C, D] -> normalised chunk-relative target."""
        offset = torch.where(self._relative, state, torch.zeros_like(state))[:, None, :]
        return (action - offset - self._target_mean) / self._target_std

    def _absolute(self, target: Tensor, state: Tensor) -> Tensor:
        """Inverse of `_target`: normalised prediction -> raw absolute chunk."""
        offset = torch.where(self._relative, state, torch.zeros_like(state))[:, None, :]
        return target * self._target_std + self._target_mean + offset

    # ---------------------------------------------------------------- policy

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = dict(batch)
        state = batch[OBS_STATE]
        batch[ACTION] = self._target(batch[ACTION], state)
        batch[OBS_STATE] = self._normalised_state(state)
        return super().forward(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        batch = dict(batch)
        state = batch[OBS_STATE]
        batch[OBS_STATE] = self._normalised_state(state)
        return self._absolute(super().predict_action_chunk(batch), state)
