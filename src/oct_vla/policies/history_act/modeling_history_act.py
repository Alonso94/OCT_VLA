"""ACT over a short observation history.

Every frame's camera features become encoder tokens, each frame's tokens
carrying a learned frame embedding on top of ACT's 2-D sine position (without
it the encoder, which is permutation-invariant apart from position, could not
tell the frames apart). The states are concatenated into the one state token,
so ACT's token layout and 1-D positions are unchanged.

Training reads the history from the dataset (`observation_delta_indices`), which
pads an episode's first frames by repeating its first observation. Rollout keeps
the history here, every step, and pads the same way -- the policy is called for
an action every step even when the chunk queue is full, so the history never
skips a frame.
"""

from collections import deque

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn
from torch.nn import functional as F

from .configuration_history_act import HistoryACTConfig


class HistoryACTPolicy(ACTPolicy):
    config_class = HistoryACTConfig
    name = "history_act"

    def __init__(self, config: HistoryACTConfig, **kwargs):
        super().__init__(config, **kwargs)
        steps, model = config.history_steps, self.model
        if config.robot_state_feature:
            width = config.robot_state_feature.shape[0] * steps
            model.encoder_robot_state_input_proj = nn.Linear(width, config.dim_model)
            if config.use_vae:
                model.vae_encoder_robot_state_input_proj = nn.Linear(width, config.dim_model)
        model.history_frame_embed = nn.Parameter(torch.randn(steps, config.dim_model) * 0.02)
        #: Which image the position-embedding module is on within one forward;
        #: images arrive frame-major, cameras inner.
        self._image_index = 0
        model.register_forward_pre_hook(self._reset_image_index)
        model.encoder_cam_feat_pos_embed.register_forward_hook(self._add_frame_embedding)

    def _reset_image_index(self, module, args):
        self._image_index = 0

    def _add_frame_embedding(self, module, args, output):
        cameras = max(1, len(self.config.image_features))
        frame = self._image_index // cameras
        self._image_index += 1
        if frame >= self.config.history_steps:
            raise RuntimeError(f"image {self._image_index - 1} maps to frame {frame}; "
                               f"only {self.config.history_steps} frames exist")
        return output + self.model.history_frame_embed[frame].to(output.dtype)[None, :, None, None]

    # ------------------------------------------------------------- batches

    def _stacked(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """[B, T, ...] observations -> ACT's inputs: the states concatenated,
        the images as one list ordered frame-major."""
        steps = self.config.history_steps
        batch = dict(batch)
        if self.config.robot_state_feature:
            state = batch[OBS_STATE]
            if state.ndim != 3 or state.shape[1] != steps:
                raise ValueError(f"{OBS_STATE} must be [B, {steps}, D]; got {tuple(state.shape)}")
            batch[OBS_STATE] = state.flatten(1)
        if self.config.image_features:
            images = []
            for frame in range(steps):
                for key in self.config.image_features:
                    if batch[key].ndim != 5 or batch[key].shape[1] != steps:
                        raise ValueError(f"{key} must be [B, {steps}, C, H, W]; "
                                         f"got {tuple(batch[key].shape)}")
                    images.append(batch[key][:, frame])
            batch[OBS_IMAGES] = images
        return batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """ACT's loss, on the stacked history. Not ACTPolicy.forward, which
        would rebuild the image list from single-frame keys."""
        batch = self._stacked(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae and log_sigma_x2_hat is not None:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            return l1_loss + mean_kld * self.config.kl_weight, loss_dict
        return l1_loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Takes [B, T, ...] observations, as training does."""
        self.eval()
        return self.model(self._stacked(batch))[0]

    # ------------------------------------------------------------- rollout

    def reset(self):
        super().reset()
        span = (self.config.history_steps - 1) * self.config.history_stride
        self._history = deque(maxlen=span + 1)

    def _observation_keys(self) -> list[str]:
        keys = list(self.config.image_features)
        if self.config.robot_state_feature:
            keys.append(OBS_STATE)
        return keys

    def _history_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Record this step's observation and return the [B, T, ...] history."""
        keys = self._observation_keys()
        for key in keys:
            if batch[key].ndim not in (2, 4):
                raise ValueError(f"select_action takes one frame; {key} is {tuple(batch[key].shape)}")
        current = {key: batch[key] for key in keys}
        if not self._history:
            # As the dataset pads an episode's opening frames: repeat the first.
            self._history.extend([current] * self._history.maxlen)
        else:
            self._history.append(current)
        frames = [self._history[i] for i in range(0, len(self._history), self.config.history_stride)]
        assert len(frames) == self.config.history_steps
        stacked = dict(batch)
        for key in keys:
            stacked[key] = torch.stack([frame[key] for frame in frames], dim=1)
        return stacked

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        stacked = self._history_batch(batch)
        if self.config.temporal_ensemble_coeff is not None:
            return self.temporal_ensembler.update(self.predict_action_chunk(stacked))
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(stacked)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
