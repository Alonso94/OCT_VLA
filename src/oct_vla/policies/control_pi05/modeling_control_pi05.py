"""Object-conditioned π0.5 with a zero-initialized residual injection.

This plugin accepts already-extracted canonical scene tokens.  Perception and
RoboTwin remain outside the policy boundary.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package
from torch import Tensor, nn

from .configuration_control_pi05 import ControlPI05Config


def _batched(value: Tensor | None, *, unbatched_ndim: int) -> Tensor | None:
    """Add the batch dimension LeRobot's preprocessor does not add for us.

    `AddBatchDimensionProcessorStep` only batches the features declared on the
    policy config, and the object tokens are not among them -- they ride
    through the pipeline untouched. During training that is invisible, because
    the dataloader has already collated a batch. At inference there is no
    dataloader: `observation.state` arrives as [1, 16] while the tokens are
    still [8, 15], and `ObjectExpert` rejects the rank-2 tensor outright. So
    the policy would train happily and then die on its first eval step.
    """
    if value is None or value.ndim != unbatched_ndim:
        return value
    return value.unsqueeze(0)


class ObjectExpert(nn.Module):
    """Summarize a padded set of precomputed object tokens."""

    def __init__(self, config: ControlPI05Config, width: int) -> None:
        super().__init__()
        self.token_projection = nn.Sequential(
            nn.LayerNorm(config.object_token_dim),
            nn.Linear(config.object_token_dim, width),
            nn.GELU(),
        )
        self.queries = nn.Parameter(torch.empty(config.object_queries, width))
        nn.init.normal_(self.queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            width, config.object_attention_heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, tokens: Tensor | None, mask: Tensor | None) -> Tensor | None:
        if tokens is None:
            return None
        if tokens.ndim == 4:
            tokens = tokens[:, -1]
        if tokens.ndim != 3:
            raise ValueError(
                f"object tokens must be [B,N,D] or [B,T,N,D], got {tuple(tokens.shape)}"
            )
        memory = self.token_projection(tokens)
        if mask is None:
            padding_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            if mask.ndim == 3:
                mask = mask[:, -1]
            if mask.shape != tokens.shape[:2]:
                raise ValueError("object token mask must have shape [B,N]")
            padding_mask = ~mask.bool()
        all_padding = padding_mask.all(dim=1)
        if all_padding.any():
            memory, padding_mask = memory.clone(), padding_mask.clone()
            memory[all_padding, 0] = 0
            padding_mask[all_padding, 0] = False
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1).to(memory.dtype)
        attended, _ = self.cross_attention(
            queries, memory, memory, key_padding_mask=padding_mask, need_weights=False
        )
        return self.output_norm(attended.mean(dim=1))


class ControlPI05Pytorch(PI05Pytorch):
    def __init__(self, config: ControlPI05Config, rtc_processor=None) -> None:
        super().__init__(config, rtc_processor=rtc_processor)
        width = self.action_in_proj.out_features
        self.object_expert = ObjectExpert(config, width)
        self.object_injection = nn.Linear(width, width)
        nn.init.zeros_(self.object_injection.weight)
        nn.init.zeros_(self.object_injection.bias)
        self._object_inputs: tuple[Tensor | None, Tensor | None] | None = None

    def set_object_inputs(self, tokens: Tensor | None, mask: Tensor | None) -> None:
        self._object_inputs = (tokens, mask)

    def clear_object_inputs(self) -> None:
        self._object_inputs = None

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor):
        action_emb, pad_masks, att_masks, adarms_cond = super().embed_suffix(
            noisy_actions, timestep
        )
        if self._object_inputs is not None:
            context = self.object_expert(*self._object_inputs)
            if context is not None:
                action_emb = action_emb + self.object_injection(context).unsqueeze(1)
        return action_emb, pad_masks, att_masks, adarms_cond


class ControlPI05Policy(PI05Policy):
    """LeRobot policy wrapper; action handling remains PI0.5's standard path."""

    config_class = ControlPI05Config
    name = "control_pi05"

    def __init__(self, config: ControlPI05Config, **kwargs: Any) -> None:
        del kwargs
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = ControlPI05Pytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def _prepare_pretrained_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        state_dict = super()._prepare_pretrained_state_dict(state_dict)
        current = self.state_dict()
        for key, value in current.items():
            if key.startswith(("model.object_expert.", "model.object_injection.")):
                state_dict.setdefault(key, value)
        return state_dict

    def _set_object_inputs(self, batch: dict[str, Tensor]) -> None:
        self.model.set_object_inputs(
            _batched(batch.get(self.config.object_token_key), unbatched_ndim=2),
            _batched(batch.get(self.config.object_token_mask_key), unbatched_ndim=1),
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self._set_object_inputs(batch)
        try:
            return super().predict_action_chunk(batch, **kwargs)
        finally:
            self.model.clear_object_inputs()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        self._set_object_inputs(batch)
        try:
            images, img_masks = self._preprocess_images(batch)
            states, state_masks = self._prepare_memory_states(batch)
            actions = self.prepare_action(batch)
            losses = self.model.forward(
                images,
                img_masks,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                actions,
                self.model.sample_noise(actions.shape, actions.device),
                self.model.sample_time(actions.shape[0], actions.device),
                states=states,
                state_masks=state_masks,
            )[:, :, : self.config.output_features[ACTION].shape[0]]
            loss_dict = {"loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist()}
            if reduction == "none":
                result = losses.mean(dim=(1, 2))
                loss_dict["loss"] = result.mean().item()
                return result, loss_dict
            result = losses.mean()
            loss_dict["loss"] = result.item()
            return result, loss_dict
        finally:
            self.model.clear_object_inputs()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        targets = super()._get_default_peft_targets()
        targets["modules_to_save"] = list(targets.get("modules_to_save", [])) + [
            "model.object_expert",
            "model.object_injection",
        ]
        return targets
