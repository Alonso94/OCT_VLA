"""Dual-arm, object-conditioned VLA-JEPA adapter.

This module deliberately owns the few upstream assumptions that are unsafe for
this project: a single gripper, dict-derived camera order, and a two-view world
model.  The action-only path remains the upstream implementation except for the
object hook, so a world-model experiment cannot silently alter the baseline.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn.functional as F
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAModel, VLAJEPAPolicy
from lerobot.policies.vla_jepa.world_model import ActionConditionedVideoPredictor
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import Tensor

from oct_vla.policies.object_conditioning import (
    ObjectConditionedPolicyMixin,
    ObjectConditioning,
)

from .configuration_control_vla_jepa import ControlVLAJEPAConfig


def merge_ordered_views(embeddings: Tensor, batch: int, views: int) -> Tensor:
    """Join camera embeddings without changing their declared view order."""
    if embeddings.ndim != 3 or embeddings.shape[0] != batch * views:
        raise ValueError("video embeddings must have shape [B * V, S, D]")
    _, steps, width = embeddings.shape
    # [B*V,S,D] -> [B,S,V*D], with head, left wrist, right wrist in V.
    return embeddings.reshape(batch, views, steps, width).permute(0, 2, 1, 3).flatten(2)


class ControlVLAJEPAModel(VLAJEPAModel):
    """VLA-JEPA model with an explicitly three-view, causally masked WM path."""

    config: ControlVLAJEPAConfig

    def __init__(self, config: ControlVLAJEPAConfig) -> None:
        super().__init__(config)
        if config.enable_world_model:
            # Upstream accidentally uses jepa_tubelet_size as its view count.
            # The predictor is new in this finetune, so rebuilding it is the
            # correct place to change its input width from 2*D to 3*D.
            image_size = getattr(self.video_encoder.config, "image_size", None)
            if image_size is None:
                first = config.image_features[config.camera_keys[0]].shape
                image_size = first[-1]
            if isinstance(image_size, (tuple, list)):
                image_size = tuple(image_size)
            else:
                image_size = (image_size, image_size)
            self.video_predictor = ActionConditionedVideoPredictor(
                num_frames=config.num_video_frames // self.video_encoder.config.tubelet_size,
                img_size=image_size,
                patch_size=16,
                tubelet_size=1,
                embed_dim=self.video_encoder.config.hidden_size * config.jepa_num_views,
                action_embed_dim=self.qwen.model.config.hidden_size,
                predictor_embed_dim=self.video_encoder.config.hidden_size,
                depth=config.predictor_depth,
                num_heads=config.predictor_num_heads,
                mlp_ratio=config.predictor_mlp_ratio,
                num_action_tokens_per_step=config.num_action_tokens_per_timestep,
            )

    def _world_model_loss(
        self, videos: Tensor, action_tokens: Tensor, video_is_pad: Tensor | None = None
    ) -> Tensor:
        """Predict only valid future temporal positions from causal context.

        ``video_is_pad`` is [B,T] in raw-frame time.  A target tubelet is valid
        only when every raw frame in it is valid; padded targets contribute no
        loss and never produce an artificial terminal-frame objective.
        """
        if videos.shape[1] != self.config.jepa_num_views:
            raise ValueError(
                f"world model expected {self.config.jepa_num_views} ordered views, got {videos.shape[1]}"
            )
        b, v, t_frames, channels, height, width = videos.shape
        flat = videos.reshape(b * v, t_frames, channels, height, width)
        pixels = self.video_processor(
            videos=list(flat), return_tensors="pt", device=self.video_encoder.device, do_rescale=False
        )["pixel_values_videos"]
        tubelet = self.video_encoder.config.tubelet_size
        encoded_positions = self.config.num_video_frames // tubelet
        if encoded_positions < 2:
            return torch.zeros((), device=pixels.device)
        with torch.no_grad():
            # Config validation requires causal context, so no full-clip context
            # embedding can enter the predictor input.
            context = merge_ordered_views(
                self._causal_video_embeddings(pixels, tubelet, encoded_positions - 1), b, v
            )
            target_all = merge_ordered_views(
                self.video_encoder.get_vision_features(pixel_values_videos=pixels), b, v
            )
        tokens_per_position = target_all.shape[1] // encoded_positions
        target = target_all[:, tokens_per_position:, :]
        expected_actions = (encoded_positions - 1) * self.config.num_action_tokens_per_timestep
        if action_tokens.shape[1] < expected_actions:
            action_tokens = torch.cat(
                [action_tokens, action_tokens[:, -1:].repeat(1, expected_actions - action_tokens.shape[1], 1)],
                dim=1,
            )
        predicted = self.video_predictor(context.float(), action_tokens[:, :expected_actions].float())
        loss = F.l1_loss(predicted, target.float(), reduction="none")
        if video_is_pad is None:
            return loss.mean()
        if video_is_pad.ndim != 2 or video_is_pad.shape != (b, t_frames):
            raise ValueError("video_is_pad must have shape [B, T] matching the raw video frames")
        # Position 0 is context only; positions 1.. are prediction targets.
        usable = video_is_pad[:, : encoded_positions * tubelet].reshape(b, encoded_positions, tubelet)
        target_valid = ~usable[:, 1:].any(dim=-1)
        token_valid = target_valid.repeat_interleave(tokens_per_position, dim=1)
        return (loss * token_valid[:, :, None].to(loss.dtype)).sum() / (
            token_valid.sum().clamp_min(1) * loss.shape[-1]
        )

    def forward(self, *args: Any, video_is_pad: Tensor | None = None, **kwargs: Any) -> dict[str, Tensor]:
        # Keep this explicit rather than delegating: upstream has no temporal-pad
        # argument and would average invalid world-model targets.
        images = kwargs.pop("images") if "images" in kwargs else args[0]
        instructions = kwargs.pop("instructions") if "instructions" in kwargs else args[1]
        videos = kwargs.pop("videos", None)
        actions = kwargs.pop("actions", None)
        state = kwargs.pop("state", None)
        action_is_pad = kwargs.pop("action_is_pad", None)
        if kwargs:
            raise TypeError(f"unexpected VLA-JEPA model inputs: {sorted(kwargs)}")
        embodied, action_tokens = self._encode_qwen(
            images, instructions, need_action_tokens=self.config.enable_world_model
        )
        if self.config.enable_world_model and videos is not None:
            wm_loss = self._world_model_loss(videos, action_tokens, video_is_pad)
        else:
            wm_loss = torch.zeros((), device=embodied.device)
        if actions is None:
            return {"wm_loss": wm_loss}
        action_loss = self._action_loss(embodied, actions, state, action_is_pad)
        return {"action_loss": action_loss, "wm_loss": wm_loss * self.config.world_model_loss_weight}


class ControlVLAJEPAPolicy(ObjectConditionedPolicyMixin, VLAJEPAPolicy):
    """LeRobot wrapper with repeat-safe object conditioning and camera ordering."""

    config_class = ControlVLAJEPAConfig
    name = "control_vla_jepa"
    object_module_path = "model.action_model"

    def __init__(self, config: ControlVLAJEPAConfig, **kwargs: Any) -> None:
        # Do not call VLAJEPAPolicy.__init__: it creates the upstream model first
        # (including the wrong-width WM predictor) before it can be replaced.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        if dataset_meta := kwargs.get("dataset_meta"):
            features = dataset_meta.features
            if OBS_STATE in features:
                config.state_dim = features[OBS_STATE]["shape"][0]
            if ACTION in features:
                config.action_dim = features[ACTION]["shape"][0]
        self.config = config
        self.model = ControlVLAJEPAModel(config)
        self.reset()
        head = self.model.action_model
        head.object_conditioning = ObjectConditioning(config, head.input_embedding_dim)
        self._object_hook = head.action_encoder.register_forward_hook(self._condition_action_embeddings)

    def _condition_action_embeddings(self, _module: Any, _inputs: Any, output: Tensor) -> Tensor:
        """Match upstream ``Tensor.repeat(r, ...)`` diffusion ordering.

        ``repeat_interleave`` would pair each scene with r adjacent samples;
        upstream repeats whole batches, so it would condition the wrong scenes.
        """
        conditioning = self.object_conditioning
        if conditioning._inputs is None or conditioning._inputs[0] is None:
            return output
        tokens, mask = conditioning._inputs
        assert tokens is not None
        if output.shape[0] == tokens.shape[0]:
            return conditioning.residual(output)
        if output.shape[0] % tokens.shape[0]:
            raise RuntimeError(
                f"action batch {output.shape[0]} is not a whole repeat of object batch {tokens.shape[0]}"
            )
        repeats = output.shape[0] // tokens.shape[0]
        repeated_tokens = tokens.repeat(repeats, *([1] * (tokens.ndim - 1)))
        repeated_mask = (
            mask.repeat(repeats, *([1] * (mask.ndim - 1))) if mask is not None else None
        )
        original = conditioning._inputs
        conditioning.set_inputs(repeated_tokens, repeated_mask)
        try:
            return conditioning.residual(output)
        finally:
            conditioning._inputs = original

    def _prepare_model_inputs(self, batch: dict[str, Tensor], training: bool = True) -> dict[str, Any]:
        keys = self.config.camera_keys
        missing = [key for key in keys if key not in batch]
        if missing:
            raise KeyError(f"VLA-JEPA requires cameras in declared order; missing {missing}")
        batch_size = batch[keys[0]].shape[0]
        frames = []
        for key in keys:
            image = batch[key]
            if image.ndim == 5:
                image = image[:, 0]
            frames.append(self.model.qwen.to_pixel_values(image))
        inputs: dict[str, Any] = {
            "images": [[frame[index] for frame in frames] for index in range(batch_size)],
        }
        task = batch.get("task")
        inputs["instructions"] = (
            ["Execute the robot action."] * batch_size if task is None
            else [task] * batch_size if isinstance(task, str) else list(task)
        )
        if self.config.enable_world_model and training:
            views = [batch[key].unsqueeze(1) if batch[key].ndim == 4 else batch[key] for key in keys]
            inputs["videos"] = self.model.qwen.to_pixel_values(torch.stack(views, dim=1))
            pad_keys = [f"{key}_is_pad" for key in keys]
            pad_missing = [key for key in pad_keys if key not in batch]
            if pad_missing:
                # LeRobot generates padding per feature, not under a collective
                # observation.images key. A one-frame inference-style batch is
                # the only valid exception.
                if views[0].shape[1] > 1:
                    raise KeyError(f"world-model batches require camera padding keys: {pad_missing}")
                pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=views[0].device)
            else:
                pads = [batch[key].bool() for key in pad_keys]
                if any(item.shape != pads[0].shape for item in pads[1:]):
                    raise ValueError("camera padding masks must share the same [B,T] shape")
                if pads[0].shape != (batch_size, views[0].shape[1]):
                    raise ValueError("camera padding masks must match the assembled video")
                # Any unavailable camera invalidates a multi-view target frame.
                pad = torch.stack(pads, dim=0).any(dim=0)
            inputs["video_is_pad"] = pad
        actions = batch.get(ACTION)
        if actions is not None:
            inputs["actions"] = (actions.unsqueeze(1) if actions.ndim == 2 else actions).float()
            if (pad := batch.get("action_is_pad")) is not None:
                inputs["action_is_pad"] = pad
        state = batch.get(OBS_STATE)
        if state is not None:
            if state.ndim > 2:
                state = state[:, 0]
            inputs["state"] = (state.unsqueeze(1) if state.ndim == 2 else state).float()
        return inputs

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self._set_object_inputs(batch)
        try:
            return super().predict_action_chunk(batch, **kwargs)
        finally:
            self._clear_object_inputs()

    def forward(self, batch: dict[str, Tensor], **kwargs: Any):
        self._set_object_inputs(batch)
        # Deltas are [t, t+1, ...]. ObjectConditioning accepts temporal sets
        # too, where its generic set encoder intentionally selects the last
        # element. JEPA needs current visual context, so select t here.
        tokens, mask = self.object_conditioning._inputs
        if tokens is not None and tokens.ndim == 4:
            tokens = tokens[:, 0]
        if mask is not None and mask.ndim == 3:
            mask = mask[:, 0]
        self.object_conditioning.set_inputs(tokens, mask)
        try:
            return super().forward(batch, **kwargs)
        finally:
            self._clear_object_inputs()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        targets = super()._get_default_peft_targets()
        saves = list(targets.get("modules_to_save", []))
        # Random cross-embodiment projections must stay trainable and be saved;
        # PEFT otherwise serializes only LoRA adapters.
        saves.extend(
            f"model.action_model.{name}"
            for name in ("action_encoder", "state_encoder", "action_decoder")
        )
        targets["modules_to_save"] = list(dict.fromkeys(saves))
        return targets

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """Permit only declared reinitialisation; never hide a missing tensor."""
        from safetensors.torch import load_file

        checkpoint = load_file(model_file, device=map_location)
        current = model.state_dict()
        allow = tuple(model.config.reinit_modules or ())
        filtered: dict[str, Tensor] = {}
        skipped: set[str] = set()
        for key, value in checkpoint.items():
            if key in current and value.shape != current[key].shape:
                if not any(key.startswith(prefix) for prefix in allow):
                    raise ValueError(
                        f"Shape mismatch for {key!r} (checkpoint {tuple(value.shape)} vs "
                        f"model {tuple(current[key].shape)}) is not in reinit_modules."
                    )
                skipped.add(key)
                continue
            filtered[key] = value
        prepare = getattr(model, "_prepare_pretrained_state_dict", None)
        if prepare is not None:
            filtered = prepare(filtered)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        undeclared_missing = [key for key in missing if key not in skipped]
        if undeclared_missing or unexpected:
            details = []
            if undeclared_missing:
                details.append(f"missing keys: {undeclared_missing}")
            if unexpected:
                details.append(f"unexpected keys: {unexpected}")
            raise RuntimeError("Refusing incomplete VLA-JEPA checkpoint load; " + "; ".join(details))
        return model
