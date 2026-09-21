"""RGB-only policy wrapper around the repaired three-view JEPA model."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import Tensor

from oct_vla.policies.control_vla_jepa.modeling_control_vla_jepa import (
    ControlVLAJEPAModel,
    ControlVLAJEPAPolicy,
)

from .configuration_adapted_vla_jepa import AdaptedVLAJEPAConfig


class AdaptedVLAJEPAModel(ControlVLAJEPAModel):
    """The RGB variant has the same safe three-camera world-model plumbing."""

    config: AdaptedVLAJEPAConfig


class AdaptedVLAJEPAPolicy(VLAJEPAPolicy):
    """No-object RGB policy using declared camera order and actual pad keys."""

    config_class = AdaptedVLAJEPAConfig
    name = "adapted_vla_jepa"

    def __init__(self, config: AdaptedVLAJEPAConfig, **kwargs: Any) -> None:
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        if dataset_meta := kwargs.get("dataset_meta"):
            if OBS_STATE in dataset_meta.features:
                config.state_dim = dataset_meta.features[OBS_STATE]["shape"][0]
            if ACTION in dataset_meta.features:
                config.action_dim = dataset_meta.features[ACTION]["shape"][0]
        self.config = config
        self.model = AdaptedVLAJEPAModel(config)
        self.reset()

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def _prepare_model_inputs(self, batch: dict[str, Tensor], training: bool = True) -> dict[str, Any]:
        keys = self.config.camera_keys
        missing = [key for key in keys if key not in batch]
        if missing:
            raise KeyError(f"VLA-JEPA requires cameras in declared order; missing {missing}")
        batch_size = batch[keys[0]].shape[0]
        frames = [
            self.model.qwen.to_pixel_values(batch[key][:, 0] if batch[key].ndim == 5 else batch[key])
            for key in keys
        ]
        task = batch.get("task")
        inputs: dict[str, Any] = {
            "images": [[frame[index] for frame in frames] for index in range(batch_size)],
            "instructions": (["Execute the robot action."] * batch_size if task is None else
                             [task] * batch_size if isinstance(task, str) else list(task)),
        }
        if self.config.enable_world_model and training:
            views = [batch[key].unsqueeze(1) if batch[key].ndim == 4 else batch[key] for key in keys]
            inputs["videos"] = self.model.qwen.to_pixel_values(torch.stack(views, dim=1))
            pad_keys = [f"{key}_is_pad" for key in keys]
            absent = [key for key in pad_keys if key not in batch]
            if absent and views[0].shape[1] > 1:
                raise KeyError(f"world-model batches require camera padding keys: {absent}")
            pads = ([torch.zeros(batch_size, 1, dtype=torch.bool, device=views[0].device)] if absent
                    else [batch[key].bool() for key in pad_keys])
            if any(pad.shape != pads[0].shape for pad in pads[1:]) or pads[0].shape != views[0].shape[:2]:
                raise ValueError("camera padding masks must share the assembled video [B,T] shape")
            inputs["video_is_pad"] = torch.stack(pads, dim=0).any(dim=0)
        actions = batch.get(ACTION)
        if actions is not None:
            inputs["actions"] = (actions.unsqueeze(1) if actions.ndim == 2 else actions).float()
            if (pad := batch.get("action_is_pad")) is not None:
                inputs["action_is_pad"] = pad
        state = batch.get(OBS_STATE)
        if state is not None:
            # Deltas are forward-looking: position zero is the current state.
            state = state[:, 0] if state.ndim > 2 else state
            inputs["state"] = (state.unsqueeze(1) if state.ndim == 2 else state).float()
        return inputs

    def _get_default_peft_targets(self) -> dict[str, Any]:
        return {
            "target_modules": self.config.lora_target_modules,
            "modules_to_save": [
                "model.action_model.action_encoder",
                "model.action_model.state_encoder",
                "model.action_model.action_decoder",
            ],
        }

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        return ControlVLAJEPAPolicy._load_as_safetensor(model, model_file, map_location, strict)
