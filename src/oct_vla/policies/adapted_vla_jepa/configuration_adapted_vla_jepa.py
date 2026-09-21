"""Safe RGB-only defaults for adapting VLA-JEPA to RoboTwin."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig


@PreTrainedConfig.register_subclass("adapted_vla_jepa")
@dataclass
class AdaptedVLAJEPAConfig(VLAJEPAConfig):
    """Three-camera VLA-JEPA with dual raw ``{0, 1}`` gripper commands."""

    camera_keys: tuple[str, str, str] = (
        "observation.images.head",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    jepa_num_views: int = 3
    # The action-only objective is the baseline. A world-model run must opt in.
    enable_world_model: bool = False
    causal_world_model_context: bool = False
    # Disable the upstream single-gripper (index 6) processor. The adapted
    # processor snaps both grippers after unnormalisation in raw [0, 1] space.
    binarize_gripper_action: bool = False
    pre_snap_gripper_action: bool = False
    # VLA-JEPA itself does not supply defaults. Without these, PEFT can attach
    # to nothing while still appearing to train the newly saved projections.
    lora_target_modules: str = field(
        default=(
            r"model\.action_model\.model\.transformer_blocks\.\d+\.attn1\.to_(q|v)"
        )
    )
    reinit_modules: list[str] | None = field(
        default_factory=lambda: [
            "model.action_model.action_encoder",
            "model.action_model.state_encoder",
            "model.action_model.action_decoder",
            # Only these dimensions change from 2 to 3 camera embeddings.
            "model.video_predictor.predictor_embed",
            "model.video_predictor.predictor_proj",
        ]
    )

    def __post_init__(self) -> None:
        requested_world_model = self.enable_world_model
        super().__post_init__()
        if requested_world_model and self.freeze_qwen:
            raise ValueError(
                "freeze_qwen=True contradicts enable_world_model=True: upstream disables the world model."
            )
        if self.enable_world_model and not self.causal_world_model_context:
            raise ValueError(
                "enable_world_model=True requires causal_world_model_context=True to prevent leakage."
            )
        if self.jepa_num_views != len(self.camera_keys):
            raise ValueError("jepa_num_views must equal the number of camera_keys")
        if len(self.camera_keys) != 3 or len(set(self.camera_keys)) != 3:
            raise ValueError("camera_keys must declare head, left wrist, and right wrist exactly once")

    @staticmethod
    def dual_gripper_dims_for(action_dim: int) -> tuple[int, int] | None:
        if action_dim == 14:
            return (6, 13)
        if action_dim == 16:
            return (7, 15)
        return None
