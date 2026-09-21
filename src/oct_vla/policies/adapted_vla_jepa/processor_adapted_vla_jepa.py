"""VLA-JEPA processors for dual-arm raw gripper commands."""

from __future__ import annotations

import torch
from lerobot.processor import EnvTransition, ProcessorStep, TransitionKey
from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors

from .configuration_adapted_vla_jepa import AdaptedVLAJEPAConfig


class DualRawGripperProcessorStep(ProcessorStep):
    """Snap both RoboTwin grippers to raw ``0`` or ``1`` after unnormalising."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        dims = AdaptedVLAJEPAConfig.dual_gripper_dims_for(action.shape[-1])
        if dims is None:
            return transition
        result = dict(transition)
        snapped = action.clone()
        for index in dims:
            snapped[..., index] = (snapped[..., index] >= self.threshold).to(snapped.dtype)
        result[TransitionKey.ACTION] = snapped
        return result

    def transform_features(self, features):
        return features


def make_adapted_vla_jepa_pre_post_processors(
    config: AdaptedVLAJEPAConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
):
    # Before the normalizer is built, always. `validate_features` is what
    # retypes the entity tokens to ENV and pins ENV normalization to IDENTITY;
    # without it they stay typed STATE, and every backbone here maps STATE to
    # MEAN_STD over `dataset.meta.stats` -- which LeRobot computes across the
    # whole repo, train and validation together. So the guard is the only thing
    # standing between an entity_v2 run and a validation-statistics leak.
    #
    # Here rather than only in the model's __init__: control_groot and
    # control_smolvla called it in neither place, and the ones that do call it
    # there rely on `make_policy` running before `make_pre_post_processors` on
    # the same mutated config -- true in lerobot_train today, but an
    # undocumented ordering dependency in third-party code. It is idempotent.
    config.validate_features()
    pre, post = make_vla_jepa_pre_post_processors(config, dataset_stats)
    # The stock one-gripper processors are disabled by this config. Insert after
    # unnormalisation and before the final to-CPU step, in raw [0, 1] space.
    post.steps.insert(-1, DualRawGripperProcessorStep(config.gripper_threshold))
    return pre, post
