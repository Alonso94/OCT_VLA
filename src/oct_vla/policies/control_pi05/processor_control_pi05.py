"""Use π0.5's standard processors for the object-conditioned residual."""

from __future__ import annotations

import torch
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

from .configuration_control_pi05 import ControlPI05Config


def make_control_pi05_pre_post_processors(
    config: ControlPI05Config, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
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
    return make_pi05_pre_post_processors(config, dataset_stats)
