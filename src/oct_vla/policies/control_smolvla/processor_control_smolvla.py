"""Use SmolVLA's standard processors for the object-conditioned residual.

LeRobot resolves processors by naming convention -- `make_{policy_type}_…` in
the `processor_*` module beside the config -- so a plugin without this file
loads for training but fails at `make_pre_post_processors`, which is what
evaluation calls. The object tokens need no processing of their own: they are
already canonical, and the policy applies any ablation itself.
"""

from __future__ import annotations

import torch
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from .configuration_control_smolvla import ControlSmolVLAConfig


def make_control_smolvla_pre_post_processors(
    config: ControlSmolVLAConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
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
    return make_smolvla_pre_post_processors(config, dataset_stats)
