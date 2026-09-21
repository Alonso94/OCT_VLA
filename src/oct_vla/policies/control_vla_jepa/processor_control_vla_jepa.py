"""Processors for the object-conditioned VLA-JEPA.

Pure delegation, as in the other plugins: the object tokens ride through the
pipeline untouched, and every other step must stay identical to the
unconditioned arm or the comparison is confounded. The module exists mainly so
it cannot be forgotten -- a plugin without one trains fine and then fails at
`make_pre_post_processors`, which is what evaluation calls.
"""

from __future__ import annotations

from oct_vla.policies.adapted_vla_jepa.processor_adapted_vla_jepa import (
    make_adapted_vla_jepa_pre_post_processors,
)

from .configuration_control_vla_jepa import ControlVLAJEPAConfig


def make_control_vla_jepa_pre_post_processors(
    config: ControlVLAJEPAConfig, dataset_stats: dict | None = None
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
    return make_adapted_vla_jepa_pre_post_processors(config, dataset_stats)
