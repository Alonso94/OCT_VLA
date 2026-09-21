"""Processors for the object-conditioned GR00T.

Pure delegation, as in the other plugins. GR00T is the one policy with real
camera-key expectations: `processor_groot.py` matches the checkpoint's
video-modality keys against the dataset's cameras and, on no match, falls back
to *alphabetical* order with only a warning. So the rename map matters more here
than anywhere else -- a mismatch feeds the wrong view and shows up only as a
success rate.
"""

from __future__ import annotations

from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

from .configuration_control_groot import ControlGrootConfig


def make_control_groot_pre_post_processors(
    config: ControlGrootConfig, dataset_stats: dict | None = None
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
    return make_groot_pre_post_processors(config, dataset_stats)
