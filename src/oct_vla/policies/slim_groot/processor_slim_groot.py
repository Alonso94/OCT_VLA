"""Processors for slim_groot: exactly GR00T's."""

from __future__ import annotations

from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

from .configuration_slim_groot import SlimGrootConfig


def make_slim_groot_pre_post_processors(config: SlimGrootConfig, dataset_stats: dict | None = None):
    return make_groot_pre_post_processors(config, dataset_stats)
