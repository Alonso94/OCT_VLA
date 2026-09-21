"""Conservative, explicit PEFT recipes for OCT-VLA backbones.

Launchers consume these plain mappings; this module never launches a run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptationRecipe:
    """Frozen-VLM LoRA settings plus backbone-specific projection training."""

    policy_type: str
    config_overrides: dict[str, object]
    peft_overrides: dict[str, object]


_LORA = {"method_type": "LORA", "r": 16, "lora_alpha": 32}


_RECIPES = {
    "control_pi05": AdaptationRecipe(
        "control_pi05",
        {"freeze_vision_encoder": True, "train_expert_only": True},
        dict(_LORA),
    ),
    "control_smolvla": AdaptationRecipe(
        "control_smolvla",
        {
            "freeze_vision_encoder": True,
            "train_expert_only": True,
            # This projection is newly trained for the embodiment.
            "train_state_proj": True,
        },
        dict(_LORA),
    ),
    "control_groot": AdaptationRecipe(
        "control_groot",
        {"lora_rank": 16, "lora_alpha": 32, "lora_full_model": False},
        dict(_LORA),
    ),
}


def adaptation_recipe(policy_type: str) -> AdaptationRecipe:
    """Return the audited default without silently widening trainable scope."""
    try:
        return _RECIPES[policy_type]
    except KeyError as error:
        raise ValueError(f"no OCT-VLA adaptation recipe for {policy_type!r}") from error
