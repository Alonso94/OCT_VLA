"""pi0.5 conditioned on the entity set (see oct_vla.policies.conditioning)."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from oct_vla.policies.object_conditioning import ObjectConditioningConfig


@PreTrainedConfig.register_subclass("control_pi05")
@dataclass
class ControlPI05Config(ObjectConditioningConfig, PI05Config):
    """Stock pi0.5 plus the arm named by ``object_conditioning``."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # KV on pi0.5 is gated on a ContextVar set in the
        # patched `paligemma_with_expert.forward`, because the monkeypatched
        # Gemma attention receives only an attention module and needs to know
        # which batch and which layer it is serving.
        #
        # Non-reentrant checkpointing replays the forward during backward, in
        # the autograd engine's worker thread. `torch.utils.checkpoint` saves
        # RNG and autocast state but not contextvars, so the replay sees the
        # ContextVar default of None and omits the object residual -- producing
        # gradients for a network that is not the one that ran forward. No
        # shape changes, no error, and checkpoint does not compare recomputed
        # values by default. Refuse rather than train something subtly wrong.
        if self.gradient_checkpointing:
            raise ValueError(
                "KV-conditioned pi0.5 requires gradient_checkpointing=False: the injection is "
                "gated on a ContextVar, which does not survive checkpoint recomputation "
                "in the autograd worker thread, so the backward pass would silently see "
                "an unconditioned network."
            )
        # Same root cause: Dynamo would have to trace the ContextVar set/reset
        # and a monkeypatched module-level function.
        if getattr(self, "compile_model", False):
            raise ValueError(
                "KV-conditioned pi0.5 requires compile_model=False: the injection relies on a "
                "ContextVar and a patched module-level attention entrypoint, neither of "
                "which survives tracing intact."
            )
