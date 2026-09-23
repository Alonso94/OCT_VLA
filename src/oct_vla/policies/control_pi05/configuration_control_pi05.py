"""Configuration for the offline-object-conditioned π0.5 policy."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from oct_vla.policies.object_conditioning import ObjectTokenConfigMixin


@PreTrainedConfig.register_subclass("control_pi05")
@dataclass
class ControlPI05Config(ObjectTokenConfigMixin, PI05Config):
    """π0.5 with a zero-initialized residual from precomputed scene tokens."""

    object_representation: str = "legacy"
    object_entity_normalizer: dict | None = None

    object_token_key: str = "observation.object_tokens"
    object_token_mask_key: str = "observation.object_token_mask"
    #: Episode-stable slot ordering, used only by the role-stripped mode.
    object_token_rank_key: str = "observation.object_token_rank"
    #: Width of the tokens *as stored in the dataset*, before any ablation.
    object_token_dim: int = 15
    #: "full", or "role_stripped" to drop the role one-hot and the
    #: target-first ordering. See oct_vla.data.token_transforms.
    object_token_mode: str = "full"
    #: Evaluation-only control: permute tokens across objects. Training with
    #: this on would teach the policy to ignore the tokens, which is the
    #: opposite of what the control is meant to detect.
    object_token_shuffle: bool = False
    #: "controlvla" for the published dual-attention injection over the
    #: unpooled object set; "pooled" for the weaker broadcast residual every
    #: result recorded before 2026-09-20 was measured with.
    object_injection_mode: str = "controlvla"
    #: Learned code across entity slots. Off keeps the encoder
    #: permutation-invariant; on makes it order-sensitive, which is what the
    #: shuffled-token control needs in order to measure anything.
    object_entity_positional: bool = False
    object_max_entities: int = 16
    #: Used by the "pooled" mode only.
    object_queries: int = 4
    object_attention_heads: int = 8
    # Encoder-side arms (LPWM, arXiv:2603.04553), each on top of layerwise so
    # an arm isolates one addition, as for control_act. AdaLN: the pooled
    # scene is added, through a zero-initialised projection, to the vector the
    # action model already modulates its norms by (exact identity at init).
    object_adaln: bool = False
    # In-context: entity tokens prepended to the action-model sequence and
    # dropped from its output. Not identity at init (they enter attention).
    object_incontext: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_object_tokens()
        if self.object_injection_mode != "layerwise":
            return
        # The layerwise injection for pi0.5 is gated on a ContextVar set in the
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
                "layerwise pi0.5 requires gradient_checkpointing=False: the injection is "
                "gated on a ContextVar, which does not survive checkpoint recomputation "
                "in the autograd worker thread, so the backward pass would silently see "
                "an unconditioned network."
            )
        # Same root cause: Dynamo would have to trace the ContextVar set/reset
        # and a monkeypatched module-level function.
        if getattr(self, "compile_model", False):
            raise ValueError(
                "layerwise pi0.5 requires compile_model=False: the injection relies on a "
                "ContextVar and a patched module-level attention entrypoint, neither of "
                "which survives tracing intact."
            )
