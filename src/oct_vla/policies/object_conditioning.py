"""Backbone-independent half of the object-conditioned policies.

π0.5 and SmolVLA differ only in where the residual is injected: both expose an
``embed_suffix`` that builds the action tokens, but π0.5 returns four values
and SmolVLA three. Everything else -- how tokens are summarised, how the
ablation modes are applied, which parameters PEFT must save -- is identical,
and lives here so the two arms cannot quietly diverge into measuring different
things.

The zero-initialised injection is the ControlVLA idea: at step 0 the policy is
exactly its pretrained self, and any deviation from that is evidence the object
tokens are being used. That is why ``object_injection`` starts at zero and why
its weight is the thing to check after a smoke run -- ``batch.get()`` returns
``None`` for a missing key rather than raising, so a healthy loss curve does not
prove the conditioning is live.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from oct_vla.data.token_transforms import TOKEN_MODES, apply_token_mode, token_dim_for_mode


def batched(value: Tensor | None, *, unbatched_ndim: int) -> Tensor | None:
    """Add the batch dimension LeRobot's preprocessor does not add for us.

    `AddBatchDimensionProcessorStep` only batches the features declared on the
    policy config, and the object tokens are not among them -- they ride
    through the pipeline untouched. During training that is invisible, because
    the dataloader has already collated a batch. At inference there is no
    dataloader: `observation.state` arrives as [1, 16] while the tokens are
    still [8, 15], and `ObjectExpert` rejects the rank-2 tensor outright. So
    the policy would train happily and then die on its first eval step.
    """
    if value is None or value.ndim != unbatched_ndim:
        return value
    return value.unsqueeze(0)


class ObjectTokenConfigMixin:
    """Validation and derived widths shared by every object-conditioned config.

    Only behaviour lives here; the fields themselves are declared on each
    backbone's own dataclass, so neither config depends on dataclass field
    ordering across a multiple-inheritance chain.
    """

    @property
    def effective_object_token_dim(self) -> int:
        """Width the ObjectExpert projects, after the mode's ablation.

        Derived rather than configured: a hand-set width that disagreed with
        the mode would surface as a shape error deep inside cross-attention,
        long after the run started.
        """
        return token_dim_for_mode(self.object_token_dim, self.object_token_mode)

    def validate_object_tokens(self) -> None:
        if self.object_token_mode not in TOKEN_MODES:
            raise ValueError(
                f"object_token_mode must be one of {TOKEN_MODES}, got {self.object_token_mode!r}"
            )
        if self.object_token_dim <= 0:
            raise ValueError("object_token_dim must be positive")
        if self.object_queries <= 0:
            raise ValueError("object_queries must be positive")
        if self.object_attention_heads <= 0:
            raise ValueError("object_attention_heads must be positive")


class ObjectExpert(nn.Module):
    """Summarize a padded set of precomputed object tokens."""

    def __init__(self, config: Any, width: int) -> None:
        super().__init__()
        self.token_projection = nn.Sequential(
            nn.LayerNorm(config.effective_object_token_dim),
            nn.Linear(config.effective_object_token_dim, width),
            nn.GELU(),
        )
        self.queries = nn.Parameter(torch.empty(config.object_queries, width))
        nn.init.normal_(self.queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            width, config.object_attention_heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, tokens: Tensor | None, mask: Tensor | None) -> Tensor | None:
        if tokens is None:
            return None
        if tokens.ndim == 4:
            tokens = tokens[:, -1]
        if tokens.ndim != 3:
            raise ValueError(
                f"object tokens must be [B,N,D] or [B,T,N,D], got {tuple(tokens.shape)}"
            )
        memory = self.token_projection(tokens)
        if mask is None:
            padding_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            if mask.ndim == 3:
                mask = mask[:, -1]
            if mask.shape != tokens.shape[:2]:
                raise ValueError("object token mask must have shape [B,N]")
            padding_mask = ~mask.bool()
        all_padding = padding_mask.all(dim=1)
        if all_padding.any():
            memory, padding_mask = memory.clone(), padding_mask.clone()
            memory[all_padding, 0] = 0
            padding_mask[all_padding, 0] = False
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1).to(memory.dtype)
        attended, _ = self.cross_attention(
            queries, memory, memory, key_padding_mask=padding_mask, need_weights=False
        )
        return self.output_norm(attended.mean(dim=1))


class ObjectInjectionMixin:
    """The ``nn.Module`` half: owns the expert and the zero-init residual."""

    def init_object_conditioning(self, config: Any, width: int) -> None:
        self.object_expert = ObjectExpert(config, width)
        self.object_injection = nn.Linear(width, width)
        nn.init.zeros_(self.object_injection.weight)
        nn.init.zeros_(self.object_injection.bias)
        self._object_inputs: tuple[Tensor | None, Tensor | None] | None = None

    def set_object_inputs(self, tokens: Tensor | None, mask: Tensor | None) -> None:
        self._object_inputs = (tokens, mask)

    def clear_object_inputs(self) -> None:
        self._object_inputs = None

    def object_residual(self, action_emb: Tensor) -> Tensor:
        """Add the object context to already-embedded action tokens.

        Broadcast over the chunk axis: the scene summary conditions the whole
        action chunk, not one step of it.
        """
        if self._object_inputs is None:
            return action_emb
        context = self.object_expert(*self._object_inputs)
        if context is None:
            return action_emb
        return action_emb + self.object_injection(context).unsqueeze(1)


class ObjectConditionedPolicyMixin:
    """The policy half: feeds the batch in, and keeps PEFT from dropping it."""

    def _set_object_inputs(self, batch: dict[str, Tensor]) -> None:
        # The single place tokens enter the model, so the arm's ablation is
        # applied here: training and evaluation then cannot disagree about the
        # layout, and the mode travels with the checkpoint's config.
        tokens, mask = apply_token_mode(
            batched(batch.get(self.config.object_token_key), unbatched_ndim=2),
            batched(batch.get(self.config.object_token_mask_key), unbatched_ndim=1),
            mode=self.config.object_token_mode,
            shuffle=self.config.object_token_shuffle,
        )
        self.model.set_object_inputs(tokens, mask)

    def _add_object_state_defaults(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Keep freshly initialised object parameters when loading base weights.

        The pretrained backbone has no object expert, so without this the
        checkpoint load would either fail on missing keys or leave them
        uninitialised.
        """
        for key, value in self.state_dict().items():
            if key.startswith(("model.object_expert.", "model.object_injection.")):
                state_dict.setdefault(key, value)
        return state_dict

    def _add_object_peft_targets(self, targets: dict[str, Any]) -> dict[str, Any]:
        """Train the object path in full, not through a low-rank adapter.

        These modules are new and randomly initialised, so there is no
        pretrained weight for LoRA to correct; `modules_to_save` is what makes
        them trainable *and* saved. Omitting it is the silent failure this
        whole design guards against -- the run trains, the loss falls, and the
        object path never moves off its zero init.
        """
        targets["modules_to_save"] = list(targets.get("modules_to_save", [])) + [
            "model.object_expert",
            "model.object_injection",
        ]
        return targets
