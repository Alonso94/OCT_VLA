"""ControlVLA object conditioning, as one module any backbone can mount.

The mechanism is a zero-initialised broadcast residual on already-embedded
action tokens::

    action_emb + injection(pool(object_tokens))

Nothing about the host's attention mask, position ids or sequence length
changes, which is why it ports cheaply: a backbone needs only *somewhere* that
produces ``[B, chunk, width]`` action embeddings. The zero initialisation is the
ControlVLA idea and the thing that makes arms comparable -- at step 0 the
conditioned policy is numerically its pretrained self, so any deviation is
evidence the tokens are being used.

Everything used to be spread across two mixins that assumed the host lived at
``self.model``. π0.5 and SmolVLA both happen to satisfy that, so the assumption
was never tested; X-VLA holds its action embedding at ``model.transformer`` and
VLA-JEPA at ``model.action_model``. So the mechanism is now a single
``ObjectConditioning`` module, and the policy side resolves its location from
one ``object_module_path`` attribute rather than hard-coding ``model``.

Mounting it on a new backbone is two lines -- construct with the host's own
width, and wrap the action embedding::

    self.object_conditioning = ObjectConditioning(config, width)
    ...
    return self.object_conditioning.residual(action_emb), *rest

The failure this guards against is silent: ``batch.get()`` returns ``None`` for
a missing key rather than raising, so a healthy loss curve does not prove the
conditioning is live. Check ``injection.weight`` has left zero after a smoke run.
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
    still [8, 15], and the pooler rejects the rank-2 tensor outright. So the
    policy would train happily and then die on its first eval step.
    """
    if value is None or value.ndim != unbatched_ndim:
        return value
    return value.unsqueeze(0)


class ObjectConditioningError(RuntimeError):
    """A backbone is wired to its conditioning module incorrectly.

    Deliberately *not* an AttributeError. `nn.Module.__getattr__` catches those
    and re-raises its own generic message, so an AttributeError from inside a
    property on a policy surfaces as "'Policy' object has no attribute
    'object_conditioning'" -- discarding exactly the detail that says which path
    component was wrong.
    """


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    """Walk a dotted attribute path, naming what was missing if it breaks.

    A plain `getattr` chain would report `'PI05Pytorch' object has no attribute
    'action_model'` without saying which policy or which path produced it, and
    the paths differ per backbone precisely because this is the part that moves.
    """
    node = root
    parts = path.split(".")
    for index, attribute in enumerate(parts):
        if not hasattr(node, attribute):
            walked = ".".join(parts[:index]) or "<policy>"
            raise ObjectConditioningError(
                f"object_module_path {path!r} is wrong for "
                f"{type(root).__name__}: {walked} has no attribute {attribute!r}"
            )
        node = getattr(node, attribute)
    return node


class ObjectTokenConfigMixin:
    """Validation and derived widths shared by every object-conditioned config.

    Only behaviour lives here; the fields themselves are declared on each
    backbone's own dataclass, so neither config depends on dataclass field
    ordering across a multiple-inheritance chain.
    """

    @property
    def effective_object_token_dim(self) -> int:
        """Width the pooler projects, after the mode's ablation.

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


class ObjectConditioning(nn.Module):
    """The whole ControlVLA mechanism: pool the tokens, return a zero residual.

    Self-contained on purpose. A host backbone constructs one with its own
    embedding width and calls `residual` on its action tokens; it needs to know
    nothing about token modes, padding, batching or PEFT.

    The per-forward inputs are held on the module rather than threaded through
    the host's call signature, because the hosts differ: π0.5's `embed_suffix`
    returns four values and SmolVLA's three, X-VLA's hook is a `forward` two
    levels down. `ObjectConditionedPolicyMixin` sets and clears them around each
    forward pass, and `clear` is called in a `finally` so a raised exception
    cannot leave one batch's tokens attached to the next.
    """

    def __init__(self, config: Any, width: int) -> None:
        super().__init__()
        self.width = width
        self.expert = ObjectExpert(config, width)
        self.injection = nn.Linear(width, width)
        # Zero, not small-random: this is what makes the conditioned policy
        # numerically identical to its pretrained self at step 0.
        nn.init.zeros_(self.injection.weight)
        nn.init.zeros_(self.injection.bias)
        self._inputs: tuple[Tensor | None, Tensor | None] | None = None

    def set_inputs(self, tokens: Tensor | None, mask: Tensor | None) -> None:
        self._inputs = (tokens, mask)

    def clear(self) -> None:
        self._inputs = None

    @property
    def is_live(self) -> bool:
        """True once the injection has moved off its zero initialisation.

        The check worth running after a smoke run: a missing token key yields
        `None` rather than an error, so the loss falls either way.
        """
        return bool(self.injection.weight.any().item())

    def residual(self, action_emb: Tensor) -> Tensor:
        """Add the object context to already-embedded action tokens.

        Broadcast over the chunk axis: the scene summary conditions the whole
        action chunk, not one step of it. A no-op when no tokens are set, so a
        host can be conditioned and unconditioned by the same code path.
        """
        if self._inputs is None:
            return action_emb
        context = self.expert(*self._inputs)
        if context is None:
            return action_emb
        return action_emb + self.injection(context).unsqueeze(1)

    def attach_to(self, action_encoder: nn.Module):
        """Apply the residual to `action_encoder`'s output, via a forward hook.

        For backbones whose action embedding is produced by a submodule rather
        than returned from an overridable method. π0.5 and SmolVLA expose
        `embed_suffix`, so they call `residual` directly; X-VLA and VLA-JEPA
        build theirs inside a longer `forward` with no seam, and both happen to
        route it through a submodule named `action_encoder`.

        A hook rather than a wrapper module, deliberately. Wrapping would insert
        a level into the module tree and rename every one of the encoder's
        state-dict keys, which breaks loading the pretrained checkpoint -- the
        entire point of using these backbones. A hook changes no keys at all.
        Overriding the host's `forward` instead would mean copying forty lines
        of upstream code that then drifts on the next LeRobot upgrade.

        Returns the handle so a caller can remove it; nothing here does, since
        the hook is installed once at construction.
        """
        return action_encoder.register_forward_hook(
            lambda module, inputs, output: self.residual(output)
        )


class ObjectConditionedPolicyMixin:
    """The policy half: feeds the batch in, and keeps PEFT from dropping it.

    `object_module_path` is the one thing a new backbone overrides. Everything
    that needs to name the conditioning module -- the per-forward plumbing, the
    state-dict defaults, the PEFT `modules_to_save` -- derives from it, so a
    backbone that mounts the module somewhere other than `self.model` cannot end
    up half-wired.
    """

    #: Dotted path from the policy to the module owning `ObjectConditioning`.
    #: π0.5 and SmolVLA embed actions in `self.model`; X-VLA does it in
    #: `model.transformer`, VLA-JEPA in `model.action_model`.
    object_module_path: str = "model"

    #: Attribute name the host mounts the module under. Declared rather than
    #: assumed so the state-dict prefix and the PEFT target agree with wherever
    #: a backbone actually put it.
    object_module_attr: str = "object_conditioning"

    @property
    def object_conditioning(self) -> ObjectConditioning:
        host = resolve_module(self, self.object_module_path)
        module = getattr(host, self.object_module_attr, None)
        if not isinstance(module, ObjectConditioning):
            raise ObjectConditioningError(
                f"{self.object_module_path}.{self.object_module_attr} is "
                f"{type(module).__name__}, not ObjectConditioning. The host module "
                "must construct one in its __init__."
            )
        return module

    @property
    def _object_state_prefix(self) -> str:
        return f"{self.object_module_path}.{self.object_module_attr}."

    def _set_object_inputs(self, batch: dict[str, Tensor]) -> None:
        # The single place tokens enter the model, so the arm's ablation is
        # applied here: training and evaluation then cannot disagree about the
        # layout, and the mode travels with the checkpoint's config.
        tokens, mask = apply_token_mode(
            batched(batch.get(self.config.object_token_key), unbatched_ndim=2),
            batched(batch.get(self.config.object_token_mask_key), unbatched_ndim=1),
            mode=self.config.object_token_mode,
            shuffle=self.config.object_token_shuffle,
            # Absent on datasets exported before the stable ordering existed;
            # apply_token_mode then falls back to per-frame position sorting.
            rank=batched(batch.get(self.config.object_token_rank_key), unbatched_ndim=1),
        )
        self.object_conditioning.set_inputs(tokens, mask)

    def _clear_object_inputs(self) -> None:
        self.object_conditioning.clear()

    def _prepare_pretrained_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Chain to the backbone's hook if it has one, then add our defaults.

        Defined here rather than in each plugin because the hook is *not*
        universal: `PI05Policy` defines it, `SmolVLAPolicy` and
        `PreTrainedPolicy` do not. The SmolVLA plugin used to call
        `super()._prepare_pretrained_state_dict(...)` unconditionally, which was
        dead code -- so its object parameters were never defaulted into a
        pretrained load, and the arm would have trained with the object path
        pinned at its zero init while the loss fell normally.

        Probing with `getattr` makes that impossible to reintroduce, and means a
        new backbone works whether or not it happens to define the hook.
        """
        parent = getattr(super(), "_prepare_pretrained_state_dict", None)
        if parent is not None:
            state_dict = parent(state_dict)
        return self._add_object_state_defaults(state_dict)

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """Same reasoning: only four LeRobot policies define this hook.

        GR00T and VLA-JEPA do not, so chaining unconditionally would raise for
        exactly the backbones this refactor exists to support. For those, the
        config supplies `lora_target_modules` -- without it LeRobot refuses the
        run outright, which is the good failure: LoRA attaching to nothing while
        the object path still trains would look like a working experiment.
        """
        parent = getattr(super(), "_get_default_peft_targets", None)
        targets = dict(parent() or {}) if parent is not None else {}
        declared = getattr(self.config, "lora_target_modules", None)
        if declared and not targets.get("target_modules"):
            targets["target_modules"] = declared
        return self._add_object_peft_targets(targets)

    def _add_object_state_defaults(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Keep freshly initialised object parameters when loading base weights.

        The pretrained backbone has no conditioning module, so without this the
        checkpoint load would either fail on missing keys or leave them
        uninitialised.

        Applied unconditionally by `ObjectConditionedPolicyMixin`, not by
        chaining to a backbone hook. `PI05Policy` happens to define
        `_prepare_pretrained_state_dict`; `SmolVLAPolicy` does not, and calling
        `super()` on a hook only one parent has is how the SmolVLA arm ended up
        never defaulting its object parameters at all.
        """
        prefix = self._object_state_prefix
        for key, value in self.state_dict().items():
            if key.startswith(prefix):
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
            self._object_state_prefix.rstrip(".")
        ]
        return targets
