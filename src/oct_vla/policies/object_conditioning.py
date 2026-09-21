"""Object conditioning, as one module any backbone can mount.

Two mechanisms, selected by ``config.object_injection_mode``.

``"controlvla"`` is the legacy single-seam dual-attention approximation.
It keeps the object set *unpooled* but injects only at the action embedding
seam, rather than at every native attention layer::

    softmax(QKᵀ/√d)V + softmax(QK_zᵀ/√d)V_z

with ``K_z = W_z·Z + B_z`` and ``V_z`` **zero-initialised**, which is what makes
the conditioned policy numerically its pretrained self at step 0. The property
that matters is that the added term is *content-dependent*: action token ``i`` at
chunk step ``t`` computes its own weights over the objects, so the policy can
attend to different objects at different points in the trajectory -- which is
what a pick-then-place task needs.

``"pooled"`` is what this module did before, kept because every result recorded
so far used it and re-reading those checkpoints must stay possible::

    action_emb + injection(pool(Z))

It collapses the object set to one vector and adds the *same* bias to every
action token. That is strictly weaker, and it is why the shuffled-token control
is a no-op: the set is destroyed before it ever meets the action stream, so
there is no per-object structure left to scramble. It is a reasonable ablation
-- "does a global scene summary suffice?" -- but it is not ControlVLA, and this
module used to say it was.

Both modes preserve the zero-init guarantee, so arms stay comparable.

``object_entity_positional`` adds a learned code across entity slots. Off by
default, which keeps the encoder permutation-invariant. Turning it on makes it
order-sensitive *on purpose*: that is the valid permutation control this project
has lacked, borrowed from the Alfa-d entity encoders, where the same flag exists
for the same diagnostic.

Mounting on a new backbone is two lines -- construct with the host's own width,
and wrap the action embedding::

    self.object_conditioning = ObjectConditioning(config, width)
    ...
    return self.object_conditioning.residual(action_emb), *rest

The failure this guards against is silent: ``batch.get()`` returns ``None`` for
a missing key rather than raising, so a healthy loss curve does not prove the
conditioning is live. Check ``is_live`` after a smoke run.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from oct_vla.data.token_transforms import TOKEN_MODES, apply_token_mode, token_dim_for_mode

#: How the object set reaches the action stream. "controlvla" is the legacy
#: single-seam unpooled approximation; "layerwise" is the native-query,
#: full-layer path. "pooled" is the weaker broadcast residual kept so earlier
#: checkpoints stay loadable.
INJECTION_MODES = ("controlvla", "pooled", "layerwise")


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


def unwrap_object_conditioning(module: nn.Module | object) -> "ObjectConditioning | None":
    """Return the same object-conditioning copy PEFT will execute.

    ``ModulesToSaveWrapper`` deep-copies a new module for each adapter.  Hooks
    installed before wrapping must use that active copy; when adapters are
    disabled or none is active, PEFT instead executes ``original_module``.
    """
    if isinstance(module, ObjectConditioning):
        return module
    original = getattr(module, "original_module", None)
    copies = getattr(module, "modules_to_save", None)
    if copies is None:
        return None
    if getattr(module, "disable_adapters", False):
        return original if isinstance(original, ObjectConditioning) else None
    active = getattr(module, "active_adapter", None)
    if active is None:
        active_many = getattr(module, "active_adapters", ())
        active = active_many[0] if active_many else None
    # peft 0.20's AuxiliaryTrainingWrapper stores `_active_adapter` as a list,
    # so `active_adapter` is `['default']` rather than `'default'`. Indexing a
    # ModuleDict with it raises `TypeError: unhashable type: 'list'` from deep
    # inside torch, naming neither PEFT nor this module. Older versions return
    # the bare string, so normalise rather than assuming either shape.
    if isinstance(active, (list, tuple)):
        active = active[0] if active else None
    if active is None:
        return original if isinstance(original, ObjectConditioning) else None
    if active not in copies:
        raise ObjectConditioningError(
            f"PEFT active adapter {active!r} has no object-conditioning copy"
        )
    candidate = copies[active]
    return candidate if isinstance(candidate, ObjectConditioning) else None


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

    @property
    def env_state_feature(self):
        # Entity sets use ENV only to opt out of generic STATE normalization;
        # they are not ACT's flattened observation.environment_state input.
        return (
            self.input_features.get("observation.environment_state")
            if self.input_features
            else None
        )

    def validate_features(self) -> None:
        super().validate_features()
        if getattr(self, "object_representation", "legacy") == "entity_v2":
            from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

            # Entity tokens are retyped ENV purely to opt out of generic STATE
            # normalization -- the policy normalizes them itself, from
            # training-only statistics carried in its config.
            #
            # LeRobot resolves a normalization mode per feature *type*, not per
            # feature, so switching ENV to IDENTITY switches it for everything
            # typed ENV -- including a genuine flat observation.environment_state,
            # which would then silently stop being normalized. Refuse instead.
            # The combination is already unreachable from our exporter, which
            # rejects entity tokens alongside a privileged export, so this
            # guards a path that should not exist rather than one in use.
            if "observation.environment_state" in (self.input_features or {}):
                raise ValueError(
                    "entity_v2 retypes its tokens as ENV and sets ENV normalization to "
                    "IDENTITY, which would also stop normalizing the flat "
                    "observation.environment_state present in this config. Export one or "
                    "the other: entity tokens, or a privileged environment_state."
                )
            for name in ("observation.entity_tokens", "observation.entity_mask"):
                if name in self.input_features:
                    self.input_features[name] = PolicyFeature(
                        type=FeatureType.ENV, shape=self.input_features[name].shape
                    )
            self.normalization_mapping[FeatureType.ENV] = NormalizationMode.IDENTITY

    def validate_object_tokens(self) -> None:
        mode = getattr(self, "object_injection_mode", "pooled")
        if mode not in INJECTION_MODES:
            raise ValueError(
                f"object_injection_mode must be one of {INJECTION_MODES}, got {mode!r}"
            )
        if self.object_token_shuffle and not getattr(self, "object_entity_positional", False):
            # Not an error -- an existing checkpoint may carry both -- but the
            # combination measures nothing and has already been reported as a
            # control. Permuting whole token rows cannot change a
            # permutation-invariant encoder's output, so the "shuffled" arm is
            # numerically the unshuffled one.
            import warnings

            warnings.warn(
                "object_token_shuffle is a no-op without object_entity_positional: "
                "the object encoder is permutation-invariant, so permuting token "
                "rows cannot change its output. Set object_entity_positional=True "
                "to make the shuffled control measure something.",
                RuntimeWarning,
                stacklevel=2,
            )
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
        representation = getattr(self, "object_representation", "legacy")
        if representation not in ("legacy", "entity_v2"):
            raise ValueError("object_representation must be 'legacy' or 'entity_v2'")
        if representation == "entity_v2":
            self.object_token_dim = 17
            self.object_token_key = "observation.entity_tokens"
            self.object_token_mask_key = "observation.entity_mask"
            if self.object_entity_positional or self.object_token_mode != "full":
                raise ValueError("entity_v2 prohibits slot embeddings and oracle-role transforms")
            if mode != "layerwise":
                raise ValueError("entity_v2 requires layerwise injection")


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


class ObjectTokenEmbedding(nn.Module):
    """Project the stored tokens to the host's width, and mask the padding.

    Shared by both injection modes so they cannot disagree about what a token
    is. The padding handling is the part worth stating: a `three_object` scene
    fills 3 of 8 slots, so five rows are zeros, and a set encoder with no mask
    attends to them as if they were objects. Alfa-d's entity encoders have
    exactly this gap -- their vector wrapper zero-pads to the max entity count
    and no encoder builds a key-padding mask -- which is why it is explicit here.
    """

    def __init__(self, config: Any, width: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(config.effective_object_token_dim),
            nn.Linear(config.effective_object_token_dim, width),
            nn.GELU(),
        )
        # Off by default, which keeps the encoder permutation-invariant: the
        # tokens are a set and nothing distinguishes slot 3 from slot 5 except
        # its contents. Turning it on makes the encoder order-sensitive on
        # purpose, which is the only way the shuffled-token control measures
        # anything -- without it, permuting whole rows cannot change the output.
        self.entity_positional = bool(getattr(config, "object_entity_positional", False))
        if self.entity_positional:
            self.entity_code = nn.Parameter(
                torch.zeros(1, getattr(config, "object_max_entities", 16), width)
            )

    def forward(self, tokens: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
        """Return `[B, N, width]` memory and a `[B, N]` bool key-padding mask."""
        if tokens.ndim == 4:
            tokens = tokens[:, -1]
        if tokens.ndim != 3:
            raise ValueError(
                f"object tokens must be [B,N,D] or [B,T,N,D], got {tuple(tokens.shape)}"
            )
        memory = self.projection(tokens)
        if self.entity_positional:
            memory = memory + self.entity_code[:, : memory.shape[1]].to(memory.dtype)
        if mask is None:
            padding = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            if mask.ndim == 3:
                mask = mask[:, -1]
            if mask.shape != tokens.shape[:2]:
                raise ValueError("object token mask must have shape [B,N]")
            padding = ~mask.bool()
        # A row masked everywhere makes softmax divide by zero and return NaN,
        # which then poisons the whole batch. Unmask one slot and zero it, so
        # the attention reads an explicit "no objects" rather than a NaN.
        empty = padding.all(dim=1)
        if empty.any():
            memory, padding = memory.clone(), padding.clone()
            memory[empty, 0] = 0
            padding[empty, 0] = False
        return memory, padding


class ObjectCrossAttention(nn.Module):
    """Legacy single-seam added attention term over the unpooled object set.

    Implements ``softmax(QK_zᵀ/√d)V_z`` with the key and value projections zero
    initialised, per arXiv:2506.16211: "We zero-initialize the additional
    KV-projection layers to ensure the expert policy behaves similarly to the
    pre-trained general-purpose policy during the early stage of fine-tuning."

    The paper reuses the host block's own Q. We inject at the action-embedding
    seam rather than inside each block -- the seam is the one thing every
    backbone here has in common -- so Q is projected from the action embeddings.
    That keeps the property the pooled mode lacks: every action token computes
    its own weights over the objects.

    Written out rather than delegated to `nn.MultiheadAttention`, because that
    module fuses Q, K and V into one packed `in_proj_weight` and zeroing only
    the K and V thirds of a fused tensor is exactly the kind of thing that looks
    right and silently is not.
    """

    def __init__(self, config: Any, width: int) -> None:
        super().__init__()
        heads = config.object_attention_heads
        if width % heads:
            raise ValueError(
                f"object_attention_heads={heads} does not divide the host width {width}"
            )
        self.heads = heads
        self.head_dim = width // heads
        self.query_norm = nn.LayerNorm(width)
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        # The zero init, and the whole comparability argument. Both projections,
        # weight and bias: K_z = W_z·Z + B_z = 0 and V_z = 0.
        for projection in (self.to_k, self.to_v):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @property
    def is_live(self) -> bool:
        return bool(self.to_v.weight.any().item() or self.to_v.bias.any().item())

    def forward(self, action_emb: Tensor, memory: Tensor, padding: Tensor) -> Tensor:
        batch, steps, _ = action_emb.shape
        objects = memory.shape[1]
        memory = memory.to(action_emb.dtype)

        def split(x: Tensor, length: int) -> Tensor:
            return x.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q = split(self.to_q(self.query_norm(action_emb)), steps)
        k = split(self.to_k(memory), objects)
        v = split(self.to_v(memory), objects)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
        attended = torch.softmax(scores, dim=-1) @ v
        return attended.transpose(1, 2).reshape(batch, steps, -1)


class ObjectConditioning(nn.Module):
    """The object-conditioning mechanism: return a zero-initialized residual.

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
        self.mode = getattr(config, "object_injection_mode", "pooled")
        if self.mode not in INJECTION_MODES:
            raise ValueError(
                f"object_injection_mode must be one of {INJECTION_MODES}, got {self.mode!r}"
            )
        if self.mode == "controlvla":
            self.embedding = ObjectTokenEmbedding(config, width)
            self.attention = ObjectCrossAttention(config, width)
        elif self.mode == "layerwise":
            self.layers = nn.ModuleDict()
            self._layer_config = config
        else:
            self.expert = ObjectExpert(config, width)
            self.injection = nn.Linear(width, width)
            # Zero, not small-random: this is what makes the conditioned policy
            # numerically identical to its pretrained self at step 0.
            nn.init.zeros_(self.injection.weight)
            nn.init.zeros_(self.injection.bias)
        self._inputs: tuple[Tensor | None, Tensor | None] | None = None

    def add_layer(self, name: str, width: int, heads: int) -> nn.Module:
        from oct_vla.policies.layerwise_attention import LayerwiseObjectAttention

        if self.mode != "layerwise":
            raise ValueError("add_layer requires layerwise mode")
        if name in self.layers:
            raise ValueError(f"duplicate object layer {name}")
        self.layers[name] = LayerwiseObjectAttention(self._layer_config, width, heads=heads)
        return self.layers[name]

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
        if self.mode == "controlvla":
            return self.attention.is_live
        if self.mode == "layerwise":
            return any(layer.is_live for layer in self.layers.values())
        return bool(self.injection.weight.any().item())

    def residual(self, action_emb: Tensor) -> Tensor:
        """Add the object context to already-embedded action tokens.

        In `controlvla` mode each action token attends to the object set on its
        own, so the conditioning varies along the chunk. In `pooled` mode a
        single scene summary is broadcast over the chunk axis instead. A no-op
        when no tokens are set, so a host can be conditioned and unconditioned
        by the same code path.
        """
        if self._inputs is None:
            return action_emb
        tokens, mask = self._inputs
        if tokens is None:
            return action_emb
        if self.mode == "controlvla":
            memory, padding = self.embedding(tokens, mask)
            return action_emb + self.attention(action_emb, memory, padding)
        if self.mode == "layerwise":
            return action_emb
        context = self.expert(tokens, mask)
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
        active = unwrap_object_conditioning(module)
        if active is None:
            raise ObjectConditioningError(
                f"{self.object_module_path}.{self.object_module_attr} is "
                f"{type(module).__name__}, not ObjectConditioning. The host module "
                "must construct one in its __init__."
            )
        return active

    @property
    def _object_state_prefix(self) -> str:
        return f"{self.object_module_path}.{self.object_module_attr}."

    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """Condition the closed-loop path too, whatever route the backbone takes.

        Each plugin wraps `forward` and `predict_action_chunk`, which is enough
        only if `select_action` reaches the network through one of them. π0.5
        and GR00T do -- both call `self.predict_action_chunk`, so the override
        runs. **SmolVLA does not**: its `select_action` calls
        `self._get_action_chunk` directly, skipping `predict_action_chunk`
        entirely, so `_inputs` stayed None and `residual` returned the action
        embedding untouched.

        That arm would have trained *with* conditioning and evaluated
        *without* it, silently -- the training loss falls normally and the
        rollout is simply the unconditioned policy's, which is indistinguishable
        from "object conditioning did not help". Wrapping here rather than in
        each plugin means a backbone's internal routing cannot reintroduce it.

        Setting the inputs twice is harmless: the backbones that do route
        through `predict_action_chunk` re-set the same tensors, and the inner
        `finally` clears only after the forward that used them.
        """
        self._set_object_inputs(batch)
        try:
            return super().select_action(batch, **kwargs)
        finally:
            self._clear_object_inputs()

    def _set_object_inputs(self, batch: dict[str, Tensor]) -> None:
        # The single place tokens enter the model, so the arm's ablation is
        # applied here: training and evaluation then cannot disagree about the
        # layout, and the mode travels with the checkpoint's config.
        if getattr(self.config, "object_representation", "legacy") == "entity_v2":
            tokens = batched(batch.get("observation.entity_tokens"), unbatched_ndim=2)
            mask = batched(batch.get("observation.entity_mask"), unbatched_ndim=1)
            if tokens is None or mask is None:
                raise ValueError("entity_v2 requires both entity_tokens and entity_mask")
            if self.config.object_token_shuffle:
                order = torch.randperm(tokens.shape[-2], device=tokens.device)
                tokens, mask = tokens[..., order, :], mask[..., order]
        else:
            tokens, mask = apply_token_mode(
                batched(batch.get(self.config.object_token_key), unbatched_ndim=2),
                batched(batch.get(self.config.object_token_mask_key), unbatched_ndim=1),
                mode=self.config.object_token_mode,
                shuffle=self.config.object_token_shuffle,
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
