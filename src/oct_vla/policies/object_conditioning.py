"""The shared layer between the conditioning method and each backbone.

The method itself is in ``oct_vla.policies.conditioning``. This module holds what
every backbone adapter (``control_*``) needs around it:

* `ObjectConditioningConfig` -- the config fields every conditioned policy
  shares, chiefly ``object_conditioning`` (``kv`` | ``kv_adaln`` | ``kv_tokens``).
* `ObjectConditioning` -- the one module a host mounts. It owns the KV branch of
  every hooked attention layer and, per arm, an AdaLN or token branch. Its
  subtree is the only thing a stage-2 run adds to a stage-1 checkpoint.
* `ObjectConditionedPolicyMixin` -- feeds each batch's entities in and clears
  them afterwards, keeps PEFT from freezing or dropping the new subtree, and
  verifies every checkpoint load.

The failure this layer guards against is silent: ``batch.get()`` returns None
for a missing key rather than raising, and a conditioning path that never runs
still gives a falling loss. `ObjectConditioning.is_live` is the check to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from oct_vla.policies.conditioning.entity import ENTITY_MASK, ENTITY_TOKENS, TOKEN_DIM
from oct_vla.policies.stage_loading import VerifiedLoadMixin

#: The arms a conditioned policy can be. Nested: each adds one path to ``kv``.
CONDITIONING = ("kv", "kv_adaln", "kv_tokens")

#: Checkpoints written before the 2026-09-23 cleanup carry these fields. Each is
#: accepted only at the one value the code still implements (None = any value,
#: for fields that never affected an entity-set policy); anything else is a
#: mechanism that no longer exists here.
_LEGACY_FIELDS = {
    "object_injection_mode": ("layerwise",),
    "object_representation": ("entity_v2",),
    "object_token_mode": ("full",),
    "object_entity_positional": (False,),
    "object_token_shuffle": (False,),
    "object_token_dim": (TOKEN_DIM,),
    "object_token_key": (ENTITY_TOKENS,),
    "object_token_mask_key": (ENTITY_MASK,),
    "object_token_rank_key": None,
    "object_queries": None,
}
LEGACY_TAG = "stageA-2026-09-23"


class ObjectConditioningError(RuntimeError):
    """A backbone is wired to its conditioning module incorrectly.

    Deliberately *not* an AttributeError: `nn.Module.__getattr__` swallows those
    and re-raises a generic message that drops the useful detail.
    """


class LegacyCheckpointError(ValueError):
    """A checkpoint uses a conditioning mechanism removed in the cleanup."""


def batched(value: Tensor | None, *, unbatched_ndim: int) -> Tensor | None:
    """Add the batch dimension LeRobot's preprocessor does not add for us.

    `AddBatchDimensionProcessorStep` batches only the features declared on the
    policy config, and the entity tensors ride through untouched. In training
    the dataloader has already batched them; at inference they arrive
    unbatched, so without this a policy trains happily and dies on its first
    rollout step.
    """
    if value is None or value.ndim != unbatched_ndim:
        return value
    return value.unsqueeze(0)


def unwrap_object_conditioning(module: nn.Module | object) -> ObjectConditioning | None:
    """The copy of the conditioning module PEFT will actually execute.

    ``ModulesToSaveWrapper`` deep-copies a module per adapter; hooks must use the
    active copy, or ``original_module`` when adapters are disabled.
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
    # peft 0.20 stores `_active_adapter` as a list; indexing a ModuleDict with
    # ['default'] raises an unhashable-type error from deep inside torch.
    if isinstance(active, (list, tuple)):
        active = active[0] if active else None
    if active is None:
        return original if isinstance(original, ObjectConditioning) else None
    if active not in copies:
        raise ObjectConditioningError(f"PEFT active adapter {active!r} has no conditioning copy")
    candidate = copies[active]
    return candidate if isinstance(candidate, ObjectConditioning) else None


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    """Walk a dotted attribute path, naming the missing component if it breaks."""
    node = root
    parts = path.split(".")
    for index, attribute in enumerate(parts):
        if not hasattr(node, attribute):
            walked = ".".join(parts[:index]) or "<policy>"
            raise ObjectConditioningError(
                f"object_module_path {path!r} is wrong for {type(root).__name__}: "
                f"{walked} has no attribute {attribute!r}"
            )
        node = getattr(node, attribute)
    return node


@dataclass
class ObjectConditioningConfig:
    """Fields and checks shared by every object-conditioned policy config.

    Put first among the bases: ``class ControlXConfig(ObjectConditioningConfig,
    XConfig)``.
    """

    #: Which arm: "kv" (ControlVLA's KV term at every hooked attention layer),
    #: "kv_adaln" (kv + scene AdaLN), or "kv_tokens" (kv + entity tokens).
    object_conditioning: str = "kv"
    #: Position/size standardisation, fitted on the training split only
    #: (scripts/entity_training_args.py writes it).
    object_entity_normalizer: dict | None = None
    object_max_entities: int = 16
    #: Heads of the AdaLN pool. KV uses each host layer's own head count.
    object_attention_heads: int = 8
    #: ACT's token gate starts here (see conditioning/tokens.py).
    object_tokens_gate_init: float = -4.0

    # --- accepted only so pre-cleanup checkpoints load; see _migrate_legacy ---
    object_injection_mode: str | None = None
    object_representation: str | None = None
    object_token_mode: str | None = None
    object_entity_positional: bool | None = None
    object_token_shuffle: bool | None = None
    object_token_dim: int | None = None
    object_token_key: str | None = None
    object_token_mask_key: str | None = None
    object_token_rank_key: str | None = None
    object_queries: int | None = None
    object_adaln: bool | None = None
    object_incontext: bool | None = None
    object_incontext_gate_init: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._migrate_legacy()
        if self.object_conditioning not in CONDITIONING:
            raise ValueError(
                f"object_conditioning must be one of {CONDITIONING}, got "
                f"{self.object_conditioning!r}"
            )
        if self.object_attention_heads <= 0:
            raise ValueError("object_attention_heads must be positive")

    def _migrate_legacy(self) -> None:
        for name, supported in _LEGACY_FIELDS.items():
            value = getattr(self, name)
            if value is not None and supported is not None and value not in supported:
                raise LegacyCheckpointError(
                    f"{name}={value!r}: that mechanism was removed in the cleanup. "
                    f"Load this checkpoint from git tag {LEGACY_TAG}."
                )
            setattr(self, name, None)
        adaln, tokens = bool(self.object_adaln), bool(self.object_incontext)
        if adaln and tokens:
            raise LegacyCheckpointError("object_adaln and object_incontext were both set")
        if adaln or tokens:
            arm = "kv_adaln" if adaln else "kv_tokens"
            if self.object_conditioning not in ("kv", arm):
                raise LegacyCheckpointError(
                    f"object_conditioning={self.object_conditioning!r} contradicts the legacy "
                    f"flag selecting {arm!r}"
                )
            self.object_conditioning = arm
        if self.object_incontext_gate_init is not None:
            self.object_tokens_gate_init = float(self.object_incontext_gate_init)
        self.object_adaln = self.object_incontext = self.object_incontext_gate_init = None

    @property
    def env_state_feature(self):
        # The entity tensors are typed ENV only to opt out of STATE
        # normalisation. They are not ACT's flat `observation.environment_state`
        # input, which ACT would otherwise pick up from any ENV feature.
        features = self.input_features or {}
        return features.get("observation.environment_state")

    def validate_features(self) -> None:
        super().validate_features()
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

        # LeRobot picks a normalisation mode per feature *type*, so setting ENV
        # to IDENTITY (the policy standardises entities itself) would also stop
        # normalising a genuine flat environment_state. Refuse the combination.
        if "observation.environment_state" in (self.input_features or {}):
            raise ValueError(
                "an entity-conditioned policy cannot also take observation.environment_state: "
                "both are typed ENV, and ENV normalisation is IDENTITY for the entities"
            )
        for name in (ENTITY_TOKENS, ENTITY_MASK):
            if name in self.input_features:
                self.input_features[name] = PolicyFeature(
                    type=FeatureType.ENV, shape=self.input_features[name].shape
                )
        self.normalization_mapping[FeatureType.ENV] = NormalizationMode.IDENTITY


class ObjectConditioning(nn.Module):
    """The module a host mounts: the KV branches, plus the arm's extra branch.

    Attribute names are part of the checkpoint format and predate the cleanup:
    ``layers`` (KV, one per hooked attention layer), ``adaln`` (a `SceneAdaLN`
    or `SceneVector`), and ``incontext`` (the `EntityTokens` branch). Hosts set
    ``adaln`` / ``incontext`` themselves, because where they attach differs.

    The current batch's entities are held here (`set_inputs` / `clear`) rather
    than threaded through each host's call signature, which differs per host;
    the policy mixin clears them in a ``finally``.
    """

    def __init__(self, config: Any, width: int) -> None:
        super().__init__()
        self.width = width
        self.layers = nn.ModuleDict()
        self._config = config
        self._inputs: tuple[Tensor | None, Tensor | None] | None = None

    def add_layer(self, name: str, width: int, heads: int) -> nn.Module:
        from oct_vla.policies.conditioning.kv import KVAttention

        if name in self.layers:
            raise ValueError(f"duplicate object layer {name}")
        self.layers[name] = KVAttention(self._config, width, heads=heads)
        return self.layers[name]

    def set_inputs(self, tokens: Tensor | None, mask: Tensor | None) -> None:
        self._inputs = (tokens, mask)

    def clear(self) -> None:
        self._inputs = None

    @property
    def is_live(self) -> bool:
        """True once any branch has moved off its initialisation."""
        branches = [*self.layers.values()]
        branches += [b for b in (getattr(self, "adaln", None), getattr(self, "incontext", None)) if b]
        return any(branch.is_live for branch in branches)


class ObjectConditionedPolicyMixin(VerifiedLoadMixin):
    """The policy half: feeds the batch in, and keeps PEFT from dropping it.

    `object_module_path` is the one thing a backbone overrides: the per-forward
    plumbing, the state-dict defaults and the PEFT ``modules_to_save`` all derive
    from it, so a backbone that mounts the module elsewhere cannot be half-wired.
    """

    #: Dotted path from the policy to the module owning `ObjectConditioning`.
    object_module_path: str = "model"
    object_module_attr: str = "object_conditioning"

    @property
    def object_conditioning(self) -> ObjectConditioning:
        host = resolve_module(self, self.object_module_path)
        module = getattr(host, self.object_module_attr, None)
        active = unwrap_object_conditioning(module)
        if active is None:
            raise ObjectConditioningError(
                f"{self.object_module_path}.{self.object_module_attr} is "
                f"{type(module).__name__}, not ObjectConditioning"
            )
        return active

    @property
    def _object_state_prefix(self) -> str:
        return f"{self.object_module_path}.{self.object_module_attr}."

    def _fresh_state_prefixes(self) -> tuple[str, ...]:
        # A stage-1 checkpoint has no conditioning subtree; everything else in
        # the model must come from the file (VerifiedLoadMixin).
        return (self._object_state_prefix,)

    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """Condition the closed-loop path too, whatever route the backbone takes.

        SmolVLA's `select_action` bypasses `predict_action_chunk`, so wrapping
        only that left its conditioning inert at evaluation while training
        normally -- which reads as "conditioning did not help". Setting the
        inputs twice on the other backbones is harmless.
        """
        self._set_object_inputs(batch)
        try:
            return super().select_action(batch, **kwargs)
        finally:
            self._clear_object_inputs()

    def _set_object_inputs(self, batch: dict[str, Tensor]) -> None:
        tokens = batched(batch.get(ENTITY_TOKENS), unbatched_ndim=2)
        mask = batched(batch.get(ENTITY_MASK), unbatched_ndim=1)
        if tokens is None or mask is None:
            raise ValueError(f"an object-conditioned policy needs {ENTITY_TOKENS} and {ENTITY_MASK}")
        self.object_conditioning.set_inputs(tokens, mask)

    def _clear_object_inputs(self) -> None:
        self.object_conditioning.clear()

    def _prepare_pretrained_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Chain to the backbone's hook if it has one, then keep fresh object weights.

        Probed with `getattr`, because the hook is not universal: `PI05Policy`
        defines it, `SmolVLAPolicy` does not, and an unconditional `super()`
        call is how SmolVLA's object weights once went undefaulted.
        """
        parent = getattr(super(), "_prepare_pretrained_state_dict", None)
        if parent is not None:
            state_dict = parent(state_dict)
        prefix = self._object_state_prefix
        for key, value in self.state_dict().items():
            if key.startswith(prefix):
                state_dict.setdefault(key, value)
        return state_dict

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """The backbone's LoRA targets, plus the conditioning subtree trained in full.

        The subtree is new and randomly initialised, so there is no pretrained
        weight for LoRA to correct; `modules_to_save` is what makes it trainable
        *and* saved. Omitting it trains a run whose object path never leaves its
        zero init.
        """
        parent = getattr(super(), "_get_default_peft_targets", None)
        targets = dict(parent() or {}) if parent is not None else {}
        declared = getattr(self.config, "lora_target_modules", None)
        if declared and not targets.get("target_modules"):
            targets["target_modules"] = declared
        targets["modules_to_save"] = list(targets.get("modules_to_save", [])) + [
            self._object_state_prefix.rstrip(".")
        ]
        return targets
