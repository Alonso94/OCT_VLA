"""Prove that a stage-2 run starts from its stage-1 weights, not from anything else.

The two-stage recipe (ControlVLA: train RGB first, then add object conditioning
from that checkpoint) is only the recipe if the checkpoint actually loads. The
loaders this project goes through fail quietly in three different ways:

* **pi0.5** catches its own load errors. If ``model.safetensors`` cannot be
  read it prints "Returning model without loading pretrained weights"; if the
  state dict cannot be applied it prints a warning. Either way it returns a
  model, and a stage-2 run then fine-tunes the *base* VLA while calling itself
  stage 2 -- a comparison nothing downstream could see was void.
* **LeRobot's generic loader** (SmolVLA, ACT) loads with ``strict=False`` and
  only logs missing keys.
* **GR00T** loads with ``strict=True``, which is the opposite problem: the new
  object keys are rightly absent from a stage-1 file, so a correct stage-2 load
  raises.

So loading is non-strict everywhere, and then *every tensor in the file is
compared against the model it was loaded into*. Keys the file lacks are allowed
only under the prefixes a policy declares fresh -- its object-conditioning
subtree -- or when they share storage with a loaded tensor (tied weights, which
safetensors writes once).

**Slim checkpoints** (``SlimCheckpointMixin``) leave out parameters that are
frozen and therefore identical to the public base model the policy rebuilds on
construction. GR00T is the case: its 6.1 GB backbone is frozen and was measured
bit-identical to ``nvidia/GR00T-N1.7-3B`` in a trained checkpoint, while a full
save with optimiser state cost ~49 GB on disk per run. A marker file names what
was left out, and the verified load accepts exactly those prefixes as rebuilt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

MODEL_FILE = "model.safetensors"
#: Written beside a slim checkpoint's weights: which prefixes were left out.
SLIM_MARKER = "slim_checkpoint.json"


class StageLoadError(RuntimeError):
    """The model does not hold the weights its checkpoint file says it should."""


def local_model_file(path: str | os.PathLike | None) -> Path | None:
    """The full-weights file in a local checkpoint directory, if there is one.

    A hub id or an adapter-only directory returns None: a hub base model goes
    through backbone-specific key remapping, and an adapter directory holds no
    base weights to compare, so neither is what this check is for.
    """
    if path is None:
        return None
    candidate = Path(path) / MODEL_FILE
    return candidate if candidate.is_file() else None


def _matches(stored: torch.Tensor, loaded: torch.Tensor) -> bool:
    if stored.shape != loaded.shape:
        return False
    loaded = loaded.detach().to("cpu")
    if stored.dtype == loaded.dtype:
        return torch.equal(stored, loaded)
    # A backbone may keep some parameters in a different precision than it
    # saved (pi0.5 upcasts selected ones). Compare at the coarser precision.
    coarse = stored.dtype if stored.element_size() <= loaded.element_size() else loaded.dtype
    return torch.equal(stored.to(coarse), loaded.to(coarse))


def verify_loaded_weights(
    model: torch.nn.Module, model_file: str | os.PathLike, *, fresh_prefixes: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Raise StageLoadError unless ``model`` holds exactly ``model_file``'s tensors."""
    from safetensors import safe_open

    state = model.state_dict()
    absent, mismatched, covered_storage = [], [], set()
    with safe_open(str(model_file), framework="pt") as handle:
        keys = list(handle.keys())
        for key in keys:
            if key not in state:
                absent.append(key)
                continue
            if not _matches(handle.get_tensor(key), state[key]):
                mismatched.append(key)
            covered_storage.add(state[key].untyped_storage().data_ptr())
    stored = set(keys)
    fresh, uncovered = [], []
    for key, value in state.items():
        if key in stored:
            continue
        if any(key.startswith(prefix) for prefix in fresh_prefixes):
            fresh.append(key)
        elif value.untyped_storage().data_ptr() not in covered_storage:
            uncovered.append(key)
    if absent or mismatched or uncovered:
        def head(items):
            return ", ".join(items[:5]) + (f" ... (+{len(items) - 5})" if len(items) > 5 else "")

        problems = []
        if mismatched:
            problems.append(f"{len(mismatched)} tensor(s) differ from the file: {head(mismatched)}")
        if absent:
            problems.append(f"{len(absent)} file key(s) not in the model: {head(absent)}")
        if uncovered:
            problems.append(
                f"{len(uncovered)} model key(s) the file does not supply: {head(uncovered)}"
            )
        raise StageLoadError(
            f"{type(model).__name__} did not load {model_file}: " + "; ".join(problems)
        )
    return {"file": str(model_file), "tensors": len(keys), "fresh": len(fresh)}


class VerifiedLoadMixin:
    """Put before the LeRobot policy class: every local full-weights load is checked."""

    def _fresh_state_prefixes(self) -> tuple[str, ...]:
        """State-dict prefixes a checkpoint may lack because they are new here."""
        return ()

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        # Non-strict always; verify_loaded_weights is the strict check, and it
        # knows which missing keys are legitimately new. GR00T passes True.
        del strict
        from lerobot.policies.pretrained import (
            load_model_as_safetensor,
            resolve_safetensors_device,
        )

        load_model_as_safetensor(
            model, model_file, strict=False, device=resolve_safetensors_device(map_location)
        )
        return model

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args: Any, **kwargs: Any):
        policy = super().from_pretrained(pretrained_name_or_path, *args, **kwargs)
        model_file = local_model_file(pretrained_name_or_path)
        if model_file is not None:
            allowed = policy._fresh_state_prefixes() + rebuilt_prefixes(model_file.parent)
            report = verify_loaded_weights(policy, model_file, fresh_prefixes=allowed)
            print(
                f"verified load: {report['tensors']} tensors from {report['file']} match; "
                f"{report['fresh']} tensor(s) fresh or rebuilt under {allowed}",
                flush=True,
            )
        return policy


def rebuilt_prefixes(directory: str | os.PathLike) -> tuple[str, ...]:
    """Prefixes a slim checkpoint in ``directory`` left for the base to supply."""
    marker = Path(directory) / SLIM_MARKER
    if not marker.is_file():
        return ()
    return tuple(json.loads(marker.read_text())["rebuilt_prefixes"])


class SlimCheckpointMixin:
    """Save everything except frozen parameters the base model rebuilds.

    ``rebuilt_prefixes`` must name parameters that are both frozen *and*
    reconstructed identically at construction. Saving refuses if any parameter
    under them is trainable, so a config change that unfreezes one (GR00T's
    ``tune_llm``, ``tune_top_llm_layers``) cannot silently drop learned weights.
    """

    rebuilt_prefixes: tuple[str, ...] = ()

    def _save_pretrained(self, save_directory) -> None:
        from huggingface_hub import save_torch_state_dict
        from lerobot.policies.pretrained import _SINGLE_FILE_SHARD_SIZE

        prefixes = self.rebuilt_prefixes
        if not prefixes:
            return super()._save_pretrained(save_directory)
        trainable = [
            name for name, param in self.named_parameters()
            if param.requires_grad and name.startswith(prefixes)
        ]
        if trainable:
            raise StageLoadError(
                f"refusing a slim save: {len(trainable)} trainable parameter(s) under "
                f"{prefixes}, e.g. {trainable[:3]}; they would not be saved"
            )
        state = self.state_dict()
        kept = {k: v for k, v in state.items() if not k.startswith(prefixes)}
        self.config._save_pretrained(save_directory)
        save_torch_state_dict(kept, str(save_directory), max_shard_size=_SINGLE_FILE_SHARD_SIZE)
        (Path(save_directory) / SLIM_MARKER).write_text(json.dumps({
            "rebuilt_prefixes": list(prefixes),
            "omitted_tensors": len(state) - len(kept),
            "rebuilt_from": str(getattr(self.config, "base_model_path", "")),
        }, indent=2) + "\n")
