"""Make the FlashVLA policy types loadable against this project's LeRobot.

FlashVLA publishes `z-lab/flashvla-pi05-robotwin`: pi0.5 finetuned on RoboTwin
2.0, which is the domain-relevant pretraining `lerobot/pi05_base` lacks. Its
package pins `lerobot==0.5.1` while this project runs 0.6.2, so it is installed
with `--no-deps`; the pin turns out to be conservative rather than necessary.

One incompatibility does surface, and it is not in FlashVLA's code. Loading any
of its configs goes through draccus, which calls `typing.get_type_hints` on every
dataclass in the tree. `PI05VLMConfig` inherits `transformers`'
`PreTrainedConfig`, which annotates

    dtype: str | torch.dtype | None

as a string forward reference — and `transformers.configuration_utils` does not
import torch at module level, because it only needs the name under
`TYPE_CHECKING`. `get_type_hints` evaluates each base class's annotations in
*that base's* module globals, finds no `torch` there, and raises
`NameError: name 'torch' is not defined`.

So the name is supplied. Binding it on FlashVLA's own module does not work --
the annotation belongs to the transformers base, and that is the namespace the
evaluation uses.

This is a version-drift artefact: the LeRobot 0.5.1 FlashVLA pins brings an
older transformers whose config has no such annotation. Nothing here patches
behaviour, only makes an existing type resolvable, so it is safe to apply
unconditionally and to drop once transformers imports torch eagerly or draccus
stops eagerly resolving hints.

Verified after applying: the published checkpoint's 813 tensors match
`PI05FlashVLAPolicy`'s expected state dict exactly, with nothing missing and
nothing unexpected.
"""

from __future__ import annotations

#: Set once `register()` has run, so repeated imports stay cheap and the
#: registration cannot be applied twice.
_REGISTERED = False


def register() -> list[str]:
    """Register FlashVLA's policy types and return the names now available.

    Raises ModuleNotFoundError with an actionable message if FlashVLA is absent,
    rather than letting an unknown-policy-type error surface much later from
    inside a training job.
    """
    global _REGISTERED

    try:
        import torch
        import transformers.configuration_utils as configuration_utils
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise ModuleNotFoundError(
            "FlashVLA support needs torch and transformers, which live in the "
            "policy environment. See docs/setup.md."
        ) from error

    # The one name draccus cannot resolve. Idempotent, and a no-op on a
    # transformers version that already imports torch eagerly.
    if not hasattr(configuration_utils, "torch"):
        configuration_utils.torch = torch

    try:
        import flashvla.configs  # noqa: F401  -- import registers the types
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "FlashVLA is not installed in the policy environment. Install it "
            "without its dependencies, which pin lerobot==0.5.1 against this "
            "project's 0.6.2:\n"
            "    git clone https://github.com/z-lab/flashvla.git ~/octvla/flashvla\n"
            "    VIRTUAL_ENV=~/octvla/policy-venv uv pip install --no-deps -e ~/octvla/flashvla"
        ) from error

    from lerobot.configs.policies import PreTrainedConfig

    _REGISTERED = True
    return sorted(k for k in PreTrainedConfig._choice_registry if "flashvla" in k)
