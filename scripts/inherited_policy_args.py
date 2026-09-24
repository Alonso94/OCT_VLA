#!/usr/bin/env python
"""The stage-1 config fields a conditioned arm must inherit, as --policy flags.

The stock arm loads with --policy.path and takes its whole config from the
checkpoint. A conditioned arm cannot -- its type is a plugin -- so it is built
from --policy.type, which starts from the config class *defaults*, and only the
weights come from --policy.pretrained_path. Any field the checkpoint set away
from its default is then silently reset.

That happened. lerobot/smolvla_base sets load_vlm_weights=true, which builds
the VLM in bfloat16; the default false builds it in float32. Every conditioned
SmolVLA arm ran its backbone at a different precision from `rgb`/`rgb_cont`,
and only the stage-2 init check noticed (action chunk 1.9 % off at step 0).

Prints one flag per line for every field the target config shares with the
checkpoint's config.json whose value differs from the target's default,
except the ones the caller sets itself.

    inherited_policy_args.py <checkpoint dir or hub id> <policy type>
"""

from __future__ import annotations

import dataclasses
import enum
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Set explicitly by the training script, or identity rather than behaviour.
CALLER_OWNED = {
    "type", "pretrained_path", "input_features", "output_features", "device",
    "n_action_steps", "push_to_hub", "repo_id", "use_amp", "dtype",
    "gradient_checkpointing", "discover_packages_path",
}


def _plain(value):
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {str(_plain(k)): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    return value


def _flag(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def inherited(source: dict, config_class) -> list[str]:
    defaults = config_class()
    flags = []
    for field in dataclasses.fields(defaults):
        name = field.name
        if name in CALLER_OWNED or name.startswith("object_") or name not in source:
            continue
        default = _plain(getattr(defaults, name))
        if source[name] != default and source[name] is not None:
            flags.append(f"--policy.{name}={_flag(source[name])}")
    return flags


def load_config(where: str) -> dict:
    path = Path(where)
    if path.is_dir():
        return json.loads((path / "config.json").read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(where, "config.json")).read_text())


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    from lerobot.configs.policies import PreTrainedConfig

    import oct_vla.policies  # noqa: F401 - registers the plugin types

    source = load_config(sys.argv[1])
    config_class = PreTrainedConfig.get_choice_class(sys.argv[2])
    for flag in inherited(source, config_class):
        print(flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
