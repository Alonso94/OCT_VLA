#!/usr/bin/env python
"""Push an exported dataset to the Hub, private, with its facts in the card.

Run after every export that is worth keeping. The card is generated from the
dataset's own metadata rather than written by hand, so it cannot drift from what
was actually exported -- the split sizes, the control space, the gripper
encoding and the entity schema are read out of `meta/info.json` and
`split_manifest.json`.

Private by default and refuses `--public` without `--i-understand-this-is-
public`, because a corpus is easy to publish by accident and impossible to
unpublish.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CARD = """---
license: mit
task_categories:
- robotics
tags:
- lerobot
- robotwin
- object-centric
- imitation-learning
---

# {title}

{summary}

## Contents

| | |
| --- | --- |
| episodes | {episodes} |
| control space | `{control_space}` |
| gripper encoding | `{gripper}` |
| entity schema | {entity} |
| cameras | {cameras} |

## Splits

{splits}

Episodes are ordered train-then-validation, so LeRobot's positional
`eval_split` reproduces the reserved seeds exactly. A seed belongs to one split
and cannot migrate after the fact.

## Reading it

Statistics are fitted on the **training partition alone** and travel in the
policy config; values on disk are raw. Do not renormalise with whole-dataset
statistics -- this repo contains both splits.

{extra}

Produced by `{repo}` at `{commit}`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--repo-id", required=True, help="e.g. user/oct-vla-something")
    parser.add_argument("--title", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--extra", default="", help="Extra markdown for the card")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--i-understand-this-is-public", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.public and not args.i_understand_this_is_public:
        raise SystemExit(
            "--public needs --i-understand-this-is-public. A corpus is easy to "
            "publish by accident and impossible to unpublish."
        )
    info_path = args.dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"{args.dataset} has no meta/info.json; is it an exported dataset?")
    info = json.loads(info_path.read_text())
    features = info.get("features", {})

    manifest_path = args.dataset / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    splits = []
    for name in ("train", "val", "identity_holdout"):
        block = manifest.get(name)
        if isinstance(block, dict) and block.get("count"):
            splits.append(f"- **{name}**: {block['count']} episodes")
    if manifest.get("eval_split") is not None:
        splits.append(f"- `eval_split` = {manifest['eval_split']}")

    entity = info.get("entity_tokens") or {}
    card = CARD.format(
        title=args.title or args.dataset.name.replace("_", " "),
        summary=args.summary or "A LeRobot dataset exported from RoboTwin/SAPIEN.",
        episodes=info.get("total_episodes", manifest.get("total_episodes", "?")),
        control_space=info.get("control_space", "?"),
        gripper=info.get("gripper_encoding", "measured_aperture"),
        entity=f"`{entity['schema']}`, {entity['token_dim']}-d, capacity "
        f"{entity.get('capacity', '?')}" if entity else "none",
        cameras=", ".join(f"`{k.split('.')[-1]}`" for k in features if "images" in k) or "none",
        splits="\n".join(splits) or "Recorded in `split_manifest.json`.",
        extra=args.extra,
        repo="OCT_VLA",
        commit=subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip() or "unknown",
    )

    if args.dry_run:
        print(card)
        return 0
    (args.dataset / "README.md").write_text(card)
    command = [
        "hf", "upload", args.repo_id, str(args.dataset), "--type", "dataset",
        "--commit-message", f"Export {args.dataset.name}",
    ]
    if not args.public:
        command.append("--private")
    print(" ".join(command), flush=True)
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
