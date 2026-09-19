#!/usr/bin/env python
"""Derive a single-encoding dataset from the unified one, without copying video.

The unified dataset carries every encoding: the canonical absolute-joint pair
under the names LeRobot expects, and the rest under `alt.` where LeRobot's
feature typing ignores them. Training, though, reads exactly `observation.state`
and `action`, and no LeRobot flag redirects those. So a run gets a *view*: a
dataset directory whose parquet has the requested pair renamed into place and
whose video files are hard links back to the unified copy.

A projection rather than a runtime wrapper, deliberately. A runtime subclass
would have to shadow LeRobotDataset's metadata, stats, feature typing, delta
timestamps and video decoding, and would then only work through an entry point
we control -- `lerobot_train` builds its own dataset from the config. A projected
directory is an ordinary dataset: stock training, stock evaluation, stock
publishing, nothing to keep in sync.

It is nearly free. The bytes that matter are the camera streams, which are
identical across views and shared by inode; a view costs only its parquet, a few
megabytes against 161 MB of video.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oct_vla.data.control_views import (  # noqa: E402
    ALT_PREFIX,
    UNIFIED_CONTROL_SPACE,
    VIEWS,
    view_for,
)


def recompute_env_stats(destination: Path) -> None:
    """Measure environment_state's statistics on the projected column itself.

    The unified dataset stores the tokens as (8, 15) and reduces their stats
    over the slot axis, so it holds 15 numbers describing "a token" rather than
    120 describing this vector. Tiling those eight times would be wrong: the
    slots are not interchangeable -- slot 0's mean position differs from slot
    1's, because the tokens are rank-ordered. The privileged export measured its
    own 120-d column, so this one does too, and the two arms stay comparable.
    """
    import numpy as np
    import pandas as pd

    key = "observation.environment_state"
    rows = [
        np.stack(pd.read_parquet(path, columns=[key])[key].values).astype(np.float64)
        for path in sorted((destination / "data").rglob("*.parquet"))
    ]
    values = np.concatenate(rows, axis=0)
    stats_path = destination / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    stats[key] = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
        **{k: q.tolist() for k, q in zip(("q01", "q10", "q50", "q90", "q99"), quantiles)},
    }
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")
    print(f"  stats    : {key} measured over {len(values)} frames x {values.shape[1]}d")


def link_tree(source: Path, destination: Path) -> int:
    """Hard-link every file under `source` into `destination`.

    Hard links rather than symlinks so the view is an ordinary directory to
    every reader, with no path resolution to get wrong and nothing that breaks
    if the unified dataset is moved within the same filesystem.
    """
    linked = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        os.link(path, target)
        linked += 1
    return linked


#: Columns holding the object-token set in a unified dataset. Only the tokens
#: themselves become `environment_state`; the mask and rank stay behind, because
#: a policy reading environment_state gets one flat vector and has nowhere to put
#: them. They remain available to the ControlVLA plugin, which reads all three.
OBJECT_TOKEN_COLUMN = "observation.object_tokens"
OBJECT_SIDECAR_COLUMNS = ("observation.object_token_mask", "observation.object_token_rank")


def project_frames(source: Path, destination: Path, view, *, objects_as_env=False):
    """Rewrite the parquet with the view's columns promoted and `alt.` dropped."""
    import numpy as np
    import pandas as pd

    frames = written = 0
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pd.read_parquet(path)
        if view.state_column != "observation.state":
            table["observation.state"] = table[view.state_column]
        if view.action_column != "action":
            table["action"] = table[view.action_column]
        if objects_as_env:
            # Flattened, not reshaped at read time: ACT takes environment_state
            # as one vector, and flattening here makes this arm's representation
            # byte-identical to the privileged arm's 120-d column. The two then
            # differ in exactly one thing -- whether the cameras are present --
            # which is the comparison the arm exists to make.
            # np.stack, not np.asarray: parquet round-trips a (8, 15) cell as an
            # object-dtype array of eight 15-element arrays, and asarray on that
            # raises "setting an array element with a sequence" rather than
            # building the matrix.
            table["observation.environment_state"] = [
                np.stack(list(v)).astype(np.float32).reshape(-1)
                for v in table[OBJECT_TOKEN_COLUMN]
            ]
            table = table.drop(columns=[
                c for c in (OBJECT_TOKEN_COLUMN, *OBJECT_SIDECAR_COLUMNS) if c in table.columns
            ])
        # Drop every alternative: a column left behind under `alt.` is harmless
        # to LeRobot but makes the view look like it still offers choices it no
        # longer does, and doubles its parquet for nothing.
        table = table.drop(columns=[c for c in table.columns if c.startswith(ALT_PREFIX)])
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target, index=False)
        frames += len(table)
        written += 1
    return frames, written


def project_meta(source: Path, destination: Path, view, *, objects_as_env=False) -> None:
    """Copy the metadata with features, stats and control space rewritten.

    Stats move with their column. Normalisation is keyed by name, so a view
    whose `action` carried the unified dataset's absolute-joint statistics while
    holding increments would normalise by a scale roughly thirteen times too
    large and silently train on noise.
    """
    shutil.copytree(source / "meta", destination / "meta", dirs_exist_ok=True)

    info = json.loads((destination / "meta" / "info.json").read_text())
    features = info["features"]
    if view.state_column != "observation.state":
        features["observation.state"] = features[view.state_column]
    if view.action_column != "action":
        features["action"] = features[view.action_column]
    if objects_as_env:
        tokens = features[OBJECT_TOKEN_COLUMN]
        count, width = tokens["shape"]
        features["observation.environment_state"] = {
            **tokens,
            "shape": [count * width],
            "names": [f"object{i}.f{j}" for i in range(count) for j in range(width)],
        }
        for key in (OBJECT_TOKEN_COLUMN, *OBJECT_SIDECAR_COLUMNS):
            features.pop(key, None)
    for key in [k for k in features if k.startswith(ALT_PREFIX)]:
        del features[key]
    info["control_space"] = view.control_space
    info["state_encoding"] = view.state_encoding
    info["derived_from"] = source.name
    (destination / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    stats_path = destination / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if view.state_column in stats:
            stats["observation.state"] = stats[view.state_column]
        if view.action_column in stats:
            stats["action"] = stats[view.action_column]
        if objects_as_env:
            # Dropped, not converted: the source stats are per-token and this
            # column is per-slot. recompute_env_stats measures the real thing
            # once the parquet exists.
            for key in (OBJECT_TOKEN_COLUMN, *OBJECT_SIDECAR_COLUMNS):
                stats.pop(key, None)
        for key in [k for k in stats if k.startswith(ALT_PREFIX)]:
            del stats[key]
        stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    for name in ("split_manifest.json", "octvla_episode_manifest.json"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified", required=True, type=Path,
                        help="The unified dataset to derive from.")
    parser.add_argument("--output", required=True, type=Path,
                        help="View directory to create; must not exist.")
    parser.add_argument("--control-space", required=True,
                        choices=sorted({v.control_space for v in VIEWS.values()}))
    parser.add_argument(
        "--objects-as-environment-state", action="store_true",
        help="Promote observation.object_tokens to a flat observation.environment_state, "
        "keeping the cameras. This is the arm the project's question needs and the "
        "exporter cannot produce: its --privileged flag means both 'emit "
        "environment_state' and 'drop the cameras', so no export gives both.",
    )
    parser.add_argument("--state-encoding", default="position",
                        choices=sorted({v.state_encoding for v in VIEWS.values()}))
    args = parser.parse_args()

    view = view_for(args.control_space, args.state_encoding)
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    info = json.loads((args.unified / "meta" / "info.json").read_text())
    if info.get("control_space") != UNIFIED_CONTROL_SPACE:
        raise SystemExit(
            f"{args.unified} is a {info.get('control_space')!r} dataset, not a unified "
            "one; there are no alternative encodings in it to project."
        )

    print(f"view       : {view.control_space} / {view.state_encoding}  ({view.slug})")
    print(f"  state    : {view.state_column} -> observation.state")
    print(f"  action   : {view.action_column} -> action")

    args.output.mkdir(parents=True)
    objects_as_env = args.objects_as_environment_state
    if objects_as_env:
        if OBJECT_TOKEN_COLUMN not in info["features"]:
            raise SystemExit(
                f"{args.unified} has no {OBJECT_TOKEN_COLUMN}; export it with "
                "--object-tokens to get the arm this flag projects."
            )
        print(f"  objects  : {OBJECT_TOKEN_COLUMN} -> observation.environment_state (flat)")
    project_meta(args.unified, args.output, view, objects_as_env=objects_as_env)
    frames, files = project_frames(args.unified, args.output, view, objects_as_env=objects_as_env)
    print(f"  parquet  : {frames} frames in {files} file(s)")
    if objects_as_env:
        recompute_env_stats(args.output)
    for directory in ("videos", "images"):
        if (args.unified / directory).is_dir():
            linked = link_tree(args.unified / directory, args.output / directory)
            print(f"  {directory:8} : {linked} file(s) hard-linked, 0 bytes copied")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
