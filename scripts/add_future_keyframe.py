#!/usr/bin/env python
"""A dataset view with the oracle next-interaction keyframe beside the entity tokens.

PFN-VLA step 2. Writes ``<source>_fk``: the source view's parquet plus
``observation.future_keyframe`` ``[3, 24]`` and ``observation.future_keyframe_mask``
``[3]`` (``data/future_keyframes.py``), videos hard-linked back to the source,
metadata extended with both features.

Keyframes come from ``tasks/shelf_restock/keyframes.py``, the same oracle the
evaluation server runs, applied to the recorded scenes frame by frame. Each
episode is cross-checked against an independent definition read off the phase
labels (``data/future_keyframes.py``): the event switches must be the same in
number and lag them by 0-10 frames (3-7 over the v4 corpus). Episodes are
matched through ``octvla_episode_manifest.json``; frame alignment is asserted as
in ``add_entity_visual.py``.

    add_future_keyframe.py --source DATASET_VIEW \\
        --collection $OCTVLA_COLLECTION_ROOT/three_object
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


#: Largest accepted lag of the oracle's event switch behind the phase label.
MAX_LAG = 10


def _event_switches(keyframes, mask) -> list[int]:
    """Frames at which the next event's kind changes (grasp, place, or none)."""
    import numpy as np

    from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM
    from oct_vla.data.future_keyframes import FUTURE_ROLES

    start = ENTITY_TOKEN_DIM + len(FUTURE_ROLES)
    kinds = [int(np.argmax(k[0, start:])) if m.all() else -1 for k, m in zip(keyframes, mask)]
    return [i for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1]]


def _switch_lags(keyframes, mask, reference, reference_mask) -> list[int] | None:
    """Lag of each event switch behind the reference's; None if they differ in count."""
    ours, theirs = _event_switches(keyframes, mask), _event_switches(reference, reference_mask)
    if len(ours) != len(theirs):
        return None
    return [a - b for a, b in zip(ours, theirs)]


def _stats(values):
    import numpy as np

    quantiles = {f"q{int(q * 100):02d}": np.quantile(values, q, axis=0).tolist()
                 for q in (0.01, 0.10, 0.50, 0.90, 0.99)}
    return {"min": values.min(0).tolist(), "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
            "count": [int(len(values))], **quantiles}


def main() -> int:
    import numpy as np
    import pandas as pd

    from oct_vla.data.future_keyframes import FUTURE_DIM, FUTURE_ROWS, oracle_future_keyframes
    from oct_vla.tasks.shelf_restock.keyframes import episode_keyframes
    from oct_vla.tasks.shelf_restock.spec import spec_for_rule
    from oct_vla.data.store import read_episode
    from oct_vla.policies.conditioning.future import FUTURE_KEYFRAME, FUTURE_MASK
    from project_dataset_view import link_tree

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Dataset view to extend")
    parser.add_argument("--output", type=Path, help="Default: <source>_fk")
    parser.add_argument("--collection", required=True, type=Path,
                        help="Canonical collection root holding seed_<n>/<source_episode>/")
    args = parser.parse_args()
    source = args.source.resolve()
    output = (args.output or source.with_name(source.name + "_fk")).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    manifest = json.loads((source / "octvla_episode_manifest.json").read_text())
    by_index = {int(e["lerobot_episode_index"]): e for e in manifest["episodes"]}

    output.mkdir(parents=True)
    linked = link_tree(source / "videos", output / "videos")
    shutil.copytree(source / "meta", output / "meta")
    for name in ("split_manifest.json", "octvla_episode_manifest.json"):
        shutil.copy2(source / name, output / name)

    real_rows, masks, episodes_done, frames_done = [], [], 0, 0
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pd.read_parquet(path)
        keyframe_column = [None] * len(table)
        mask_column = [None] * len(table)
        for episode_index, rows in table.groupby("episode_index").groups.items():
            rows = table.loc[rows].sort_values("frame_index")
            entry = by_index[int(episode_index)]
            episode = read_episode(args.collection / f"seed_{entry['seed']}" / entry["source_episode"])
            if len(rows) != len(episode.samples):
                raise SystemExit(f"episode {episode_index}: {len(rows)} dataset frames, "
                                 f"{len(episode.samples)} recorded")
            tokens = np.stack([np.stack(t) for t in rows["observation.entity_tokens"]])
            entity_mask = np.stack([np.asarray(m) for m in rows["observation.entity_mask"]]) > 0.5
            for frame in (0, len(rows) // 2, len(rows) - 1):
                movable = tokens[frame][entity_mask[frame] & (tokens[frame][:, 13] > 0.5)][:, :3]
                scene = np.array([o.pose.position for o in episode.samples[frame].scene.objects])
                if not np.allclose(np.sort(movable, axis=0), np.sort(scene, axis=0), atol=1e-5):
                    raise SystemExit(f"episode {episode_index} frame {frame}: tokens and recorded "
                                     "scene disagree; the frames are not aligned")
            keyframes, mask = episode_keyframes(episode, spec_for_rule(episode.metadata["rule"]))
            lags = _switch_lags(keyframes, mask, *oracle_future_keyframes(episode))
            if lags is None or not all(0 <= lag <= MAX_LAG for lag in lags):
                raise SystemExit(f"episode {episode_index}: oracle events disagree with the phase "
                                 f"labels (lags {lags}); refusing to export it")
            keyframes = keyframes[rows["frame_index"].to_numpy()]
            mask = mask[rows["frame_index"].to_numpy()]
            for position, index in enumerate(rows.index):
                at = table.index.get_loc(index)
                keyframe_column[at] = keyframes[position].tolist()
                mask_column[at] = mask[position].astype(np.float32).tolist()
            real_rows.append(keyframes[mask])
            masks.append(mask.astype(np.float32))
            episodes_done += 1
            frames_done += len(rows)
            print(f"episode {episode_index} ({entry['source_episode']}): {len(rows)} frames, "
                  f"{len(lags)} events (lag {min(lags)}-{max(lags)}), keyframe on {int(mask.all(1).sum())} frames", flush=True)
        if any(value is None for value in keyframe_column):
            raise SystemExit(f"{path}: a row got no keyframe")
        table[FUTURE_KEYFRAME] = keyframe_column
        table[FUTURE_MASK] = mask_column
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target, index=False)

    real = np.concatenate(real_rows)
    if len(real) == 0:
        raise SystemExit("no frame has a keyframe: the phase labels are wrong")
    info = json.loads((output / "meta" / "info.json").read_text())
    info["features"][FUTURE_KEYFRAME] = {"dtype": "float32", "shape": [FUTURE_ROWS, FUTURE_DIM],
                                         "names": None}
    info["features"][FUTURE_MASK] = {"dtype": "float32", "shape": [FUTURE_ROWS], "names": None}
    info["derived_from"] = source.name
    info["future_keyframe"] = {"source": "oracle", "collection": str(args.collection.resolve()),
                               "rows": FUTURE_ROWS, "dim": FUTURE_DIM}
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats[FUTURE_KEYFRAME] = _stats(real)
    stats[FUTURE_MASK] = _stats(np.concatenate(masks))
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")
    print(f"\nwrote {output}: {episodes_done} episodes, {frames_done} frames, "
          f"{len(real)} keyframe rows, {linked} video files linked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
