#!/usr/bin/env python
"""A dataset view with the paired script oracle's per-arm via-points.

Writes ``<source>_arms``: the source view's parquet plus ``observation.arm_targets``
``[2, VIA, 8]`` and ``observation.arm_targets_mask`` ``[2, VIA]``
(``tasks/shelf_restock/arm_targets.py``), videos hard-linked back to the source. The
source's own columns (entity tokens, future keyframe, ...) are kept, so the same view
serves the KV and the conditional-source arms.

Each episode is cross-checked against the recorded expert's phase labels (used for the
check only, never as input): the share of frames whose oracle stage agrees with the
phase must reach ``MIN_AGREEMENT``, or the episode is refused.

    add_arm_targets.py --source $OCTVLA_DATASET_ROOT/three_object_v4l_eeabs_fkwpr \\
        --collection $OCTVLA_COLLECTION_ROOT/three_object
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ARM_TARGETS = "observation.arm_targets"
ARM_TARGETS_MASK = "observation.arm_targets_mask"
#: Recorded phases each oracle stage may coincide with (stage boundaries lag the
#: phase labels by a few frames, so a neighbour phase counts too).
STAGE_PHASES = {
    "grasp": {"open", "pregrasp", "descend", "close", "lift", "retreat", "park compactor"},
    "carry": {"lift", "preplace", "place"},
    "release": {"place", "release", "retreat"},
    "retreat": {"retreat", "release"},
    "park_mover": {"park mover", "retreat"},
    "compact": {"park mover", "compact close", "compact approach", "compact contact",
                "compact push", "compact retreat", "park compactor"},
    "none": {"park compactor", "retreat", "park mover"},
}
MIN_AGREEMENT = 0.95


def agreement(stages, phases) -> float:
    return float(np.mean([p in STAGE_PHASES[s] for s, p in zip(stages, phases)]))


def main() -> int:
    import pandas as pd

    from add_future_keyframe import _stats
    from oct_vla.data.store import read_episode
    from oct_vla.tasks.shelf_restock.arm_targets import VIA, VIA_DIM, episode_arm_targets
    from oct_vla.tasks.shelf_restock.spec import spec_for_rule
    from project_dataset_view import link_tree

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--collection", required=True, type=Path)
    ap.add_argument("--output", type=Path, help="Default: <source>_arms")
    args = ap.parse_args()
    source = args.source.resolve()
    output = (args.output or source.with_name(source.name + "_arms")).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    manifest = json.loads((source / "octvla_episode_manifest.json").read_text())
    by_index = {int(e["lerobot_episode_index"]): e for e in manifest["episodes"]}
    output.mkdir(parents=True)
    linked = link_tree(source / "videos", output / "videos")
    shutil.copytree(source / "meta", output / "meta")
    for name in ("split_manifest.json", "octvla_episode_manifest.json"):
        shutil.copy2(source / name, output / name)

    real, masks, agree, frames = [], [], [], 0
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pd.read_parquet(path)
        values, mask_col = [None] * len(table), [None] * len(table)
        for episode_index, rows in table.groupby("episode_index").groups.items():
            rows = table.loc[rows].sort_values("frame_index")
            entry = by_index[int(episode_index)]
            episode = read_episode(args.collection / f"seed_{entry['seed']}"
                                   / entry["source_episode"])
            if len(rows) != len(episode.samples):
                raise SystemExit(f"episode {episode_index}: {len(rows)} dataset frames, "
                                 f"{len(episode.samples)} recorded")
            vias, mask, stages = episode_arm_targets(episode,
                                                     spec_for_rule(episode.metadata["rule"]))
            a = agreement(stages, [s.phase for s in episode.samples])
            if a < MIN_AGREEMENT:
                raise SystemExit(f"episode {episode_index}: oracle stages agree with the phase "
                                 f"labels on {a:.3f} of frames (< {MIN_AGREEMENT}); refusing")
            agree.append(a)
            order = rows["frame_index"].to_numpy()
            for position, index in enumerate(rows.index):
                at = table.index.get_loc(index)
                values[at] = vias[order[position]].tolist()
                mask_col[at] = mask[order[position]].astype(np.float32).tolist()
            real.append(vias[mask])
            masks.append(mask.reshape(len(mask), -1).astype(np.float32))
            frames += len(rows)
            counts = {s: stages.count(s) for s in sorted(set(stages))}
            print(f"episode {episode_index}: {len(rows)} frames, agreement {a:.3f}, {counts}",
                  flush=True)
        if any(v is None for v in values):
            raise SystemExit(f"{path}: a row got no arm targets")
        table[ARM_TARGETS] = values
        table[ARM_TARGETS_MASK] = mask_col
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target, index=False)

    info = json.loads((output / "meta" / "info.json").read_text())
    info["features"][ARM_TARGETS] = {"dtype": "float32", "shape": [2, VIA, VIA_DIM],
                                     "names": None}
    info["features"][ARM_TARGETS_MASK] = {"dtype": "float32", "shape": [2, VIA], "names": None}
    info["derived_from"] = source.name
    info["arm_targets"] = {"source": "oracle_arms", "collection": str(args.collection.resolve()),
                           "via": VIA, "dim": VIA_DIM,
                           "phase_agreement_min": float(min(agree)),
                           "phase_agreement_mean": float(np.mean(agree))}
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats[ARM_TARGETS] = _stats(np.concatenate(real))
    stats[ARM_TARGETS_MASK] = _stats(np.concatenate(masks))
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    item = LeRobotDataset(f"local/{output.name}", root=output).hf_dataset[0]
    got = np.asarray(item[ARM_TARGETS]).shape
    if got != (2, VIA, VIA_DIM) and np.asarray(item[ARM_TARGETS]).size != 2 * VIA * VIA_DIM:
        raise SystemExit(f"{output}: loaded {ARM_TARGETS} has shape {got}")
    print(f"\nwrote {output}: {len(agree)} episodes, {frames} frames, phase agreement "
          f"min {min(agree):.3f} mean {np.mean(agree):.3f}, {linked} video files linked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
