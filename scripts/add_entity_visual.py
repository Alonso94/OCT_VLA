#!/usr/bin/env python
"""A dataset view with per-entity LeJEPA features beside the entity tokens.

Plan Phase E, step 3. Writes `<source>_vis`: the source view's parquet with an
`observation.entity_visual` column `[max_entities, visual_dim]` added, its
videos hard-linked back to the source (a view costs only its parquet), and its
metadata extended with the new feature.

Features come from the **canonical raw head frames** (the dataset's AV1 video is
not what evaluation sees), matched to dataset episodes through
`octvla_episode_manifest.json`, and from the dataset's own entity tokens -- the
same inputs, through the same function (`perception/lejepa/entity_visual.py`),
that live evaluation uses. The frame alignment is asserted per episode: the
movable token positions must equal the recorded scene's object positions.

    add_entity_visual.py --source DATASET_VIEW --lejepa RUN_DIR \\
        --collection $HPCVAULT/octvla-collection-v3paired/three_object [--device cuda]
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


def main() -> int:
    import numpy as np
    import pandas as pd

    from oct_vla.perception.lejepa.crops import read_head_episode
    from oct_vla.perception.lejepa.entity_visual import ENTITY_VISUAL, EntityVisualEncoder
    from project_dataset_view import link_tree

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Dataset view to extend")
    parser.add_argument("--output", type=Path, help="Default: <source>_vis")
    parser.add_argument("--lejepa", required=True, type=Path, help="LeJEPA run directory")
    parser.add_argument("--collection", required=True, type=Path,
                        help="Canonical collection root holding seed_<n>/<source_episode>/")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-episodes", type=int,
                        help="Smoke test only: stop after this many episodes (the view is incomplete)")
    args = parser.parse_args()
    source = args.source.resolve()
    output = (args.output or source.with_name(source.name + "_vis")).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")

    encoder = EntityVisualEncoder(args.lejepa, device=args.device)
    manifest = json.loads((source / "octvla_episode_manifest.json").read_text())
    by_index = {int(e["lerobot_episode_index"]): e for e in manifest["episodes"]}

    output.mkdir(parents=True)
    linked = link_tree(source / "videos", output / "videos")
    shutil.copytree(source / "meta", output / "meta")
    for name in ("split_manifest.json", "octvla_episode_manifest.json"):
        shutil.copy2(source / name, output / name)

    real_features = []
    frames_done = 0
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pd.read_parquet(path)
        column = [None] * len(table)
        for episode, rows in table.groupby("episode_index").groups.items():
            if args.max_episodes is not None and len(real_features) >= args.max_episodes:
                break
            rows = table.loc[rows].sort_values("frame_index")
            entry = by_index[int(episode)]
            recorded = read_head_episode(args.collection / f"seed_{entry['seed']}" / entry["source_episode"])
            tokens = np.stack([np.stack(t) for t in rows["observation.entity_tokens"]])
            mask = np.stack([np.asarray(m) for m in rows["observation.entity_mask"]]) > 0.5
            if len(rows) != len(recorded.frames):
                raise SystemExit(f"episode {episode}: {len(rows)} dataset frames, "
                                 f"{len(recorded.frames)} recorded")
            for frame in (0, len(rows) // 2, len(rows) - 1):
                movable = tokens[frame][mask[frame] & (tokens[frame][:, 13] > 0.5)][:, :3]
                scene = np.array([o.pose.position for o in recorded.scenes[frame].objects])
                if not np.allclose(np.sort(movable, axis=0), np.sort(scene, axis=0), atol=1e-5):
                    raise SystemExit(f"episode {episode} frame {frame}: tokens and recorded "
                                     "scene disagree; the frames are not aligned")
            features = encoder(list(recorded.frames[rows["frame_index"].to_numpy()]), tokens, mask)
            nonzero = np.abs(features).sum(-1) > 0
            real_features.append(features[nonzero])
            for position, index in enumerate(rows.index):
                column[table.index.get_loc(index)] = features[position].tolist()
            frames_done += len(rows)
            print(f"episode {episode} ({entry['source_episode']}): {len(rows)} frames, "
                  f"{int(nonzero.sum())} object crops", flush=True)
        if args.max_episodes is not None:
            keep = [i for i, value in enumerate(column) if value is not None]
            table = table.iloc[keep].copy()
            column = [column[i] for i in keep]
        table[ENTITY_VISUAL] = column
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target, index=False)

    real = np.concatenate(real_features)
    if len(real) == 0:
        raise SystemExit("no entity produced a crop: the camera or the tokens are wrong")
    info = json.loads((output / "meta" / "info.json").read_text())
    max_entities = info["features"]["observation.entity_tokens"]["shape"][0]
    info["features"][ENTITY_VISUAL] = {"dtype": "float32",
                                       "shape": [max_entities, encoder.visual_dim], "names": None}
    info["derived_from"] = source.name
    info["entity_visual"] = {"lejepa_run": str(args.lejepa.resolve()),
                             "visual_dim": encoder.visual_dim}
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    quantiles = {f"q{int(q * 100):02d}": np.quantile(real, q, axis=0).tolist()
                 for q in (0.01, 0.10, 0.50, 0.90, 0.99)}
    stats[ENTITY_VISUAL] = {"min": real.min(0).tolist(), "max": real.max(0).tolist(),
                            "mean": real.mean(0).tolist(), "std": real.std(0).tolist(),
                            "count": [int(len(real))], **quantiles}
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")
    print(f"\nwrote {output}: {frames_done} frames, {len(real)} object features "
          f"(dim {encoder.visual_dim}), {linked} video files linked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
