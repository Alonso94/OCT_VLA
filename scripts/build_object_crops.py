#!/usr/bin/env python
"""Cut per-object 64x64 head-camera crops from the canonical full-run episodes.

The training data for the LeJEPA object encoder (oct_vla.perception.lejepa).
Only one split's seeds are read, taken from a dataset's split_manifest.json,
so the encoder never sees a validation or identity-holdout episode -- the same
split the policies train on, which is what lets its embeddings be used by a
policy later without leaking evaluation scenes into the representation.

Output, per split, under `<out>/<split>/`:

* `crops.npy`   -- (N, 64, 64, 3) uint8, one row per (frame, object);
* `index.npz`   -- per-row `episode`, `seed`, `frame`, `track`, `model_id`,
  `size_xyz`, `square` (x0, y0, side in source pixels) and `eef_distance`
  (metres from the object centre to the nearer end effector);
* `meta.json`   -- camera K and extrinsic, parameters, counts, episode list.

Rows of one (episode, track) are in frame order, which is how training finds
the same object a few steps away. CPU-only; one worker per episode.

    PYTHONPATH=src $OCTVLA_POLICY_PYTHON scripts/build_object_crops.py --split train
    PYTHONPATH=src $OCTVLA_POLICY_PYTHON scripts/build_object_crops.py \\
        --overlay-dir /tmp/overlay --overlay-seeds 100 204
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from oct_vla.perception.lejepa.crops import (
    CROP_SIZE,
    DEFAULT_PADDING,
    MIN_SIDE_PX,
    MIN_VISIBLE_FRACTION,
    crop_object,
    default_head_camera,
    draw_overlay,
    head_camera_from_robotwin,
    model_id_by_size,
    read_head_episode,
)

DEFAULT_COLLECTION = "/home/vault/g107ea/g107ea12/octvla-collection-v3paired/three_object"
DEFAULT_DATASET = "three_object_paired_full_eeabs"
ASSET = "113_coffee-box"


def _episode_dir(collection: Path, seed: int) -> Path:
    # Only the full run: atomic clips are cut from it, so reading them too
    # would duplicate every frame under another name.
    return collection / f"seed_{seed}" / f"episode_{seed:04d}_full"


def _track_index(track_id: str) -> int:
    prefix, _, number = track_id.partition("_")
    if prefix != "obj" or not number.isdigit():
        raise ValueError(f"unexpected track_id {track_id!r}")
    return int(number)


def _crop_episode(job: tuple) -> dict:
    episode_index, directory, stride, sizes_to_id = job
    camera = default_head_camera()
    episode = read_head_episode(directory)
    if episode.frames.shape[1:3] != (camera.height, camera.width):
        raise ValueError(f"{directory}: head frames {episode.frames.shape} do not match the camera")
    spawned = {int(v) for v in episode.metadata.get("model_ids", "").split(",") if v}
    rows: dict[str, list] = {k: [] for k in ("frame", "track", "model_id", "size", "square", "eef")}
    crops = []
    skipped = 0
    for frame in range(0, len(episode.frames), stride):
        image = episode.frames[frame]
        for obj in episode.scenes[frame].objects:
            result = crop_object(image, camera, obj)
            if result is None:
                skipped += 1
                continue
            crop, square = result
            key = tuple(round(v, 4) for v in obj.size_xyz)
            model_id = sizes_to_id.get(key, -1)
            # The id recovered from size must be one the episode says it spawned;
            # otherwise the size table is for another asset and every label wrong.
            if spawned and model_id not in spawned:
                raise ValueError(
                    f"{directory}: size {key} maps to model {model_id}, not in spawned {spawned}"
                )
            crops.append(crop)
            rows["frame"].append(frame)
            rows["track"].append(_track_index(obj.track_id))
            rows["model_id"].append(model_id)
            rows["size"].append(obj.size_xyz)
            rows["square"].append(square)
            centre = np.asarray(obj.pose.position)
            rows["eef"].append(
                float(np.linalg.norm(episode.eef_positions[frame] - centre, axis=1).min())
            )
    count = len(crops)
    return {
        "episode": episode_index,
        "seed": episode.seed,
        "directory": str(directory),
        "frames": len(episode.frames),
        "skipped": skipped,
        "crops": np.stack(crops) if crops else np.zeros((0, CROP_SIZE, CROP_SIZE, 3), np.uint8),
        "frame": np.asarray(rows["frame"], np.int32).reshape(count),
        "track": np.asarray(rows["track"], np.int16).reshape(count),
        "model_id": np.asarray(rows["model_id"], np.int16).reshape(count),
        "size_xyz": np.asarray(rows["size"], np.float32).reshape(count, 3),
        "square": np.asarray(rows["square"], np.int16).reshape(count, 3),
        "eef_distance": np.asarray(rows["eef"], np.float32).reshape(count),
    }


def _write_overlays(collection: Path, seeds: list[int], frames: list[int], out: Path) -> None:
    import cv2

    camera = default_head_camera()
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        episode = read_head_episode(_episode_dir(collection, seed))
        for frame in frames:
            frame = min(frame, len(episode.frames) - 1)
            image = draw_overlay(episode.frames[frame], camera, episode.scenes[frame])
            image = cv2.resize(
                image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_NEAREST
            )
            path = out / f"seed{seed}_f{frame:03d}.png"
            cv2.imwrite(str(path), image[:, :, ::-1])
            print(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--collection", type=Path, default=Path(DEFAULT_COLLECTION))
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="dataset under $OCTVLA_DATASET_ROOT whose split_manifest.json is used",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="default: $HPCVAULT/octvla-lejepa/crops_v1"
    )
    parser.add_argument("--stride", type=int, default=3, help="keep every k-th recorded frame")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="only draw projected boxes for --overlay-seeds and exit",
    )
    parser.add_argument("--overlay-seeds", type=int, nargs="*", default=[100, 204])
    parser.add_argument("--overlay-frames", type=int, nargs="*", default=[0, 150, 300])
    args = parser.parse_args(argv)

    robotwin_root = os.environ.get("OCTVLA_ROBOTWIN_ROOT")
    if not robotwin_root:
        print("OCTVLA_ROBOTWIN_ROOT is required (source ~/octvla/env.sh)", file=sys.stderr)
        return 2
    # Refuses to go on if RoboTwin's own config disagrees with the pinned camera.
    camera = head_camera_from_robotwin(robotwin_root)

    if args.overlay_dir is not None:
        _write_overlays(args.collection, args.overlay_seeds, args.overlay_frames, args.overlay_dir)
        return 0

    manifest_path = Path(os.environ["OCTVLA_DATASET_ROOT"]) / args.dataset / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    seeds = [int(s) for s in manifest[args.split]["seeds"]]
    if len(seeds) != int(manifest[args.split]["count"]):
        raise ValueError(
            f"{manifest_path}: {args.split} lists {len(seeds)} seeds, count says otherwise"
        )
    held_out = set(manifest.get("identity_holdout", {}).get("seeds", []))
    if held_out & set(seeds):
        raise ValueError(f"identity-holdout seeds in {args.split}: {sorted(held_out & set(seeds))}")
    directories = [_episode_dir(args.collection, seed) for seed in seeds]
    missing = [str(d) for d in directories if not (d / "episode.json").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} episodes missing, e.g. {missing[0]}")

    out_root = args.out or Path(os.environ["HPCVAULT"]) / "octvla-lejepa" / "crops_v1"
    out = out_root / args.split
    if (out / "meta.json").exists():
        print(f"{out} already holds a crop build; remove it first", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    sizes_to_id = model_id_by_size(ASSET, Path(robotwin_root) / "assets")
    jobs = [(i, d, args.stride, sizes_to_id) for i, d in enumerate(directories)]
    start = time.time()
    with Pool(min(args.workers, len(jobs))) as pool:
        parts = sorted(pool.imap_unordered(_crop_episode, jobs), key=lambda p: p["episode"])
    total = sum(len(p["crops"]) for p in parts)
    if total == 0:
        raise RuntimeError("no crops were produced")

    store = np.lib.format.open_memmap(
        out / "crops.npy", mode="w+", dtype=np.uint8, shape=(total, CROP_SIZE, CROP_SIZE, 3)
    )
    offset = 0
    for part in parts:
        store[offset : offset + len(part["crops"])] = part["crops"]
        offset += len(part["crops"])
    store.flush()
    del store

    def cat(key):
        return np.concatenate([p[key] for p in parts])

    index = {
        "episode": np.concatenate(
            [np.full(len(p["crops"]), p["episode"], np.int32) for p in parts]
        ),
        "seed": np.concatenate([np.full(len(p["crops"]), p["seed"], np.int32) for p in parts]),
        **{
            key: cat(key)
            for key in ("frame", "track", "model_id", "size_xyz", "square", "eef_distance")
        },
    }
    np.savez(out / "index.npz", **index)

    unknown = int((index["model_id"] < 0).sum())
    ids, id_counts = np.unique(index["model_id"], return_counts=True)
    meta = {
        "split": args.split,
        "dataset": args.dataset,
        "collection": str(args.collection),
        "stride": args.stride,
        "crop_size": CROP_SIZE,
        "padding": DEFAULT_PADDING,
        "min_side_px": MIN_SIDE_PX,
        "min_visible_fraction": MIN_VISIBLE_FRACTION,
        "camera": {
            "intrinsic_cv": camera.intrinsic.tolist(),
            "extrinsic_cv": camera.extrinsic.tolist(),
            "width": camera.width,
            "height": camera.height,
        },
        "episodes": [
            {
                "seed": p["seed"],
                "directory": p["directory"],
                "frames": p["frames"],
                "crops": len(p["crops"]),
                "skipped": p["skipped"],
            }
            for p in parts
        ],
        "count": total,
        "skipped": sum(p["skipped"] for p in parts),
        "model_id_counts": {int(i): int(c) for i, c in zip(ids, id_counts, strict=True)},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    size_mb = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"split       : {args.split} ({len(seeds)} episodes)")
    print(f"crops       : {total}  (skipped off-image/small: {meta['skipped']})")
    print(f"model ids   : {meta['model_id_counts']}  (unknown: {unknown})")
    print(f"disk        : {size_mb:.1f} MB at {out}")
    print(f"elapsed     : {time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
