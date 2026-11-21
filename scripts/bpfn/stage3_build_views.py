#!/usr/bin/env python
"""Stage 3, executor views: a dataset view whose future set holds BimanualPFN pose tokens.

Writes ``<source>_<kind>K`` from an eeabs view: the source parquet plus
``observation.future_keyframe`` ``[K * R, 19]`` and ``observation.future_keyframe_mask``
``[K * R]``. Videos are hard-linked. For every episode, K supports are drawn from *other*
recorded runs of the same rule (never the episode's own parent run):

- ``--kind pose_window``: for every frame t, the frozen PFN's window (R = 2H tokens) from
  the causal query prefix ending at t, one per support;
- ``--kind support_table``: the support's own table (both wrists over time, every object's
  start and end), no PFN: the direct-demonstration baseline.

The executor takes one entry per training sample and entry 0 at evaluation, so training
pairings change across epochs (conditioning/pose_tokens.py). ``info["future_keyframe"]``
records the kind, bank, rows, horizon, PFN run and checkpoint hash, and every episode's
supports. The tokens come from ``oct_vla.bpfn.downstream.views``; this script handles the
LeRobot view (parquet, meta, linked videos, stats).

    stage3_build_views.py --source $V/colour_eeabs --kind pose_window --run RUN \\
        --supports restock_red=ROOT --supports restock_blue=ROOT --bank 4 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _pairs(values):
    out = {}
    for v in values or []:
        rule, path = v.split("=", 1)
        out.setdefault(rule, []).append(Path(path))
    return out


def main() -> int:
    import pandas as pd

    from add_future_keyframe import _stats
    from oct_vla.data.store import read_states
    from oct_vla.policies.conditioning import pose_tokens as P
    from oct_vla.policies.conditioning.future import FUTURE_KEYFRAME, FUTURE_MASK
    from oct_vla.bpfn.data.real import RealData
    from oct_vla.bpfn.downstream.views import SUPPORT_ROWS, PoseTokenBuilder, choose_supports
    from project_dataset_view import link_tree

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--kind", choices=P.POSE_KINDS, required=True)
    ap.add_argument("--run", type=Path, help="PFN run (pose_window)")
    ap.add_argument("--supports", action="append", required=True, help="RULE=ROOT support pool")
    ap.add_argument("--bank", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    source = args.source.resolve()
    suffix = "pw" if args.kind == "pose_window" else "st"
    output = (args.output or source.with_name(f"{source.name}_{suffix}{args.bank}")).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    info = json.loads((source / "meta" / "info.json").read_text())
    if FUTURE_KEYFRAME in info["features"]:
        raise SystemExit(f"{source} already has {FUTURE_KEYFRAME}")
    manifest = json.loads((source / "octvla_episode_manifest.json").read_text())
    by_index = {int(e["lerobot_episode_index"]): e for e in manifest["episodes"]}
    if any("source_path" not in e for e in by_index.values()):
        raise SystemExit(f"{source}: the manifest records no source_path; re-export")
    recordings = RealData(_pairs(args.supports), check=0, cache_size=32)
    pool = recordings.runs
    if args.kind == "pose_window":
        if args.run is None:
            raise SystemExit("--kind pose_window needs --run")
        from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN

        model = PoseWindowPFN.from_run(args.run, args.device)
        horizon = model.horizon
        ckpt = hashlib.sha256((args.run / "model.pt").read_bytes()).hexdigest()[:16]
    else:
        model, horizon, ckpt = None, None, None
    builder = PoseTokenBuilder(args.kind, model, args.device)
    rows = builder.rows

    output.mkdir(parents=True)
    linked = link_tree(source / "videos", output / "videos")
    shutil.copytree(source / "meta", output / "meta")
    for name in ("split_manifest.json", "octvla_episode_manifest.json"):
        shutil.copy2(source / name, output / name)

    supports_used, real_rows, frames = {}, [], 0

    for path in sorted((source / "data").rglob("*.parquet")):
        table = pd.read_parquet(path)
        col_t, col_m = [None] * len(table), [None] * len(table)
        for episode_index, idx in table.groupby("episode_index").groups.items():
            part = table.loc[idx].sort_values("frame_index")
            entry = by_index[int(episode_index)]
            own = Path(entry["source_path"])
            rule = read_states(own).metadata.get("rule")
            chosen = choose_supports(pool.get(rule, []), own, args.bank, args.seed, int(episode_index))
            supports_used[int(episode_index)] = [str(p) for p in chosen]
            query = recordings.episode(own)
            if len(part) != query.T:
                raise SystemExit(f"episode {episode_index}: {len(part)} rows, {query.T} recorded")
            supports = [recordings.episode(p) for p in chosen]
            per_frame = builder.frame_entries(query, supports)
            order = part["frame_index"].to_numpy()
            for position, index in enumerate(part.index):
                toks, mask = P.bank(per_frame[order[position]], rows)
                at = table.index.get_loc(index)
                col_t[at], col_m[at] = toks.tolist(), mask.tolist()
                real_rows.append(toks[mask > 0.5])
            frames += len(part)
            print(f"episode {episode_index} ({rule}): {len(part)} frames, supports "
                  f"{[p.parent.name for p in chosen]}", flush=True)
        table[FUTURE_KEYFRAME] = col_t
        table[FUTURE_MASK] = col_m
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target, index=False)

    info = json.loads((output / "meta" / "info.json").read_text())
    info["features"][FUTURE_KEYFRAME] = {"dtype": "float32", "shape": [args.bank * rows, P.POSE_TOKEN_DIM], "names": None}
    info["features"][FUTURE_MASK] = {"dtype": "float32", "shape": [args.bank * rows], "names": None}
    info["derived_from"] = source.name
    info["future_keyframe"] = {
        "kind": args.kind, "source": "bimanual_pfn" if model is not None else "support_table",
        "bank": args.bank, "rows": rows, "dim": P.POSE_TOKEN_DIM, "horizon": horizon,
        "run": str(args.run) if args.run else None, "checkpoint_sha256_16": ckpt,
        "support_rows": None if model is not None else SUPPORT_ROWS,
        "support_pool": {r: len(v) for r, v in pool.items()}, "seed": args.seed,
        "target_source": "next_measured_pose"}
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    (output / "meta" / "pose_window_supports.json").write_text(json.dumps(supports_used, indent=1) + "\n")
    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats[FUTURE_KEYFRAME] = _stats(np.concatenate(real_rows))
    stats[FUTURE_MASK] = _stats(np.ones((1, 1)))
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    item = LeRobotDataset(f"local/{output.name}", root=output).hf_dataset[0]
    if np.asarray(item[FUTURE_KEYFRAME]).size != args.bank * rows * P.POSE_TOKEN_DIM:
        raise SystemExit(f"{output}: loaded {FUTURE_KEYFRAME} has the wrong size")
    print(f"\nwrote {output}: {len(supports_used)} episodes, {frames} frames, {linked} videos linked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
