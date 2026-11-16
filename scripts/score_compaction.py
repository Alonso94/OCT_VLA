#!/usr/bin/env python
"""Does the robot still compact? Compaction pushes in rollouts, against the demonstrations.

Contract A advances the effect to the next object at let-go, before the compaction push.
So the push is the policy's own motor routine, not something Z asks for, and a branch
that reads Z could make the compactor skip it. This counts it, from object positions,
both arms' positions and the measured apertures only, with the same timeline
(``oct_vla.data.effects.object_timeline``) the effect labels use:

- ``placed``: objects let go on the upper shelf;
- ``compactable``: placed objects after the first (the first has no neighbour);
- ``compacted``: of those, pushed by the compactor (right arm) before the next lift,
  by at least ``MIN_SHIFT`` (the object's own push, as contract B defines it);
- ``shift_cm``: how far each such push moved its object.

Sources: rollout traces (``--trace-dir`` .npz from ``eval_shelf_restock_policy.py``) or,
with ``--demos``, the recorded episodes behind a dataset view (its provenance manifest).

    score_compaction.py --traces $OCTVLA_OUTPUT_ROOT/traces/GEF-kv-s1000-seen-leftmost
    score_compaction.py --demos $OCTVLA_DATASET_ROOT/three_object_v4lr_eeabs_effA
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MIN_SHIFT = 0.01  # m: a push that moves its object less is not compaction


def compaction(pos: np.ndarray, eef: np.ndarray, aperture: np.ndarray,
               upper: np.ndarray) -> dict:
    """``pos`` [T, N, 3], ``eef`` [T, 2, 3], ``aperture`` [T, 2], ``upper`` [T, N]."""
    from oct_vla.data.effects import object_timeline

    lines = []
    for j in range(pos.shape[1]):
        if np.isnan(pos[:, j]).any():
            continue
        line = object_timeline(pos[:, j], eef, aperture, upper[:, j])
        if line.get("lift") is not None:
            lines.append((j, line))
    lines.sort(key=lambda jl: jl[1]["lift"])
    placed = [(i, j, l) for i, (j, l) in enumerate(lines) if l.get("let_go") is not None]
    out = {"lifted": len(lines), "placed": len(placed), "compactable": 0, "compacted": 0,
           "shift_cm": [], "first_pushed": 0}
    for n, (i, j, line) in enumerate(placed):
        next_lift = lines[i + 1][1]["lift"] if i + 1 < len(lines) else None
        own = (line.get("push_start") is not None and line.get("pusher") == "right"
               and (next_lift is None or line["push_start"] < next_lift))
        shift = (float(np.linalg.norm(pos[line["push_end"], j]
                                      - pos[max(line["push_start"] - 1, 0), j]))
                 if own else 0.0)
        if n == 0:
            out["first_pushed"] += own and shift >= MIN_SHIFT
            continue
        out["compactable"] += 1
        if own and shift >= MIN_SHIFT:
            out["compacted"] += 1
            out["shift_cm"].append(100 * shift)
    return out


def summarise(per: dict[str, dict]) -> dict:
    tot = {k: sum(r[k] for r in per.values())
           for k in ("lifted", "placed", "compactable", "compacted", "first_pushed")}
    shifts = [s for r in per.values() for s in r["shift_cm"]]
    tot["episodes"] = len(per)
    tot["compaction_rate"] = tot["compacted"] / tot["compactable"] if tot["compactable"] else None
    tot["shift_cm_median"] = float(np.median(shifts)) if shifts else None
    return tot


def _upper(pos, spec):
    return np.array([[not np.isnan(x).any() and spec.upper_shelf.contains(tuple(x))
                      for x in row] for row in pos])


def from_traces(trace_dir: Path, spec) -> dict[str, dict]:
    per = {}
    for path in sorted(trace_dir.glob("*.npz")):
        with np.load(path, allow_pickle=True) as z:
            pos, eef, ap = (np.asarray(z[k], float) for k in ("objects", "eef", "gripper"))
        per[path.stem] = compaction(pos, eef, ap, _upper(pos, spec))
    if not per:
        raise SystemExit(f"no traces in {trace_dir}")
    return per


def from_demos(view: Path, spec) -> dict[str, dict]:
    from oct_vla.data.effects import episode_tracks
    from oct_vla.data.store import read_episode

    manifest = json.loads((view / "octvla_episode_manifest.json").read_text())
    per = {}
    for entry in sorted(manifest["episodes"], key=lambda e: int(e["lerobot_episode_index"])):
        episode = read_episode(entry["source_path"])
        tracks = episode_tracks(episode, spec)
        rule = episode.metadata.get("rule", "?")
        per[f"{rule}_seed{episode.seed}"] = compaction(tracks.pos, tracks.eef,
                                                       tracks.aperture, tracks.upper)
        del episode, tracks
        gc.collect()
    return per


def main() -> int:
    from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC

    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--traces", type=Path, nargs="+")
    src.add_argument("--demos", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.output is not None and args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    results = {}
    if args.demos is not None:
        per = from_demos(args.demos, DEFAULT_SPEC)
        for rule in sorted({k.split("_seed")[0] for k in per}):
            results[f"demos:{rule}"] = summarise({k: v for k, v in per.items()
                                                  if k.startswith(rule + "_")})
    else:
        for d in args.traces:
            results[d.name] = summarise(from_traces(d, DEFAULT_SPEC))
    print(f"{'source':44s} {'eps':>4s} {'placed':>7s} {'compactable':>12s} {'compacted':>10s} "
          f"{'rate':>5s} {'shift cm':>9s} {'1st pushed':>10s}")
    for name, r in results.items():
        rate = "-" if r["compaction_rate"] is None else f"{r['compaction_rate']:.2f}"
        shift = "-" if r["shift_cm_median"] is None else f"{r['shift_cm_median']:.1f}"
        print(f"{name:44s} {r['episodes']:4d} {r['placed']:7d} {r['compactable']:12d} "
              f"{r['compacted']:10d} {rate:>5s} {shift:>9s} {r['first_pushed']:10d}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
