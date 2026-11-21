#!/usr/bin/env python
"""Calibrate the synthetic prior against recorded runs (plan 0.4); writes calibration.json.

- **Tool frame** (shelf grasps): at each grasp onset (the mover's aperture starts closing
  next to the object, from data/effects timelines), the grasped object's centre in the
  EEF frame, and the EEF's local axes in the world. The generator's TOOL_AXIS /
  TOOL_OFFSET must match.
- **Phase durations** (seconds, from observation-only timelines): approach (previous
  let-go -> grasp), lift (grasp -> lift), carry (lift -> place), release (place ->
  let-go), compaction push (push start -> end); median and IQR. The same quantities are
  measured on generated episodes for comparison.

    stage1_calibrate_tool.py --runs ROOT [--limit 40] --out configs/bimanual_pfn/calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PHASES = ("approach", "lift", "carry", "release", "push")


def phase_durations(ep, lines) -> dict[str, list[float]]:
    dt = float(np.median(np.diff(ep.time)))
    out = {k: [] for k in PHASES}
    order = sorted((v for v in lines.values() if v.get("lift") is not None), key=lambda v: v["lift"])
    prev = 0
    for v in order:
        if v.get("grasp") is not None:
            out["approach"].append((v["grasp"] - prev) * dt)
            out["lift"].append((v["lift"] - v["grasp"]) * dt)
        if v.get("place") is not None:
            out["carry"].append((v["place"] - v["lift"]) * dt)
        if v.get("let_go") is not None and v.get("place") is not None:
            out["release"].append((v["let_go"] - v["place"]) * dt)
            prev = v["let_go"]
        if v.get("push_start") is not None:
            out["push"].append((v["push_end"] - v["push_start"]) * dt)
    return out


def _lines(ep):
    from oct_vla.data import effects as E
    from oct_vla.bpfn.schema import MOVABLE

    mov = [j for j in range(len(ep.track_ids)) if ep.kind[j] == MOVABLE]
    upper = ep.objects[:, mov, 2] > 0.86
    tracks = E.Tracks(tuple(ep.track_ids[j] for j in mov), ep.objects[:, mov, :3],
                      ep.objects[:, mov, 3:], ep.size[mov], ep.wrist[..., :3], ep.aperture, upper)
    return E.timelines(tracks), mov


def summary(values: list[float]) -> dict:
    v = np.asarray(values, float)
    return {"n": int(len(v)), "median": float(np.median(v)) if len(v) else None,
            "q25": float(np.percentile(v, 25)) if len(v) else None,
            "q75": float(np.percentile(v, 75)) if len(v) else None}


def main() -> int:
    from oct_vla.bpfn.data.syn_data import generator
    from oct_vla.bpfn import quat
    from oct_vla.bpfn.data.real_data.robotwin import read_recording
    from oct_vla.bpfn.data.real import find_runs
    from oct_vla.bpfn.data.syn_data.trajectory import TOOL_AXIS, TOOL_OFFSET

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, action="append", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--synthetic", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")

    local, axes, real = [], [], {k: [] for k in PHASES}
    runs = [p for root in args.runs for p in find_runs(root)][:args.limit]
    for path in runs:
        ep = read_recording(path)
        lines, mov = _lines(ep)
        for tid, v in lines.items():
            if v.get("grasp") is None:
                continue
            t, j = v["grasp"], ep.track_ids.index(tid)
            arm = 0 if v["mover"] == "left" else 1
            w, o = ep.wrist[t, arm], ep.objects[t, j]
            local.append(quat.rotate(quat.inverse(w[3:]), o[:3] - w[:3]))
            axes.append(quat.to_matrix(w[3:]))
        for k, vals in phase_durations(ep, lines).items():
            real[k] += vals
    local, axes = np.array(local), np.array(axes)
    tool = {"n_grasps": int(len(local)),
            "object_in_eef_mean": local.mean(0).round(4).tolist(),
            "object_in_eef_std": local.std(0).round(4).tolist(),
            "eef_local_axes_in_world_mean": {n: axes[:, :, i].mean(0).round(3).tolist()
                                             for i, n in enumerate("xyz")},
            "generator": {"tool_axis": TOOL_AXIS.tolist(), "tool_offset": TOOL_OFFSET}}
    tool["generator_consistent"] = bool(
        np.allclose(local.mean(0), TOOL_AXIS * TOOL_OFFSET, atol=0.01))

    synth = {k: [] for k in PHASES}
    rng = np.random.default_rng(0)
    progs = [p for p in generator.all_programs() if p.manipulation.startswith("carry")]
    for i in range(args.synthetic):
        p = progs[int(rng.integers(len(progs)))]
        try:
            ep = generator.sample_episode(p, 3, 0 if p.eligibility_kind == "all" else 1, rng)
        except generator.Infeasible:
            continue
        lines, _ = _lines(ep)
        for k, vals in phase_durations(ep, lines).items():
            synth[k] += vals
    durations = {k: {"recorded": summary(real[k]), "synthetic": summary(synth[k])} for k in PHASES}
    for k, d in durations.items():
        r, s = d["recorded"]["median"], d["synthetic"]["median"]
        d["ratio_recorded_to_synthetic"] = (r / s) if (r and s) else None
    out = {"source_runs": [str(p) for p in args.runs], "runs_used": len(runs),
           "tool": tool, "durations_s": durations,
           "provenance": "measured on recorded RoboTwin runs (observation-only timelines); "
                         "exchange timing is synthetic until handover recordings exist"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"tool": tool, "durations": {k: (d["recorded"]["median"], d["synthetic"]["median"],
                                                     d["ratio_recorded_to_synthetic"])
                                                 for k, d in durations.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
