#!/usr/bin/env python
"""Inspect time-row items and check every label against the v1 generator.

Samples items as `time_row_batches.sample_time_row_items` does, keeping the episodes. It
then re-derives every support row's labels and the query's hidden targets *independently*:
- the next keyframe comes from the row's time (segment index), read off the `Episode`;
- columns are mapped through the permutations recorded in `meta`.

It reports mismatch counts, which must be zero, and prints one item: support rows with
their labelled left/right targets, and the query with both targets masked.

    time_row_inspect.py --items 500 [--split train] [--show 1]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from oct_vla.prior import programs as P
    from oct_vla.prior import time_row_schema as S
    from oct_vla.prior.batches import ItemConfig, _scene_for
    from oct_vla.prior.generator import Infeasible
    from oct_vla.prior.schema import APERTURE, POS
    from oct_vla.prior.time_row_batches import TimeRowConfig, pack_time_row_item

    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=500)
    ap.add_argument("--split", default="train", choices=("train", "composition", "shelf_family"))
    ap.add_argument("--rows-per-segment", type=int, default=1)
    ap.add_argument("--show", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = TimeRowConfig(item=ItemConfig(programs=P.program_split(args.split)),
                        rows_per_segment=args.rows_per_segment)

    mismatches, counts = Counter(), Counter()
    shown = 0
    done = 0
    while done < args.items:
        prog = cfg.item.programs[rng.integers(len(cfg.item.programs))]
        k = int(rng.integers(1, 5))
        try:
            demos = [_scene_for(rng, [prog])[0] for _ in range(k)]
            (q,) = _scene_for(rng, [prog])
        except Infeasible:
            continue
        j = int(rng.integers(len(q.keyframes) - 1))
        alpha = float(rng.choice([0.0, rng.uniform(0, 0.9)]))
        inputs, targets, meta = pack_time_row_item(rng, prog, demos, q, j, alpha, 1, cfg)
        done += 1
        counts[f"arm={prog.arm}"] += 1

        def check(where, ep, kf_index, labels, inv):
            kf = ep.keyframes[kf_index]
            n = ep.scene.n
            counts["labels"] += 1
            if labels["g_event"] != kf.event:
                mismatches[f"{where}:event"] += 1
            if labels["g_object"] != inv[kf.obj]:
                mismatches[f"{where}:object"] += 1
            for a in (0, 1):
                acting = a == kf.gripper
                if labels["y_participation"][a] != float(acting):
                    mismatches[f"{where}:participation"] += 1
                if not np.allclose(labels["y_pos"][a], kf.feats[n + a, POS], atol=1e-5):
                    mismatches[f"{where}:pos"] += 1
                if abs(labels["y_aperture"][a] - kf.feats[n + a, APERTURE]) > 1e-5:
                    mismatches[f"{where}:aperture"] += 1
                want_ev = kf.event if acting else S.HOLD
                want_obj = inv[kf.obj] if acting else -1
                if labels["y_event"][a] != want_ev or labels["y_object"][a] != want_obj:
                    mismatches[f"{where}:arm_event_object"] += 1

        for d, ep in enumerate(demos):
            inv = np.argsort(np.asarray(meta["support_perms"][d]))
            last = len(ep.keyframes) - 1
            for r in np.flatnonzero(inputs["s_row_valid"][d]):
                seg = int(np.floor(inputs["s_row_time"][d, r] + 1e-6))
                labels = {k[2:]: inputs[k][d, r] for k in inputs
                          if k.startswith("s_") and k[2:] in S.ROW_LABELS}
                if seg == last:
                    counts["terminal"] += 1
                    if labels["target_valid"] != 0:
                        mismatches["support:terminal_not_masked"] += 1
                    continue
                check("support", ep, seg + 1, labels, inv)
        check("query", q, j + 1, targets, np.argsort(np.asarray(meta["query_perm"])))
        for k in targets:
            if f"q_{k}" in inputs:
                mismatches["query:target_in_inputs"] += 1

        if shown < args.show and prog.arm == "handover" or (shown < args.show and done > 200):
            shown += 1
            _show(prog, inputs, targets, meta, S)

    print(f"\nchecked {done} items ({args.split}, rows/segment {args.rows_per_segment}): "
          f"{counts['labels']} labelled rows, {counts['terminal']} terminal rows; "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()) if k.startswith("arm=")))
    print("mismatches:", dict(mismatches) if mismatches else "none")
    return 1 if mismatches else 0


def _show(prog, inputs, targets, meta, S):
    names = S.ARM_EVENTS
    print(f"\n=== program {prog} (id {meta['program']}), query keyframe j={meta['j']}, "
          f"alpha={meta['alpha']:.2f}, objects={meta['n_objects']}")
    d = 0
    rows = np.flatnonzero(inputs["s_row_valid"][d])
    print(f"support demo 0: {len(rows)} time rows; per row: time | left target | right target "
          "| group event, object")
    for r in rows:
        if inputs["s_target_valid"][d, r] == 0:
            print(f"  t={inputs['s_row_time'][d, r]:5.2f} | terminal (masked)")
            continue
        cells = []
        for a in (0, 1):
            part = "ACT" if inputs["s_y_participation"][d, r, a] else (
                "eng" if inputs["s_y_engaged"][d, r, a] else "idle")
            pos = inputs["s_y_pos"][d, r, a]
            cells.append(f"{part:4s} {names[inputs['s_y_event'][d, r, a]]:7s} "
                         f"obj {inputs['s_y_object'][d, r, a]:2d} "
                         f"({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:.2f}) ap {inputs['s_y_aperture'][d, r, a]:.2f}")
        print(f"  t={inputs['s_row_time'][d, r]:5.2f} | L: {cells[0]} | R: {cells[1]} "
              f"| {S.GROUP_EVENTS[inputs['s_g_event'][d, r]]}, obj {inputs['s_g_object'][d, r]}")
    print("query: initial + current scene given; left target = [MASKED], right target = [MASKED]")
    print(f"  hidden (targets dict, not an input): group {S.GROUP_EVENTS[targets['g_event']]}, "
          f"object {targets['g_object']}, participation {targets['y_participation'].tolist()}")


if __name__ == "__main__":
    raise SystemExit(main())
