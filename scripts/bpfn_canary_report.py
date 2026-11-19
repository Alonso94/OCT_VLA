#!/usr/bin/env python
"""Check the colour canary collection (plan 1a) and gate the pilot on it.

For every collected run under ``--root`` (``*/full_run/*/seed_*/episode_*``):

- ``colour_known``: every object carries a colour, with the counts the run asked for;
- ``rendered``: red and blue are both visible in the first head-camera frame (pixels
  near the tint colours), and a labelled contact sheet of all three cameras is saved
  for a human look (a fake-actor test cannot show this);
- ``round_trip``: colours survive the wire codec (store round trip is the read itself);
- ``distractors_still``: every object the rule does not select ends within 2 cm of its
  start;
- ``commands``: ``episode.commands.npz`` exists, has one frame tick per sample, and the
  commanded joints at each frame track the next frame's measured joints.

Writes ``<root>/canary_report.json``, ``<root>/canary_contact_sheet.png`` and, only if
every run passes every check, ``<root>/CANARY_PASSED`` (``submit_bpfn.sh pilot`` needs it).

    bpfn_canary_report.py --root $HPCVAULT/octvla-collection-bpfn/canary
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

DISTRACTOR_TOL = 0.02
JOINT_TRACK_TOL = 0.05     # rad, median |commanded - next measured|
PIXEL_TOL = 0.18           # normalised RGB distance to a tint colour
MIN_PIXELS = 50


def _frame(rgb) -> np.ndarray:
    return np.frombuffer(rgb.data, np.uint8).reshape(rgb.height, rgb.width, 3)


def tint_pixels(image: np.ndarray, rgb) -> int:
    """Pixels whose normalised colour is within PIXEL_TOL of ``rgb`` (chromaticity, so
    shading does not matter)."""
    x = image.reshape(-1, 3).astype(float)
    norm = x / np.maximum(x.sum(1, keepdims=True), 1.0)
    target = np.asarray(rgb, float) / sum(rgb)
    bright = x.sum(1) > 60
    return int(np.sum((np.linalg.norm(norm - target, axis=1) < PIXEL_TOL) & bright))


def check_run(path: Path) -> tuple[dict, list[np.ndarray]]:
    from oct_vla.data.commands import read_commands
    from oct_vla.data.store import read_episode
    from oct_vla.serve import codec
    from oct_vla.tasks.shelf_restock import colours
    from oct_vla.tasks.shelf_restock.spec import RULE_ELIGIBLE_COLOUR

    ep = read_episode(path)
    rule = ep.metadata.get("rule")
    first, last = ep.samples[0].scene, ep.samples[-1].scene
    objs = {o.track_id: o for o in first.objects}
    names = [o.colour for o in first.objects]
    out = {"path": str(path), "rule": rule, "seed": ep.seed, "colours": names,
           "samples": len(ep.samples)}
    out["colour_known"] = all(c in colours.RGB for c in names)
    head = _frame(ep.samples[0].observation.head_rgb)
    counts = {c: tint_pixels(head, colours.RGB[c]) for c in ("red", "blue")}
    out["tint_pixels"] = counts
    out["rendered"] = out["colour_known"] and all(counts[c] >= MIN_PIXELS for c in set(names))
    back = codec.scene_from_json(codec.scene_to_json(first))
    out["round_trip"] = [o.colour for o in back.objects] == names
    wanted = RULE_ELIGIBLE_COLOUR.get(rule)
    moved = {t: float(np.linalg.norm(np.subtract(last.get(t).pose.position,
                                                 objs[t].pose.position)))
             for t in objs if wanted is not None and objs[t].colour != wanted}
    out["distractor_shift_m"] = moved
    out["distractors_still"] = all(d <= DISTRACTOR_TOL for d in moved.values())
    log = read_commands(path)
    if log is None:
        out["commands"] = False
        out["commands_detail"] = "episode.commands.npz missing"
    else:
        aligned = len(log.frame_ticks) == len(ep.samples)
        q, _ = log.at_frames()
        measured = [s.observation.joints for s in ep.samples]
        err = []
        for t in range(len(ep.samples) - 1):
            nxt = measured[t + 1]
            if nxt is None:
                continue
            m = np.array([nxt.left.positions, nxt.right.positions])
            err.append(np.abs(q[t, :, :m.shape[1]] - m).max())
        med = float(np.median(err)) if err else None
        out["commands"] = aligned and med is not None and med < JOINT_TRACK_TOL
        out["commands_detail"] = {"aligned": aligned, "joint_track_median_rad": med}
    frames = [_frame(ep.samples[0].observation.head_rgb),
              _frame(ep.samples[0].observation.left_wrist_rgb),
              _frame(ep.samples[0].observation.right_wrist_rgb)]
    del ep
    gc.collect()
    return out, frames


def contact_sheet(rows: list[tuple[dict, list[np.ndarray]]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.6 * len(rows)), squeeze=False)
    for r, (info, frames) in enumerate(rows):
        for c, (img, name) in enumerate(zip(frames, ("head", "left wrist", "right wrist"))):
            axes[r, c].imshow(img)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].set_title(f"{info['rule']} seed {info['seed']} · {name}", fontsize=7)
        axes[r, 0].set_ylabel(",".join(str(c) for c in info["colours"]), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    runs = sorted(args.root.glob("*/full_run/*/seed_*/episode_*"))
    if not runs:
        raise SystemExit(f"no collected runs under {args.root}")
    rows = [check_run(p) for p in runs]
    checks = ("colour_known", "rendered", "round_trip", "distractors_still", "commands")
    passed = all(info[k] for info, _ in rows for k in checks)
    report = {"runs": [info for info, _ in rows], "checks": checks, "passed": passed}
    (args.root / "canary_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    contact_sheet(rows, args.root / "canary_contact_sheet.png")
    for info, _ in rows:
        print(info["rule"], info["seed"], {k: info[k] for k in checks})
    marker = args.root / "CANARY_PASSED"
    if passed:
        marker.write_text("canary passed; see canary_report.json and the contact sheet\n")
    print("PASSED" if passed else "FAILED", "->", args.root / "canary_report.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
