#!/usr/bin/env python
"""Gate 1 of the conditional-source flow: is the kinematic proposal a better start than noise?

For every frame of a keyframe view (e.g. ``three_object_v4l_eeabs_fkwpr``), build the
40-step proposal (``policies/conditioning/action_source.py``) and compare it, in GR00T's
normalised action space, with the recorded expert chunk (the next 40 action rows,
valid steps only, as GR00T's own mask does). Against two references:

- **persistence:** both arms hold their current pose and gripper for 40 steps;
- **Gaussian mean:** the zero chunk (the flow's default source, in expectation).

Per arm and column group (position, rotation, gripper), and per phase read off the
keyframe (approach / descend for grasp, transport / lower for place, retreat for the
post-release phase, none). Also reports the proposal's RMS residual per group: the
sigma a source must have to cover the expert.

The speed of the proposal is fitted here, from the expert's own moving steps, on the
view's training episodes only, and written to the output for the training config.

    diagnose_action_source.py --dataset $OCTVLA_DATASET_ROOT/three_object_v4l_eeabs_fkwpr \\
        --out $OCTVLA_OUTPUT_ROOT/diagnostics/action_source_fkwpr.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GROUPS = {"left_pos": slice(0, 3), "left_rot": slice(3, 7), "left_grip": slice(7, 8),
          "right_pos": slice(8, 11), "right_rot": slice(11, 15), "right_grip": slice(15, 16)}
#: An expert step that moves the left EEF at least this far counts as moving (speed fit).
MOVING = 0.002


def load(dataset: Path):
    import pandas as pd

    t = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(f"{dataset}/data/**/*.parquet",
                                                               recursive=True)))
    t = t.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    stack = lambda col: np.stack([np.stack(x) if isinstance(x[0], np.ndarray) else np.asarray(x)
                                  for x in t[col]])
    return {"episode": t["episode_index"].to_numpy(), "frame": t["frame_index"].to_numpy(),
            "tokens": stack("observation.entity_tokens"), "emask": stack("observation.entity_mask"),
            "kf": stack("observation.future_keyframe"),
            "kmask": stack("observation.future_keyframe_mask"),
            "action": np.stack([np.asarray(a) for a in t["action"]])}


def chunks(action, episode, horizon=40):
    """Expert chunk [T, H, 16] and its validity [T, H]: rows t..t+H-1 of the same episode."""
    T = len(action)
    idx = np.arange(T)[:, None] + np.arange(horizon)[None]
    ok = idx < T
    idx = np.minimum(idx, T - 1)
    ok &= episode[idx] == episode[:, None]
    return action[idx], ok


def phase(kf, kmask):
    from oct_vla.data.future_keyframes import FUTURE_EVENTS, event_columns

    ev = kf[:, 0, event_columns(kf.shape[-1])].argmax(-1)
    names = np.array(FUTURE_EVENTS)[ev]
    wp = kmask[:, 3] > 0.5 if kf.shape[1] == 4 else np.zeros(len(kf), bool)
    out = np.where(names == "grasp", np.where(wp, "approach", "descend"),
                   np.where(names == "place", np.where(wp, "transport", "lower"),
                            np.where(names == "release", "retreat", "other")))
    return np.where(kmask[:, 0] > 0.5, out, "none")


def main() -> int:
    import torch

    from oct_vla.policies.conditioning.action_source import kinematic_proposal, normalize_actions

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--speed", type=float, default=None, help="Default: fitted on train episodes")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    d = load(args.dataset)
    split = json.loads((args.dataset / "split_manifest.json").read_text())
    lo_ep, hi_ep = split["train"]["episodes"]
    train = (d["episode"] >= lo_ep) & (d["episode"] <= hi_ep)
    stats = json.loads((args.dataset / "meta" / "stats.json").read_text())["action"]
    lo, hi = torch.tensor(stats["min"], dtype=torch.float64), torch.tensor(stats["max"],
                                                                           dtype=torch.float64)
    expert, ok = chunks(d["action"], d["episode"])

    # Speed: the expert's moving steps (left arm, the mover), training episodes only.
    step = np.linalg.norm(np.diff(d["action"][:, :3], axis=0), axis=-1)
    same = (d["episode"][1:] == d["episode"][:-1]) & train[1:]
    moving = step[same & (step > MOVING)]
    speed = args.speed or float(np.median(moving) * 15.0)

    T = len(d["action"])
    props = []
    for s in range(0, T, 512):
        sl = slice(s, min(s + 512, T))
        with torch.no_grad():
            p, _ = kinematic_proposal(
                torch.tensor(d["tokens"][sl], dtype=torch.float64),
                torch.tensor(d["emask"][sl] > 0.5),
                torch.tensor(d["kf"][sl], dtype=torch.float64),
                torch.tensor(d["kmask"][sl] > 0.5), speed=speed)
        props.append(p)
    prop = torch.cat(props)
    with torch.no_grad():
        persist, _ = kinematic_proposal(
            torch.tensor(d["tokens"], dtype=torch.float64), torch.tensor(d["emask"] > 0.5),
            torch.zeros_like(torch.tensor(d["kf"], dtype=torch.float64)),
            torch.zeros(d["kmask"].shape, dtype=torch.bool), speed=speed)
    norm = lambda x: normalize_actions(torch.as_tensor(x, dtype=torch.float64), lo, hi).numpy()
    E, P, H = norm(expert), norm(prop), norm(persist)
    Z = np.zeros_like(E)
    ph = phase(d["kf"], d["kmask"])

    def mse(src, rows, g):
        err = ((src[rows][..., GROUPS[g]] - E[rows][..., GROUPS[g]]) ** 2).mean(-1)
        w = ok[rows]
        return float((err * w).sum() / max(w.sum(), 1))

    report = {"dataset": str(args.dataset), "speed_m_per_s": speed,
              "moving_steps": int(len(moving)), "frames": int(T),
              "train_frames": int(train.sum()), "by_phase": {}}
    for name in ["all", *sorted(set(ph.tolist()))]:
        rows = train & (ph != "none") if name == "all" else train & (ph == name)
        if not rows.any():
            continue
        entry = {"n": int(rows.sum())}
        for g in GROUPS:
            m_p, m_h, m_z = mse(P, rows, g), mse(H, rows, g), mse(Z, rows, g)
            entry[g] = {"proposal": m_p, "persistence": m_h, "gaussian_mean": m_z,
                        "gain_vs_persistence": None if m_h == 0 else 1 - m_p / m_h,
                        "sigma_hat": float(np.sqrt(m_p))}
        report["by_phase"][name] = entry
    a = report["by_phase"]["all"]
    gains = {k: report["by_phase"].get(k, {}).get("left_pos", {}).get("gain_vs_persistence")
             for k in ("all", "approach", "transport")}
    report["gate1"] = {
        "left_pos_gain": gains,
        "grip_not_worse": a["left_grip"]["proposal"] <= a["left_grip"]["persistence"] * 1.05,
        "pass": all(g is not None and g >= 0.30 for g in gains.values())
        and a["left_grip"]["proposal"] <= a["left_grip"]["persistence"] * 1.05,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"speed {speed:.3f} m/s from {len(moving)} moving steps")
    for name, entry in report["by_phase"].items():
        print(name, entry["n"], {g: (round(v["proposal"], 4), round(v["persistence"], 4),
                                     round(v["gaussian_mean"], 4)) for g, v in entry.items()
                                 if g != "n" and g.startswith("left")})
    print("gate1", report["gate1"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
