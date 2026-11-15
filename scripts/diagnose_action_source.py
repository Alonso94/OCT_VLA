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
    def dense(x):  # parquet nests list columns as object arrays, one level per axis
        return np.stack([dense(v) for v in x]) if isinstance(x, (np.ndarray, list)) and len(x) \
            and isinstance(x[0], (np.ndarray, list)) else np.asarray(x, np.float32)

    stack = lambda col: np.stack([dense(x) for x in t[col]])
    return {"episode": t["episode_index"].to_numpy(), "frame": t["frame_index"].to_numpy(),
            "tokens": stack("observation.entity_tokens"), "emask": stack("observation.entity_mask"),
            "kf": stack("observation.future_keyframe"),
            "kmask": stack("observation.future_keyframe_mask"),
            **({"vias": stack("observation.arm_targets"),
                "vmask": stack("observation.arm_targets_mask")}
               if "observation.arm_targets" in t else {}),
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


def arm_phase(vias, vmask, tokens, emask):
    """Phases of the paired script oracle's targets: the compactor at work; the mover
    open with a close ahead (grasp: approach and descend); closed (carrying); open with
    no close ahead (release, retreat, park); none."""
    from oct_vla.policies.conditioning.action_source import GRIPPER_OPEN_THRESHOLD, LEFT_TYPE

    is_left = (tokens[..., LEFT_TYPE] > 0.5) & (emask > 0.5)
    aperture = (tokens[..., 12] * is_left).sum(-1)
    open_now = aperture >= GRIPPER_OPEN_THRESHOLD
    close_ahead = ((vias[:, 0, :, 7] < 0.5) & (vmask[:, 0] > 0.5)).any(-1)
    has = vmask[:, 0].any(-1)
    return np.where(vmask[:, 1].any(-1), "compact",
                    np.where(~has, "none",
                             np.where(~open_now, "carry",
                                      np.where(close_ahead, "grasp", "after_release"))))


def main() -> int:
    import torch

    from oct_vla.policies.conditioning.action_source import (
        kinematic_proposal,
        normalize_actions,
        via_point_proposal,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--speed", type=float, default=None, help="Default: fitted on train episodes")
    ap.add_argument("--dwell", type=int, default=None,
                    help="arms: steps paused at a gripper switch; default: fitted on train")
    ap.add_argument("--targets", choices=("keyframe", "arms"), default="keyframe",
                    help="keyframe: the 3/4-row keyframe; arms: the per-arm via-points")
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
    rstep = np.linalg.norm(np.diff(d["action"][:, 8:11], axis=0), axis=-1)
    right_speed = float(np.median(rstep[same & (rstep > MOVING)]) * 15.0)

    T = len(d["action"])

    def build(rows, speeds, dwell):
        out = []
        for s in range(0, len(rows), 512):
            r = rows[s:s + 512]
            with torch.no_grad():
                p, _ = via_point_proposal(
                    torch.tensor(d["tokens"][r], dtype=torch.float64),
                    torch.tensor(d["emask"][r] > 0.5),
                    torch.tensor(d["vias"][r], dtype=torch.float64),
                    torch.tensor(d["vmask"][r] > 0.5), speed=speeds, dwell=dwell)
            out.append(p)
        return torch.cat(out)

    dwell = args.dwell
    if args.targets == "arms" and dwell is None:
        # Fit on training frames only: the dwell with the lowest normalised MSE overall.
        rows = np.flatnonzero(train)
        e_tr = normalize_actions(torch.as_tensor(expert[rows], dtype=torch.float64), lo, hi).numpy()
        w = ok[rows][..., None]
        def fit_err(dw):
            pr = normalize_actions(build(rows, (speed, right_speed), dw), lo, hi).numpy()
            return float((((pr - e_tr) ** 2) * w).sum() / w.sum())
        errs = {dw: fit_err(dw) for dw in (0, 4, 8, 12, 16, 24)}
        dwell = min(errs, key=errs.get)
        print("dwell fit (train MSE):", {k: round(v, 4) for k, v in errs.items()})
    props = []
    for s in range(0, T, 512) if args.targets != "arms" else []:
        sl = slice(s, min(s + 512, T))
        with torch.no_grad():
            tok = torch.tensor(d["tokens"][sl], dtype=torch.float64)
            em = torch.tensor(d["emask"][sl] > 0.5)
            if True:
                p, _ = kinematic_proposal(tok, em, torch.tensor(d["kf"][sl], dtype=torch.float64),
                                          torch.tensor(d["kmask"][sl] > 0.5), speed=speed)
        props.append(p)
    prop = build(np.arange(T), (speed, right_speed), dwell) if args.targets == "arms" \
        else torch.cat(props)
    with torch.no_grad():
        persist, _ = kinematic_proposal(
            torch.tensor(d["tokens"], dtype=torch.float64), torch.tensor(d["emask"] > 0.5),
            torch.zeros_like(torch.tensor(d["kf"], dtype=torch.float64)),
            torch.zeros(d["kmask"].shape, dtype=torch.bool), speed=speed)
    norm = lambda x: normalize_actions(torch.as_tensor(x, dtype=torch.float64), lo, hi).numpy()
    E, P, H = norm(expert), norm(prop), norm(persist)
    Z = np.zeros_like(E)
    ph = arm_phase(d["vias"], d["vmask"], d["tokens"], d["emask"]) if args.targets == "arms" else phase(d["kf"], d["kmask"])

    def mse(src, rows, g):
        err = ((src[rows][..., GROUPS[g]] - E[rows][..., GROUPS[g]]) ** 2).mean(-1)
        w = ok[rows]
        return float((err * w).sum() / max(w.sum(), 1))

    report = {"dataset": str(args.dataset), "targets": args.targets, "speed_m_per_s": speed,
              "right_speed_m_per_s": right_speed, "dwell_steps": dwell,
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
    gain = lambda k, g: report["by_phase"].get(k, {}).get(g, {}).get("gain_vs_persistence")
    if args.targets == "arms":
        gains = {"all/left_pos": gain("all", "left_pos"),
                 "grasp/left_pos": gain("grasp", "left_pos"),
                 "carry/left_pos": gain("carry", "left_pos"),
                 "compact/right_pos": gain("compact", "right_pos")}
    else:
        gains = {k: gain(k, "left_pos") for k in ("all", "approach", "transport")}
    report["gate1"] = {
        "left_pos_gain": gains,
        "grip_not_worse": all(a[g]["proposal"] <= a[g]["persistence"] * 1.05 for g in ("left_grip", "right_grip")),
        "pass": all(g is not None and g >= 0.30 for g in gains.values())
        and all(a[g]["proposal"] <= a[g]["persistence"] * 1.05 for g in ("left_grip", "right_grip")),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"speed {speed:.3f} m/s (right {right_speed:.3f}) from {len(moving)} moving steps, "
          f"dwell {dwell}")
    for name, entry in report["by_phase"].items():
        print(name, entry["n"], {g: (round(v["proposal"], 4), round(v["persistence"], 4),
                                     round(v["gaussian_mean"], 4)) for g, v in entry.items()
                                 if g != "n" and g.endswith(("pos", "grip"))})
    print("gate1", report["gate1"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
