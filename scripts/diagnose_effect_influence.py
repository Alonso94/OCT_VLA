#!/usr/bin/env python
"""Does the effect Z causally steer an effect-conditioned policy? (offline, before rollouts)

On recorded frames of the effect view, holding the image, proprioception and object
state fixed, it predicts the action chunk under different effects Z, all with the same
GR00T draw:

- ``true``: the frame's hindsight effect;
- ``alt``: another valid effect, the same placement goal but a *different* object still
  waiting on the lower shelf (what another rule would have named next);
- ``shuffled``: the effect of a frame from a different episode;
- ``none``: no effect (all rows masked).

Scored on **approach frames**: no object held, at least two objects left on the lower
shelf. For each chunk the mover's (left arm's) predicted end-effector path is compared
with the two candidate objects. ``heads_to`` is the object whose horizontal distance
the chunk reduces more; ``follows`` is the share of frames where that is Z's object,
under ``true`` and under ``alt``. A causal effect branch follows Z under both: it heads
for the true object with the true Z and for the other one with the alternative.
A branch that heads for the same object under both ignores Z (rule-ambiguous kv would).

Also reported: the chunk change between variants (RMS, normalised action space), and
the share of frames where the true-Z chunk heads to the true object at all.

    diagnose_effect_influence.py --checkpoint $RUN/checkpoints/last/pretrained_model \\
        --dataset-root $OCTVLA_DATASET_ROOT/three_object_v4lr_eeabs_effA --split val \\
        --output $OCTVLA_OUTPUT_ROOT/diagnostics/effect_influence_GEF-kv-s1000-val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOWER_Z = 0.85       # m: an object centre below this rests on the lower shelf
HOLD_DIST = 0.20     # m: an object this close to the mover counts as held / engaged
MOVABLE = 13         # entity token column of the "movable" type one-hot
LEFT = slice(0, 3)   # the mover's (left arm's) position columns of the eeabs action


def movables(tokens, mask):
    """Indices of valid movable entities and their positions [K, 3]."""
    idx = [k for k in range(len(tokens)) if mask[k] > 0.5 and tokens[k, MOVABLE] > 0.5]
    return idx, tokens[idx, :3]


def approach_case(tokens, mask, z_rows, z_mask, eef):
    """(true object position, alternative object index) if the frame is an approach
    frame with an alternative, else None. ``eef`` is the mover position now."""
    if not (z_mask[0] > 0.5 and z_mask[2] > 0.5):
        return None
    idx, pos = movables(tokens, mask)
    if len(idx) < 2:
        return None
    true_k = int(np.argmin(np.linalg.norm(pos - z_rows[0, :3], axis=1)))
    if np.linalg.norm(pos[true_k] - z_rows[0, :3]) > 1e-3:
        return None  # Z's object is not an entity of this frame (should not happen)
    if np.linalg.norm(pos - eef, axis=1).min() < HOLD_DIST:
        return None  # engaged with an object: not an approach
    lower = [k for k in range(len(idx)) if pos[k, 2] < LOWER_Z and k != true_k]
    if pos[true_k, 2] >= LOWER_Z or not lower:
        return None
    alt_k = min(lower, key=lambda k: np.linalg.norm(pos[k, :2] - pos[true_k, :2]))
    return true_k, alt_k, idx, pos


def alt_rows(z_rows, tokens, entity):
    """The same goal, a different object: row 0 the other object's token (re-posed
    upright as the effect rows are), row 2 the goal position with that object's size."""
    from oct_vla.data.effects import yaw_token

    tok = tokens[entity]
    yaw = float(np.arctan2(tok[4], tok[3]))
    quat = (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))
    size = tuple(float(v) for v in tok[9:12])
    out = z_rows.copy()
    out[0, :17] = yaw_token(tok[:3], quat, size)
    goal = z_rows[2]
    gyaw = float(np.arctan2(goal[4], goal[3]))
    out[2, :17] = yaw_token(goal[:3], (0.0, 0.0, float(np.sin(gyaw / 2)),
                                       float(np.cos(gyaw / 2))), size)
    return out


def heads_to(path, eef, a, b):
    """Which of two object positions the path closes on more, horizontally: 0 or 1."""
    gain = [np.linalg.norm(eef[:2] - o[:2]) - np.linalg.norm(path[-1, :2] - o[:2])
            for o in (a, b)]
    return int(gain[1] > gain[0])


def main() -> int:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    import oct_vla.policies  # noqa: F401
    from oct_vla.data.policy_inputs import rename_map_for
    from oct_vla.policies.conditioning.future import FUTURE_KEYFRAME, FUTURE_MASK

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=2000, help="Candidate frames sampled")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = args.checkpoint
    if not getattr(config, "object_future", False):
        raise SystemExit("the checkpoint has no future (effect) branch")
    rename_map = rename_map_for(config.type)
    repo_id = f"local/{args.dataset_root.name}"
    meta = LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    if not info.get("future_keyframe", {}).get("goal_only"):
        raise SystemExit(f"{args.dataset_root} is not an effect view")
    dataset = LeRobotDataset(repo_id, root=args.dataset_root)
    policy = make_policy(cfg=config, ds_meta=meta, rename_map=rename_map)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                "rename_observations_processor": {"rename_map": rename_map}})
    lo = np.asarray(meta.stats["action"]["min"], float)
    hi = np.asarray(meta.stats["action"]["max"], float)

    split = json.loads((args.dataset_root / "split_manifest.json").read_text())[args.split]
    first, last = split["episodes"]
    hf = dataset.hf_dataset
    ep = np.asarray(hf["episode_index"])
    candidates = np.flatnonzero((ep >= first) & (ep <= last))
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(candidates, size=min(args.frames, len(candidates)), replace=False)

    cases = []
    for i in np.sort(picked):
        item = hf[int(i)]
        tokens = np.asarray(item["observation.entity_tokens"], float)
        emask = np.asarray(item["observation.entity_mask"], float)
        z = np.asarray(item[FUTURE_KEYFRAME], float).reshape(3, -1)
        zm = np.asarray(item[FUTURE_MASK], float)
        eef = np.asarray(item["observation.state"], float)[0:3]
        case = approach_case(tokens, emask, z, zm, eef)
        if case is None:
            continue
        true_k, alt_k, idx, pos = case
        cases.append({"i": int(i), "episode": int(ep[i]), "z": z, "zm": zm, "eef": eef,
                      "true": pos[true_k], "alt": pos[alt_k],
                      "alt_rows": alt_rows(z, tokens, idx[alt_k])})
    if not cases:
        raise SystemExit("no approach frame with an alternative object")
    # A shuffled effect: the true rows of a case from another episode.
    for n, case in enumerate(cases):
        others = [c for c in cases if c["episode"] != case["episode"]]
        case["shuffled"] = others[n % len(others)]["z"] if others else case["z"]
    print(f"{len(cases)} approach frames from {len({c['episode'] for c in cases})} "
          f"{args.split} episodes", flush=True)

    def predict(items, variant):
        raw = torch.utils.data.default_collate([dataset[c["i"]] for c in items])
        rows = np.stack([c["z"] if variant == "true" else c["alt_rows"] if variant == "alt"
                         else c["shuffled"] if variant == "shuffled" else c["z"]
                         for c in items])
        mask = np.stack([c["zm"] for c in items])
        if variant == "none":
            mask = np.zeros_like(mask)
        raw[FUTURE_KEYFRAME] = torch.tensor(rows, dtype=torch.float32)
        raw[FUTURE_MASK] = torch.tensor(mask, dtype=torch.float32)
        batch = _preprocess_dataset_batch(raw, meta.camera_keys, rename_map, preprocessor)
        torch.manual_seed(args.seed)
        with torch.no_grad():
            return policy.predict_action_chunk(batch).float().cpu().numpy()[..., :16]

    variants = ("true", "alt", "shuffled", "none")
    chunks = {v: [] for v in variants}
    for start in range(0, len(cases), args.batch):
        items = cases[start:start + args.batch]
        for v in variants:
            chunks[v].append(predict(items, v))
        print(f"  {start + len(items)}/{len(cases)}", flush=True)
    chunks = {v: np.concatenate(c) for v, c in chunks.items()}

    def raw_path(norm):
        return (norm + 1) / 2 * (hi[LEFT] - lo[LEFT]) + lo[LEFT]

    out = {"checkpoint": str(args.checkpoint), "dataset": str(args.dataset_root),
           "split": args.split, "approach_frames": len(cases),
           "episodes": len({c["episode"] for c in cases})}
    for v in variants:
        toward_true = [heads_to(raw_path(chunks[v][n][:, LEFT]), c["eef"], c["true"],
                                c["alt"]) == 0 for n, c in enumerate(cases)]
        out[f"heads_to_true_{v}"] = float(np.mean(toward_true))
    out["follows_true"] = out["heads_to_true_true"]
    out["follows_alt"] = 1.0 - out["heads_to_true_alt"]
    out["follows_both"] = float(np.mean([
        heads_to(raw_path(chunks["true"][n][:, LEFT]), c["eef"], c["true"], c["alt"]) == 0
        and heads_to(raw_path(chunks["alt"][n][:, LEFT]), c["eef"], c["true"], c["alt"]) == 1
        for n, c in enumerate(cases)]))
    for v in ("alt", "shuffled", "none"):
        out[f"chunk_change_{v}"] = float(np.sqrt(np.mean((chunks[v] - chunks["true"]) ** 2)))
        out[f"mover_change_{v}"] = float(np.sqrt(np.mean(
            (chunks[v][..., LEFT] - chunks["true"][..., LEFT]) ** 2)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
