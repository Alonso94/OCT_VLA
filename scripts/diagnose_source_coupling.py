#!/usr/bin/env python
"""Does a trained conditional-source policy use where its flow starts?

For a checkpoint trained with ``action_source="via"``, on recorded frames of the view it
was trained on, predict the action chunk twice under the *same* GR00T draw: once with
``rho = 0`` (every sample starts from N(0, I)) and once with ``rho = 1`` (every sample
with via-points starts from mu + sigma * eps). Everything is compared in GR00T's
normalised action space ([-1, 1] min-max, the space mu lives in), on the 16 eeabs
columns, per arm, over samples that have a proposal:

- ``shift``: RMS(chunk_rho1 - chunk_rho0). Near zero means the flow ignores its start:
  the source is inert after training, whatever it was at init.
- ``to_expert``: RMS(chunk - recorded expert chunk) at rho 0 and 1, and for the
  proposal mu itself. Shows whether starting at the proposal moves the prediction
  toward the expert, toward the proposal, or neither.
- ``to_proposal``: RMS(chunk - mu) at rho 0 and 1.
- ``pull``: the share of the way from the rho-0 chunk to mu that the rho-1 chunk
  travels, <(c1 - c0), (mu - c0)> / |mu - c0|^2: 0 = ignores the source, 1 = copies it.

Validation loss is not involved and nothing here ranks cells; it tells apart "GR00T
ignores the source" from "the source is followed and is the problem".

    diagnose_source_coupling.py --checkpoint $RUN/checkpoints/last/pretrained_model \\
        --dataset-root $OCTVLA_DATASET_ROOT/three_object_v4l_eeabs_fkwpr_arms \\
        --output $OCTVLA_OUTPUT_ROOT/diagnostics/source_coupling_GAF-kv_src-s1000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = {"left": slice(0, 8), "right": slice(8, 16)}
POSITION = {"left": slice(0, 3), "right": slice(8, 11)}


def rms(x, axes):
    return np.sqrt(np.mean(np.square(x), axis=axes))


def summarise(c0, c1, mu, expert, valid):
    """Per-arm statistics over valid samples; arrays [N, H, 16] normalised."""
    out = {"samples": int(valid.sum())}
    c0, c1, mu, expert = (a[valid] for a in (c0, c1, mu, expert))
    for name, cols in {**{f"{k}_all": v for k, v in ARMS.items()},
                       **{f"{k}_pos": v for k, v in POSITION.items()}}.items():
        d0, d1, dm = c0[..., cols], c1[..., cols], mu[..., cols]
        e = expert[..., cols]
        gap = (dm - d0).reshape(len(d0), -1)
        move = (d1 - d0).reshape(len(d0), -1)
        denom = np.square(gap).sum(1)
        keep = denom > 1e-8
        pull = (move * gap).sum(1)[keep] / denom[keep]
        out[name] = {
            "shift": float(rms(d1 - d0, None)),
            "to_expert_rho0": float(rms(d0 - e, None)),
            "to_expert_rho1": float(rms(d1 - e, None)),
            "to_expert_proposal": float(rms(dm - e, None)),
            "to_proposal_rho0": float(rms(d0 - dm, None)),
            "to_proposal_rho1": float(rms(d1 - dm, None)),
            "pull_median": float(np.median(pull)) if keep.any() else None,
            "pull_mean": float(np.mean(pull)) if keep.any() else None,
        }
    return out


def main() -> int:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    import oct_vla.policies  # noqa: F401
    from oct_vla.policies.conditioning.action_source import normalize_actions

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--split", default="train", help="split_manifest.json key")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    from oct_vla.data.policy_inputs import rename_map_for

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = args.checkpoint
    if getattr(config, "action_source", "gaussian") == "gaussian":
        raise SystemExit("not a conditional-source checkpoint")
    trained_rho = config.action_source_rho
    rename_map = rename_map_for(config.type)
    repo_id = f"local/{args.dataset_root.name}"
    meta = LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    horizon = config.chunk_size
    dataset = LeRobotDataset(repo_id, root=args.dataset_root, delta_timestamps={
        "action": [i / meta.fps for i in range(horizon)]})
    policy = make_policy(cfg=config, ds_meta=meta, rename_map=rename_map)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                "rename_observations_processor": {"rename_map": rename_map}})
    inner = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    lo = torch.tensor(config.action_source_min)
    hi = torch.tensor(config.action_source_max)

    # Frames from the named split's episodes, evenly spread, deterministic.
    split = json.loads((args.dataset_root / "split_manifest.json").read_text())
    episodes = (split.get(args.split) or {}).get("episodes")
    if not episodes:
        raise SystemExit(f"split {args.split!r} not in split_manifest.json "
                         f"(keys {list(split)})")
    # Recorded as an inclusive [first, last] range of LeRobot episode indices.
    entry = split[args.split]
    if len(episodes) != 2 or episodes[1] - episodes[0] + 1 != entry["count"]:
        raise SystemExit(f"split {args.split!r}: episodes {episodes} is not an inclusive "
                         f"range of {entry['count']}")
    episodes = set(range(int(episodes[0]), int(episodes[1]) + 1))
    ep_index = np.asarray(dataset.hf_dataset["episode_index"])
    candidates = np.flatnonzero(np.isin(ep_index, sorted(episodes)))
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(candidates, size=min(args.frames, len(candidates)),
                                replace=False))
    print(f"{len(chosen)} frames from {len(episodes)} {args.split} episodes", flush=True)

    collect = {k: [] for k in ("c0", "c1", "mu", "expert", "valid", "pad")}
    for start in range(0, len(chosen), args.batch):
        items = [dataset[int(i)] for i in chosen[start:start + args.batch]]
        raw = torch.utils.data.default_collate(items)
        expert = normalize_actions(raw["action"].float(), lo, hi).numpy()
        pad = raw.get("action_is_pad")
        batch = _preprocess_dataset_batch(raw, meta.camera_keys, rename_map, preprocessor)
        chunks = {}
        for rho in (0.0, 1.0):
            inner.config.action_source_rho = rho
            torch.manual_seed(args.seed * 100003 + start)
            with torch.no_grad():
                chunks[rho] = policy.predict_action_chunk(batch).float().cpu().numpy()[..., :16]
            source = inner._last_source
            if rho == 1.0:
                assert bool((source.use == source.valid.to(source.use.device)).all()), \
                    "at rho 1 every sample with via-points must use the source"
        collect["c0"].append(chunks[0.0])
        collect["c1"].append(chunks[1.0])
        collect["mu"].append(source.mu.float().cpu().numpy())
        collect["valid"].append(source.valid.cpu().numpy())
        collect["expert"].append(expert)
        collect["pad"].append(np.zeros(expert.shape[:2], bool) if pad is None
                              else pad.numpy().astype(bool))
        print(f"  {start + len(items)}/{len(chosen)}", flush=True)
    inner.config.action_source_rho = trained_rho
    arr = {k: np.concatenate(v) for k, v in collect.items()}
    # predict_action_chunk returns the executed steps (n_action_steps), not the whole
    # horizon: compare over those, which are the ones a rollout acts on.
    steps = arr["c0"].shape[1]
    for k in ("mu", "expert", "pad"):
        arr[k] = arr[k][:, :steps]
    if arr["c0"].shape != arr["expert"].shape:
        raise SystemExit(f"chunk {arr['c0'].shape} vs expert {arr['expert'].shape}")
    # Padded expert steps (episode end) are replaced by the last real step, as LeRobot does.
    valid = arr["valid"].astype(bool) & ~arr["pad"].any(1)
    report = {
        "checkpoint": str(args.checkpoint), "dataset": str(args.dataset_root),
        "split": args.split, "frames": len(chosen), "trained_rho": trained_rho,
        "sigma": config.action_source_sigma, "space": "normalised [-1, 1], eeabs columns",
        "steps_compared": int(steps),
        "with_proposal": summarise(arr["c0"], arr["c1"], arr["mu"], arr["expert"], valid),
        "share_with_proposal": float(arr["valid"].mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
