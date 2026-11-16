#!/usr/bin/env python
"""A dataset view whose future-keyframe columns hold the hindsight object effect.

Writes ``<source>_eff<contract>``: the source view's parquet plus
``observation.future_keyframe`` [3, 24] and ``observation.future_keyframe_mask`` [3],
the goal-only effect rows of ``oct_vla.data.effects`` (row 0 the object to change, now;
row 2 its goal; gripper row masked; event empty). Videos are hard-linked back to the
source. The policy reads them through the existing future branch.

Labels come from object poses, EEF positions and the measured aperture only. Each
episode is then checked against the online oracle that will serve the same contract
at evaluation (``ShelfEffectOracle(spec_for_rule(rule))``, the server's ``oracle_effect``),
and refused unless, for contract A:

- the effect's object matches the oracle's target on >= MIN_AGREEMENT of frames,
- every switch is within MAX_SWITCH_LAG frames of the oracle's,
- every transfer's goal is within MAX_GOAL_CM of the oracle's (median over its frames),

and the whole corpus is refused unless the median goal distance over all transfers is
<= MAX_CORPUS_GOAL_CM. The goal bounds catch a *systematic* mismatch (the oracle
planning beside an uncompacted neighbour was ~5 cm on every compaction segment). They
do not ask the two to agree to the millimetre: the labels are where the object really
came to rest, the oracle serves where it plans to put it, and objects shift in the
gripper (on the v4 runs: median 0.6 cm, max 4.3 cm, correlated 0.5 with the in-gripper
shift). Object identity and switch timing are the effect's discrete content and stay
strict.

Contract B holds the placed object through compaction, which the online oracle does
not (yet) do: its agreement is recorded, not gated.

    add_effects.py --source $OCTVLA_DATASET_ROOT/three_object_v4lr_eeabs --contract A
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MIN_AGREEMENT = 0.98
MAX_SWITCH_LAG = 5
MAX_GOAL_CM = 5.0          # per transfer: a bug, not placement noise
MAX_CORPUS_GOAL_CM = 2.0   # median over all transfers: a systematic offset


def oracle_trace(episode, spec):
    """The online effect oracle over the recording: its target and rows per frame,
    exactly what the server serves (``ShelfEffectOracle``)."""
    from oct_vla.tasks.shelf_restock.keyframes import ShelfEffectOracle

    oracle = ShelfEffectOracle(spec)
    rows, mask = zip(*(oracle(s.scene, s.observation.eef) for s in episode.samples))
    if oracle.unplannable_steps:
        raise SystemExit(f"seed {episode.seed}: oracle unplannable on "
                         f"{oracle.unplannable_steps} frames ({oracle.last_failure})")
    return [t for _, t in oracle.history], np.stack(rows), np.stack(mask)


def switches(targets) -> dict[str, int]:
    out = {}
    for t in range(1, len(targets)):
        if targets[t - 1] is not None and targets[t] != targets[t - 1] \
                and targets[t - 1] not in out:
            out[targets[t - 1]] = t
    return out


def compare(target, rows, mask, o_target, o_rows, o_mask) -> dict:
    """Hindsight vs oracle: object agreement, switch lags, goal distance (cm)."""
    agree = float(np.mean([a == b for a, b in zip(target, o_target)]))
    ours, theirs = switches(target), switches(o_target)
    lags = [theirs[k] - v for k, v in ours.items() if k in theirs]
    both = [t for t in range(len(target)) if target[t] is not None and target[t] == o_target[t]
            and mask[t, 2] and o_mask[t, 2]]
    goal = [100 * float(np.linalg.norm(rows[t, 2, :3] - o_rows[t, 2, :3])) for t in both]
    transfers = {}
    for t, g in zip(both, goal):
        transfers.setdefault(target[t], []).append(g)
    # Row 0 is the object's current pose in both: identical tokens, or the formats differ.
    row0 = max((float(np.abs(rows[t, 0] - o_rows[t, 0]).max()) for t in both), default=0.0)
    if row0 > 1e-5:
        raise SystemExit(f"hindsight and online row 0 differ by {row0}: the row formats "
                         "disagree")
    return {"agreement": agree, "lags": lags, "missing_switches": sorted(set(ours) - set(theirs)),
            "goal_cm_median": float(np.median(goal)) if goal else None,
            "goal_cm_p90": float(np.percentile(goal, 90)) if goal else None,
            "goal_cm_transfer": {k: float(np.median(v)) for k, v in transfers.items()}}


def main() -> int:
    import pandas as pd

    from add_future_keyframe import _stats
    from oct_vla.data.effects import CONTRACTS, hindsight_effects
    from oct_vla.data.future_keyframes import FUTURE_DIM, FUTURE_ROWS
    from oct_vla.data.store import read_episode
    from oct_vla.policies.conditioning.future import FUTURE_KEYFRAME, FUTURE_MASK
    from oct_vla.tasks.shelf_restock.spec import spec_for_rule
    from project_dataset_view import link_tree

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--contract", required=True, choices=CONTRACTS)
    ap.add_argument("--output", type=Path, help="Default: <source>_eff<contract>")
    args = ap.parse_args()
    source = args.source.resolve()
    output = (args.output or source.with_name(f"{source.name}_eff{args.contract}")).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    info = json.loads((source / "meta" / "info.json").read_text())
    if FUTURE_KEYFRAME in info["features"]:
        raise SystemExit(f"{source} already has {FUTURE_KEYFRAME}; derive from a view without")
    manifest = json.loads((source / "octvla_episode_manifest.json").read_text())
    by_index = {int(e["lerobot_episode_index"]): e for e in manifest["episodes"]}
    if any("source_path" not in e for e in by_index.values()):
        raise SystemExit(f"{source}: the provenance manifest records no source_path (exported "
                         "before mixed-rule support); re-export")

    output.mkdir(parents=True)
    linked = link_tree(source / "videos", output / "videos")
    shutil.copytree(source / "meta", output / "meta")
    for name in ("split_manifest.json", "octvla_episode_manifest.json"):
        shutil.copy2(source / name, output / name)

    real, masks, checks, frames = [], [], {}, 0
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pd.read_parquet(path)
        rows_col, mask_col = [None] * len(table), [None] * len(table)
        for episode_index, idx in table.groupby("episode_index").groups.items():
            part = table.loc[idx].sort_values("frame_index")
            entry = by_index[int(episode_index)]
            episode = read_episode(entry["source_path"])
            if len(part) != len(episode.samples):
                raise SystemExit(f"episode {episode_index}: {len(part)} dataset frames, "
                                 f"{len(episode.samples)} recorded")
            rule = episode.metadata["rule"]
            spec = spec_for_rule(rule)
            rows, mask, target, _ = hindsight_effects(episode, spec, args.contract)
            o_target, o_rows, o_mask = oracle_trace(episode, spec)
            check = compare(target, rows, mask, o_target, o_rows, o_mask)
            check.update(rule=rule, seed=episode.seed)
            checks[int(episode_index)] = check
            del episode
            gc.collect()
            if args.contract == "A":
                bad = (check["agreement"] < MIN_AGREEMENT or check["missing_switches"]
                       or any(abs(lag) > MAX_SWITCH_LAG for lag in check["lags"])
                       or not check["goal_cm_transfer"]
                       or max(check["goal_cm_transfer"].values()) > MAX_GOAL_CM)
                if bad:
                    raise SystemExit(f"episode {episode_index} ({rule} seed {check['seed']}): "
                                     f"hindsight and online effects disagree: {check}")
            order = part["frame_index"].to_numpy()
            for position, index in enumerate(part.index):
                at = table.index.get_loc(index)
                rows_col[at] = rows[order[position]].tolist()
                mask_col[at] = mask[order[position]].astype(np.float32).tolist()
            real.append(rows[mask])
            masks.append(mask.astype(np.float32))
            frames += len(part)
            print(f"episode {episode_index} ({rule} seed {check['seed']}): {len(part)} frames, "
                  f"effect on {int(mask[:, 0].sum())}, agreement {check['agreement']:.3f}, "
                  f"lags {check['lags']}, goal {check['goal_cm_median']:.2f} cm", flush=True)
        if any(v is None for v in rows_col):
            raise SystemExit(f"{path}: a row got no effect")
        table[FUTURE_KEYFRAME] = rows_col
        table[FUTURE_MASK] = mask_col
        target_path = output / path.relative_to(source)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target_path, index=False)

    agreement = [c["agreement"] for c in checks.values()]
    goal = [c["goal_cm_median"] for c in checks.values() if c["goal_cm_median"] is not None]
    lags = [lag for c in checks.values() for lag in c["lags"]]
    per_transfer = [g for c in checks.values() for g in c["goal_cm_transfer"].values()]
    corpus_goal = float(np.median(per_transfer))
    if args.contract == "A" and corpus_goal > MAX_CORPUS_GOAL_CM:
        raise SystemExit(f"median goal distance over {len(per_transfer)} transfers is "
                         f"{corpus_goal:.2f} cm > {MAX_CORPUS_GOAL_CM}: a systematic offset")
    info = json.loads((output / "meta" / "info.json").read_text())
    info["features"][FUTURE_KEYFRAME] = {"dtype": "float32", "shape": [FUTURE_ROWS, FUTURE_DIM],
                                         "names": None}
    info["features"][FUTURE_MASK] = {"dtype": "float32", "shape": [FUTURE_ROWS], "names": None}
    info["derived_from"] = source.name
    info["future_keyframe"] = {
        "source": "oracle_effect", "labels": "hindsight", "contract": args.contract,
        "kind": "keyframe", "rows": FUTURE_ROWS, "dim": FUTURE_DIM, "goal_only": True,
        "rules": sorted({c["rule"] for c in checks.values()}),
        "oracle_check": {"agreement_min": min(agreement), "agreement_mean": float(np.mean(agreement)),
                         "switch_lag_min": min(lags) if lags else None,
                         "switch_lag_max": max(lags) if lags else None,
                         "goal_cm_median_max": max(goal) if goal else None,
                         "goal_cm_transfer_median": corpus_goal,
                         "goal_cm_transfer_p90": float(np.percentile(per_transfer, 90)),
                         "goal_cm_transfer_max": max(per_transfer),
                         "goal_cm_transfer_over_2": int(sum(g > 2 for g in per_transfer)),
                         "transfers": len(per_transfer),
                         "max_goal_cm": MAX_GOAL_CM, "max_corpus_goal_cm": MAX_CORPUS_GOAL_CM,
                         "gated": args.contract == "A"},
    }
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    (output / "meta" / "effect_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats[FUTURE_KEYFRAME] = _stats(np.concatenate(real))
    stats[FUTURE_MASK] = _stats(np.concatenate(masks))
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    item = LeRobotDataset(f"local/{output.name}", root=output).hf_dataset[0]
    if np.asarray(item[FUTURE_KEYFRAME]).size != FUTURE_ROWS * FUTURE_DIM:
        raise SystemExit(f"{output}: loaded {FUTURE_KEYFRAME} has the wrong size")
    print(f"\nwrote {output}: {len(checks)} episodes, {frames} frames; oracle agreement "
          f"min {min(agreement):.3f}, switch lags {min(lags)}..{max(lags)}, "
          f"goal per transfer median {corpus_goal:.2f} / max {max(per_transfer):.2f} cm; {linked} video files linked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
