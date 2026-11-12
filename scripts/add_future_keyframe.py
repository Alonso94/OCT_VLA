#!/usr/bin/env python
"""A dataset view with the oracle next-interaction keyframe beside the entity tokens.

PFN-VLA step 2. Writes ``<source>_fk``: the source view's parquet plus
``observation.future_keyframe`` ``[3, 24]`` and ``observation.future_keyframe_mask``
``[3]`` (``data/future_keyframes.py``), videos hard-linked back to the source,
metadata extended with both features.

Keyframes come from ``tasks/shelf_restock/keyframes.py``, the same oracle the
evaluation server runs, applied to the recorded scenes frame by frame. Each
episode is cross-checked against an independent definition read off the phase
labels (``data/future_keyframes.py``): the event switches must be the same in
number and lag them by 0-10 frames (3-7 over the v4 corpus). Episodes are
matched through ``octvla_episode_manifest.json``; frame alignment is asserted as
in ``add_entity_visual.py``.

    add_future_keyframe.py --source DATASET_VIEW \\
        --collection $OCTVLA_COLLECTION_ROOT/three_object

**Learned source** (PFN-VLA step 3, ``--keyframes pfn``): the frozen prior's prediction
through ``prior/shelf_adapter.py``, the path the evaluation client will use. Each
query is the episode's first frame plus the current one. Its support is K
demonstrations drawn, per episode and seeded, from the view's *training* split,
excluding the episode itself. ``--kind keyframe`` stores the decoded ``[3, 24]`` rows,
``--kind latent`` the prior's current-row cells ``[N_MAX + 1, width]``. Both use the
same two keys, and ``meta/info.json`` records the kind. The support pool goes to
``meta/pfn_support.npz`` for the evaluation client. No phase label reaches a query.
For a diagnostic only, the exporter records how often the predicted object is the
recording's next one.

    add_future_keyframe.py --source DATASET_VIEW --collection ... --keyframes pfn \\
        --prior $P/runs/P1-interaction-s0 --lang $P/language/minilm.pt [--kind latent]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


#: Largest accepted lag of the oracle's event switch behind the phase label.
MAX_LAG = 10


def _event_switches(keyframes, mask) -> list[int]:
    """Frames at which the next event's kind changes (grasp, place, or none)."""
    import numpy as np

    from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM
    from oct_vla.data.future_keyframes import FUTURE_ROLES

    start = ENTITY_TOKEN_DIM + len(FUTURE_ROLES)
    kinds = [int(np.argmax(k[0, start:])) if m.all() else -1 for k, m in zip(keyframes, mask)]
    return [i for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1]]


def _switch_lags(keyframes, mask, reference, reference_mask) -> list[int] | None:
    """Lag of each event switch behind the reference's; None if they differ in count."""
    ours, theirs = _event_switches(keyframes, mask), _event_switches(reference, reference_mask)
    if len(ours) != len(theirs):
        return None
    return [a - b for a, b in zip(ours, theirs)]


def _stats(values):
    import numpy as np

    quantiles = {f"q{int(q * 100):02d}": np.quantile(values, q, axis=0).tolist()
                 for q in (0.01, 0.10, 0.50, 0.90, 0.99)}
    return {"min": values.min(0).tolist(), "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
            "count": [int(len(values))], **quantiles}


class _LearnedSource:
    """Frozen-prior keyframes or latent cells for every frame of every episode."""

    def __init__(self, args, source, by_index):
        from oct_vla.data.entity_tokens import shelf_support_entities
        from oct_vla.data.store import read_episode
        from oct_vla.prior import shelf_adapter as A
        from oct_vla.prior.data import load_language
        from oct_vla.tasks.shelf_restock.spec import spec_for_rule

        self.A, self.args = A, args
        self.k = args.k or A.K_SUPPORT
        split = json.loads((source / "split_manifest.json").read_text())
        lo, hi = split["train"]["episodes"]
        self.pool_index = list(range(lo, hi + 1))
        self.model = A.load_prior(args.prior, args.device)
        self.lang = load_language(args.lang)
        self.supports = {}
        self.demos = {}
        for index in self.pool_index:
            entry = by_index[index]
            episode = read_episode(args.collection / f"seed_{entry['seed']}"
                                   / entry["source_episode"])
            rule = episode.metadata["rule"]
            self.supports.setdefault(rule, shelf_support_entities(spec_for_rule(rule)))
            self.demos[index] = A.support_demo(episode, self.supports[rule])
        if len(self.demos) <= self.k:
            raise SystemExit(f"{len(self.demos)} training demos cannot give K={self.k} supports")
        latent = args.kind == "latent"
        width = self.model.config["width"]
        from oct_vla.prior.generator import N_MAX

        self.shape = (N_MAX + 1, width) if latent else (3, 24)
        self.hits: list[bool] = []
        print(f"prior {args.prior} ({self.model.config['head']}), {len(self.demos)} support "
              f"demos in the pool, K={self.k}, kind={args.kind}", flush=True)

    def __call__(self, episode_index, episode, frames):
        from oct_vla.data.future_keyframes import interaction_events
        from oct_vla.data.entity_tokens import shelf_support_entities
        from oct_vla.tasks.shelf_restock.spec import spec_for_rule

        A = self.A
        rng = np.random.default_rng([self.args.seed, episode_index])
        pool = [i for i in self.pool_index if i != episode_index]
        chosen = sorted(int(i) for i in rng.choice(pool, self.k, replace=False))
        supports = shelf_support_entities(spec_for_rule(episode.metadata["rule"]))
        s0 = episode.samples[0]
        queries = [A.query(s0.scene, s0.observation.eef, episode.samples[t].scene,
                           episode.samples[t].observation.eef, supports) for t in frames]
        out = A.infer(self.model, A.pack([self.demos[i] for i in chosen], queries), self.lang,
                      self.args.device, latent=self.args.kind == "latent")
        # Diagnostic only: does the predicted object match the recording's next one?
        events = interaction_events(episode)
        nxt = [next((e.track_id for e in events if e.frame > t), None) for t in frames]
        hits = [A.predicted_track(out, n, q) == want
                for n, (q, want) in enumerate(zip(queries, nxt)) if want is not None]
        self.hits += hits
        if self.args.kind == "latent":
            values, mask = out["latent"], out["latent_ok"]
        else:
            pairs = [A.keyframe_rows(out, n, q) for n, q in enumerate(queries)]
            values = np.stack([r for r, _ in pairs])
            mask = np.stack([m for _, m in pairs])
        note = (f"supports {chosen}, predicted object = next recorded on "
                f"{np.mean(hits):.2f} of {len(hits)} frames")
        return values.astype(np.float32), mask.astype(bool), note

    def provenance(self):
        return {"prior": str(self.args.prior.resolve()), "prior_config": self.model.config,
                "lang": str(self.args.lang.resolve()), "kind": self.args.kind,
                "k": self.k, "support_seed": self.args.seed,
                "support_pool": "meta/pfn_support.npz", "instruction": self.A.INSTRUCTION,
                "eef_to_grip": self.A.EEF_TO_GRIP,
                "diagnostic_next_object_agreement": float(np.mean(self.hits))}

    def save_support(self, path):
        demos = [self.demos[i] for i in self.pool_index]
        np.savez_compressed(path, episode_index=np.array(self.pool_index),
                            seed=np.array([d.seed for d in demos]),
                            **{f: np.stack([getattr(d, f) for d in demos])
                               for f in ("feats", "types", "rows", "event", "gripper", "flag")})


def _learned_source(args, source, by_index):
    return _LearnedSource(args, source, by_index)


def main() -> int:
    import numpy as np
    import pandas as pd

    from oct_vla.data.future_keyframes import FUTURE_DIM, FUTURE_ROWS, oracle_future_keyframes
    from oct_vla.tasks.shelf_restock.keyframes import episode_keyframes
    from oct_vla.tasks.shelf_restock.spec import spec_for_rule
    from oct_vla.data.store import read_episode
    from oct_vla.policies.conditioning.future import FUTURE_KEYFRAME, FUTURE_MASK
    from project_dataset_view import link_tree

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Dataset view to extend")
    parser.add_argument("--output", type=Path, help="Default: <source>_fk")
    parser.add_argument("--collection", required=True, type=Path,
                        help="Canonical collection root holding seed_<n>/<source_episode>/")
    parser.add_argument("--keyframes", choices=("oracle", "pfn"), default="oracle")
    parser.add_argument("--kind", choices=("keyframe", "latent"), default="keyframe")
    parser.add_argument("--prior", type=Path, help="Prior run directory (pfn only)")
    parser.add_argument("--lang", type=Path, help="Instruction embeddings (pfn only)")
    parser.add_argument("--k", type=int, default=None, help="Support demos per query")
    parser.add_argument("--seed", type=int, default=0, help="Support sampling seed")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.keyframes == "oracle" and (args.kind != "keyframe" or args.prior is not None):
        raise SystemExit("the oracle writes keyframe rows and takes no prior")
    if args.keyframes == "pfn" and (args.prior is None or args.lang is None):
        raise SystemExit("--keyframes pfn needs --prior and --lang")
    source = args.source.resolve()
    if args.keyframes == "oracle":
        default_name = source.name + "_fk"
    else:
        suffix = {"keyframe": "pfn", "latent": "pfnlat"}[args.kind]
        if args.kind == "keyframe" and "goal" in args.prior.name:
            suffix = "pfngoal"
        default_name = f"{source.name}_{suffix}_{args.prior.name.rsplit('-', 1)[-1]}"
    output = (args.output or source.with_name(default_name)).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    manifest = json.loads((source / "octvla_episode_manifest.json").read_text())
    by_index = {int(e["lerobot_episode_index"]): e for e in manifest["episodes"]}

    learned = _learned_source(args, source, by_index) if args.keyframes == "pfn" else None
    output.mkdir(parents=True)
    linked = link_tree(source / "videos", output / "videos")
    shutil.copytree(source / "meta", output / "meta")
    for name in ("split_manifest.json", "octvla_episode_manifest.json"):
        shutil.copy2(source / name, output / name)

    real_rows, masks, episodes_done, frames_done = [], [], 0, 0
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pd.read_parquet(path)
        keyframe_column = [None] * len(table)
        mask_column = [None] * len(table)
        for episode_index, rows in table.groupby("episode_index").groups.items():
            rows = table.loc[rows].sort_values("frame_index")
            entry = by_index[int(episode_index)]
            episode = read_episode(args.collection / f"seed_{entry['seed']}" / entry["source_episode"])
            if len(rows) != len(episode.samples):
                raise SystemExit(f"episode {episode_index}: {len(rows)} dataset frames, "
                                 f"{len(episode.samples)} recorded")
            tokens = np.stack([np.stack(t) for t in rows["observation.entity_tokens"]])
            entity_mask = np.stack([np.asarray(m) for m in rows["observation.entity_mask"]]) > 0.5
            for frame in (0, len(rows) // 2, len(rows) - 1):
                movable = tokens[frame][entity_mask[frame] & (tokens[frame][:, 13] > 0.5)][:, :3]
                scene = np.array([o.pose.position for o in episode.samples[frame].scene.objects])
                if not np.allclose(np.sort(movable, axis=0), np.sort(scene, axis=0), atol=1e-5):
                    raise SystemExit(f"episode {episode_index} frame {frame}: tokens and recorded "
                                     "scene disagree; the frames are not aligned")
            if learned is not None:
                keyframes, mask, note = learned(int(episode_index), episode,
                                                rows["frame_index"].to_numpy())
            else:
                keyframes, mask = episode_keyframes(episode,
                                                    spec_for_rule(episode.metadata["rule"]))
                lags = _switch_lags(keyframes, mask, *oracle_future_keyframes(episode))
                if lags is None or not all(0 <= lag <= MAX_LAG for lag in lags):
                    raise SystemExit(f"episode {episode_index}: oracle events disagree with the "
                                     f"phase labels (lags {lags}); refusing to export it")
                note = (f"{len(lags)} events (lag {min(lags)}-{max(lags)}), "
                        f"keyframe on {int(mask.all(1).sum())} frames")
                keyframes = keyframes[rows["frame_index"].to_numpy()]
                mask = mask[rows["frame_index"].to_numpy()]
            for position, index in enumerate(rows.index):
                at = table.index.get_loc(index)
                keyframe_column[at] = keyframes[position].tolist()
                mask_column[at] = mask[position].astype(np.float32).tolist()
            real_rows.append(keyframes[mask])
            masks.append(mask.astype(np.float32))
            episodes_done += 1
            frames_done += len(rows)
            print(f"episode {episode_index} ({entry['source_episode']}): {len(rows)} frames, "
                  f"{note}", flush=True)
        if any(value is None for value in keyframe_column):
            raise SystemExit(f"{path}: a row got no keyframe")
        table[FUTURE_KEYFRAME] = keyframe_column
        table[FUTURE_MASK] = mask_column
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target, index=False)

    real = np.concatenate(real_rows)
    if learned is not None and not np.isfinite(real).all():
        raise SystemExit("the prior produced non-finite values")
    if len(real) == 0:
        raise SystemExit("no frame has a keyframe: the phase labels are wrong")
    info = json.loads((output / "meta" / "info.json").read_text())
    rows_n, dim = (FUTURE_ROWS, FUTURE_DIM) if learned is None else learned.shape
    info["features"][FUTURE_KEYFRAME] = {"dtype": "float32", "shape": [rows_n, dim],
                                         "names": None}
    info["features"][FUTURE_MASK] = {"dtype": "float32", "shape": [rows_n], "names": None}
    info["derived_from"] = source.name
    info["future_keyframe"] = {"source": args.keyframes, "kind": args.kind,
                               "collection": str(args.collection.resolve()),
                               "rows": rows_n, "dim": dim}
    if learned is not None:
        info["future_keyframe"].update(learned.provenance())
        learned.save_support(output / "meta" / "pfn_support.npz")
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats[FUTURE_KEYFRAME] = _stats(real)
    stats[FUTURE_MASK] = _stats(np.concatenate(masks))
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")
    print(f"\nwrote {output}: {episodes_done} episodes, {frames_done} frames, "
          f"{len(real)} keyframe rows, {linked} video files linked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
