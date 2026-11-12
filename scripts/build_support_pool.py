#!/usr/bin/env python
"""A support-demo pool for the learned future source, from one rule's collection.

The pool a PFN view carries (``meta/pfn_support.npz``) holds its own rule's training
demos. Evaluating a policy with *another* rule's demos (the support-swap intervention:
does the robot change what it does when only the demonstrations change?) needs that
rule's pool in the same format. Training seed blocks only (``prior/finetune.split_of``).

    build_support_pool.py --collection rightmost=$C/rightmost/full_run/three_object \\
        --out $P/support/rightmost_train.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from oct_vla.data.entity_tokens import shelf_support_entities
    from oct_vla.data.store import read_episode
    from oct_vla.prior import shelf_adapter as A
    from oct_vla.prior.finetune import split_of
    from oct_vla.tasks.shelf_restock.spec import spec_for_rule

    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True, help="RULE=PATH")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    rule, path = args.collection.split("=", 1)
    supports = shelf_support_entities(spec_for_rule(rule))
    demos = []
    for d in sorted(p for p in Path(path).glob("seed_*/*") if p.is_dir()):
        seed = int(d.parent.name.split("_")[1])
        try:
            if split_of(seed) != "train":
                continue
        except ValueError:
            continue
        episode = read_episode(d)
        if episode.success:
            if episode.metadata.get("rule", rule) != rule:
                raise SystemExit(f"{d} is a {episode.metadata['rule']} episode, not {rule}")
            demos.append(A.support_demo(episode, supports))
    if not demos:
        raise SystemExit("no successful training-seed episodes")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, episode_index=np.arange(len(demos)),
                        seed=np.array([d.seed for d in demos]), rule=np.array(rule),
                        **{f: np.stack([getattr(d, f) for d in demos])
                           for f in ("feats", "types", "rows", "event", "gripper", "flag")})
    print(f"wrote {args.out}: {len(demos)} {rule} demos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
