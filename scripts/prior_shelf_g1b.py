#!/usr/bin/env python
"""Gate G1b: the frozen synthetic prior, offline, on recorded shelf episodes.

The question: does the prior name the right next object for the demonstrated order on
real shelf scenes, from support demos alone? Queries and supports go through
``prior/shelf_adapter.py``, the path the policy will use. The reference is each frame's
next grasp or place, read off the phase labels (``data/future_keyframes.py``).

Conditions, per prior run and per query collection:
- ``same``: K support demos of the query's own rule (other episodes);
- ``other``: K support demos of the other rule;
- ``none``: no support.

All use the ambiguous instruction, so only the support carries the rule.

Reported:
- next-object (pointer), event and gripper accuracy over every evaluated frame, split
  by the reference event (grasp / place);
- position errors against the reference rows;
- the frame-0 contrast: at frame 0 every object is on the lower shelf, so both rules'
  first object is defined (leftmost = min x, rightmost = max x). It asks whether, on the
  same scene, the prior follows the support it is given.

    prior_shelf_g1b.py --collection leftmost=$C/leftmost/full_run/three_object \\
        --collection rightmost=$C/rightmost/full_run/three_object \\
        --runs $P/runs/P1-interaction-s{0,1,2} $P/runs/P2-goal-s{0,1,2} \\
        --lang $P/language/minilm.pt --out $P/results/g1b-shelf-v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RULE_SELECT = {"leftmost": np.argmin, "rightmost": np.argmax}
#: Phase label -> stage of a transfer, for the per-stage breakdown only (evaluation side;
#: nothing here reaches the prior). The compactor's push has no counterpart in the prior.
STAGES = {"open": "to_grasp", "pregrasp": "to_grasp", "descend": "to_grasp", "close": "to_grasp",
          "lift": "carry", "preplace": "carry", "place": "carry", "release": "carry",
          "retreat": "after_place", "park mover": "after_place"}


def _stage(phase: str) -> str:
    if phase.startswith("compact") or phase == "park compactor":
        return "compaction"
    return STAGES.get(phase, "other")


def _episodes(path: Path):
    from oct_vla.data.store import read_episode

    eps = [read_episode(d) for d in sorted(p for p in path.glob("seed_*/*") if p.is_dir())]
    return [e for e in eps if e.success]


def _reference(episode):
    """Per frame: (track, kind, side) of the first grasp or place strictly after it, or None."""
    from oct_vla.data.future_keyframes import interaction_events

    events = interaction_events(episode)
    out, k = [], 0
    for t in range(len(episode.samples)):
        while k < len(events) and events[k].frame <= t:
            k += 1
        out.append(events[k] if k < len(events) else None)
    return out


#: Confidence bins of the reliability table.
BINS = (0.0, 0.5, 0.8, 0.95, 1.0 + 1e-9)


def _calibration(records) -> dict:
    """Is the prior right as often as it is sure? Accuracy against mean confidence per
    bin, and the expected calibration error. The confidence gate assumes this holds."""
    if not records:
        return {}
    conf = np.array([c for c, _ in records])
    hit = np.array([h for _, h in records], float)
    table, ece = [], 0.0
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (conf >= lo) & (conf < hi)
        if sel.any():
            table.append({"lo": lo, "hi": min(hi, 1.0), "n": int(sel.sum()),
                          "acc": float(hit[sel].mean()), "conf": float(conf[sel].mean())})
            ece += sel.mean() * abs(hit[sel].mean() - conf[sel].mean())
    return {"bins": table, "ece": float(ece), "mean_conf": float(conf.mean()),
            "acc": float(hit.mean())}


def _acc(hits) -> dict:
    hits = np.asarray(hits, bool)
    return {"acc": float(hits.mean()) if len(hits) else None, "n": int(len(hits))}


def main() -> int:
    import torch

    from oct_vla.data.entity_tokens import shelf_support_entities
    from oct_vla.data.future_keyframes import oracle_future_keyframes
    from oct_vla.prior import shelf_adapter as A
    from oct_vla.prior.data import load_language
    from oct_vla.tasks.shelf_restock.spec import spec_for_rule

    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", action="append", required=True,
                        help="RULE=PATH to a canonical collection of that rule")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--lang", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=A.K_SUPPORT)
    parser.add_argument("--stride", type=int, default=5, help="Evaluate every n-th frame")
    parser.add_argument("--query-split", choices=("all", "val"), default="all",
                        help="all: every episode, supports leave-one-out (a frozen prior). "
                             "val: validation-seed queries, training-seed supports (the "
                             "held-out check for a prior fine-tuned on the training seeds)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--supports-from-run", action="store_true",
                        help="Draw each run's support demos only from the episodes it was "
                             "trained on (its config.json), for the low-data curve: a run "
                             "must not be shown demonstrations it was not given")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    torch.set_num_threads(8)
    lang = load_language(args.lang)
    collections = dict(c.split("=", 1) for c in args.collection)
    if set(collections) != set(RULE_SELECT):
        raise SystemExit(f"need one collection per rule {sorted(RULE_SELECT)}")

    episodes, demos = {}, {}
    for rule, path in collections.items():
        supports = shelf_support_entities(spec_for_rule(rule))
        every = _episodes(Path(path))
        if args.query_split == "val":
            from oct_vla.prior.finetune import split_of

            episodes[rule] = [e for e in every if split_of(int(e.seed)) == "val"]
            pool = [e for e in every if split_of(int(e.seed)) == "train"]
        else:
            episodes[rule], pool = every, every
        demos[rule] = [A.support_demo(e, supports) for e in pool]
        print(f"{rule}: {len(episodes[rule])} query episodes, {len(pool)} support demos",
              flush=True)

    results = {}
    meta = {"args": {k: str(v) for k, v in vars(args).items()},
            "episodes": {r: [int(e.seed) for e in eps] for r, eps in episodes.items()},
            "adapter": {"eef_to_grip": A.EEF_TO_GRIP, "instruction": A.INSTRUCTION}}
    partial = args.out.with_name(args.out.name + ".partial")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        model = A.load_prior(run, args.device)
        run_demos = demos
        if args.supports_from_run:
            trained = json.loads((run / "config.json").read_text()).get("episodes")
            if trained:  # a pretrained-only run has none: it saw no real demos
                run_demos = {r: [d for d in demos[r] if d.seed in set(trained.get(r, []))]
                             for r in demos}
                if any(not v for v in run_demos.values()):
                    raise SystemExit(f"{run.name}: no support demo left for a rule")
            print(f"{run.name}: support pools "
                  f"{ {r: len(v) for r, v in run_demos.items()} }", flush=True)
        name = run.name
        for rule in RULE_SELECT:
            other = next(r for r in RULE_SELECT if r != rule)
            supports = shelf_support_entities(spec_for_rule(rule))
            for condition in ("same", "other", "none"):
                rng = np.random.default_rng([args.seed, len(rule), len(condition)])
                calibration: dict[str, list] = {}
                hits = {k: [] for k in ("pointer", "pointer_grasp", "pointer_place", "event",
                                        "gripper", "first")}
                err = {k: [] for k in ("obj_cm", "grip_cm", "goal_cm")}
                for i, ep in enumerate(episodes[rule]):
                    if condition == "none":
                        support = []
                    else:
                        pool = [d for j, d in enumerate(run_demos[rule if condition == "same"
                                                                  else other])
                                if not (condition == "same" and j == i
                                        and args.query_split == "all")]
                        k = min(args.k, len(pool))
                        support = [pool[j] for j in rng.choice(len(pool), k, replace=False)]
                    ref = _reference(ep)
                    ref_rows, ref_mask = oracle_future_keyframes(ep)
                    frames = [t for t in range(0, len(ep.samples), args.stride)
                              if ref[t] is not None]
                    s0 = ep.samples[0]
                    qs = [A.query(s0.scene, s0.observation.eef, ep.samples[t].scene,
                                  ep.samples[t].observation.eef, supports) for t in frames]
                    out = A.infer(model, A.pack(support, qs), lang, args.device)
                    for n, t in enumerate(frames):
                        e = ref[t]
                        right = A.predicted_track(out, n, qs[n]) == e.track_id
                        hits["pointer"].append(right)
                        hits[f"pointer_{e.kind}"].append(right)
                        stage = _stage(ep.samples[t].phase)
                        # Correct = right object (and, for an interaction prior, right
                        # event and arm): what the gate's confidence is a product over.
                        whole = right
                        if "event" in out:
                            whole = (right and ("grasp", "place", "regrasp", "release")[
                                int(out["event"][n].argmax())] == e.kind
                                and ("left", "right")[int(out["gripper"][n].argmax())] == e.side)
                        record = (A.confidence(out, n), bool(whole))
                        calibration.setdefault("all", []).append(record)
                        calibration.setdefault(stage, []).append(record)
                        hits.setdefault(f"pointer@{stage}", []).append(right)
                        if "event" in out:
                            hits["event"].append(
                                ("grasp", "place", "regrasp", "release")[int(out["event"][n].argmax())]
                                == e.kind)
                            hits["gripper"].append(("left", "right")[int(out["gripper"][n].argmax())]
                                                   == e.side)
                            hits.setdefault(f"event@{stage}", []).append(hits["event"][-1])
                            hits.setdefault(f"gripper@{stage}", []).append(hits["gripper"][-1])
                            rows, _ = A.keyframe_rows(out, n, qs[n])
                            for r, key in enumerate(("obj_cm", "grip_cm", "goal_cm")):
                                err[key].append(100 * float(np.linalg.norm(
                                    rows[r, :3] - ref_rows[t, r, :3])))
                        else:
                            rows, _ = A.keyframe_rows(out, n, qs[n])
                            err["goal_cm"].append(100 * float(np.linalg.norm(
                                rows[2, :3] - ref_rows[t, 2, :3])))
                    # Frame 0: does the pointer name the support rule's first object?
                    want_rule = rule if condition != "other" else other
                    xs = [s0.scene.get(tid).pose.position[0] for tid in qs[0].order]
                    first = qs[0].order[int(RULE_SELECT[want_rule](xs))]
                    hits["first"].append(frames[0] == 0 and A.predicted_track(out, 0, qs[0]) == first)
                key = f"{name}/{rule}/{condition}"
                results[key] = {**{f"calibration@{k}": _calibration(v)
                                   for k, v in calibration.items()},
                                **{k: _acc(v) for k, v in hits.items() if v},
                                **{k: {"median": float(np.median(v)), "mean": float(np.mean(v))}
                                   for k, v in err.items() if v}}
                print(key, {k: round(v["acc"], 3) for k, v in results[key].items()
                            if isinstance(v, dict) and v.get("acc") is not None}, flush=True)
        partial.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    args.out.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    partial.unlink(missing_ok=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
