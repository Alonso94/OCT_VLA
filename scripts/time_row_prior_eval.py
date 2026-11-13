#!/usr/bin/env python
"""Score Phase-2 priors on the fixed time-row packs: B1 / A1 runs and A0 (v1) checkpoints.

Every model sees the same items. Support conditions:
- `correct`;
- `shuffled`: another item's support, rolled by 7 so a conflict partner never supplies
  it;
- `none`.

Metrics, over valid query targets:
- **group:** event accuracy, pointer accuracy, and "keyframe" (event + pointer + the
  participating arm all right);
- **arms:** both arms' participation right, both arms' event right;
- **positions (cm):** the acting arm's and the idle arm's target position, the goal;
- **v2:** approach-axis angle error of the acting arm (degrees);
- **contrast:** both conflict partners' keyframe right.

A0 predicts one acting arm and no idle-arm position, so those entries are NaN.

    time_row_prior_eval.py --pack $P/eval/time_row_v1 --runs $P/runs_timerow/B1-* ... \\
        --keyframe-runs $P/runs/P1-interaction-s0 --lang $P/language/minilm.pt --out res.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONDITIONS = ("correct", "shuffled", "none")
SUPPORT_PREFIX = ("s_",)


def apply_condition(inputs: dict, condition: str) -> dict:
    out = dict(inputs)
    if condition == "correct":
        return out
    keys = [k for k in inputs if k.startswith(SUPPORT_PREFIX) or k == "demo_valid"]
    if condition == "shuffled":
        for k in keys:
            out[k] = np.roll(inputs[k], 7, axis=0)
    elif condition == "none":
        out["demo_valid"] = np.zeros_like(inputs["demo_valid"])
        out["s_row_valid"] = np.zeros_like(inputs["s_row_valid"])
    else:
        raise ValueError(condition)
    return out


def predict_paired(kind, model, inputs, targets, lang, device, chunk=128) -> dict:
    """Paired predictions in metric units (numpy)."""
    import torch

    from oct_vla.prior import model as M
    from oct_vla.prior.time_row_model import to_torch_time_row
    from oct_vla.prior.time_row_v1_bridge import a0_paired_predictions, to_keyframe_batch

    parts = []
    n = len(inputs["q_feats"])
    for s in range(0, n, chunk):
        sub_in = {k: v[s:s + chunk] for k, v in inputs.items()}
        sub_tg = {k: v[s:s + chunk] for k, v in targets.items()}
        with torch.no_grad():
            if kind == "a0":
                kb = to_keyframe_batch(sub_in, sub_tg)
                parts.append(a0_paired_predictions(M.predict(model, kb, lang, device)))
                continue
            if kind == "keyframe_paired":
                kb = to_keyframe_batch(sub_in, sub_tg)
                out = model.forward_paired(M.to_torch(kb, lang, device), teacher=False)
            else:
                out = model(to_torch_time_row(sub_in, None, lang, device), teacher=False)
            parts.append(_to_metric(out))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _to_metric(out) -> dict:
    from oct_vla.prior.model import POS_CENTRE, POS_SCALE

    res = {}
    for k, v in out.items():
        v = v.detach().float().cpu().numpy()
        if k == "pos":
            v = v * POS_SCALE + np.asarray(POS_CENTRE)
        elif k == "goal":
            v = v.copy()
            v[..., :3] = v[..., :3] * POS_SCALE + np.asarray(POS_CENTRE)
        res[k] = v
    res["participation"] = (res["participation"] > 0).astype(np.float32)
    res["arm_event"] = res["arm_event"].argmax(-1)
    res["group_event"] = res["group_event"].argmax(-1)
    res["group_pointer"] = res["group_pointer"].argmax(-1)
    return res


def score(pred: dict, targets: dict, metas: list) -> dict:
    valid = np.asarray(targets["target_valid"]) > 0.5
    if "goal_only" in pred:  # predicts object and goal only; nothing else is scored
        ptr = pred["group_pointer"] == targets["g_object"]
        goal = np.linalg.norm(pred["goal"][:, :3] - targets["g_goal"][:, :3], axis=-1)
        out = {"n": int(valid.sum()), "group_pointer": float(np.mean(ptr[valid])),
               "goal_cm": float(np.median(goal[valid]) * 100)}
        pairs = [(i, m["pair"]) for i, m in enumerate(metas) if m["pair"] > i]
        if pairs:
            out["contrast_pointer"] = float(np.mean([ptr[i] and ptr[j] for i, j in pairs]))
        return out
    part_t = np.asarray(targets["y_participation"]) > 0.5
    acting = part_t.argmax(-1)
    rows = np.arange(len(acting))
    ev_ok = pred["group_event"] == targets["g_event"]
    ptr_ok = pred["group_pointer"] == targets["g_object"]
    arm_ok = (pred["participation"] > 0.5) == part_t
    kf_ok = ev_ok & ptr_ok & arm_ok.all(-1)
    act_err = np.linalg.norm(pred["pos"][rows, acting] - targets["y_pos"][rows, acting], axis=-1)
    idle_err = np.linalg.norm(pred["pos"][rows, 1 - acting] - targets["y_pos"][rows, 1 - acting],
                              axis=-1)
    goal_err = np.linalg.norm(pred["goal"][:, :3] - targets["g_goal"][:, :3], axis=-1)

    def acc(x):
        return float(np.mean(x[valid])) if valid.any() else None

    def cm(x):
        x = x[valid]
        return None if np.isnan(x).all() else float(np.nanmedian(x) * 100)

    out = {"n": int(valid.sum()), "group_event": acc(ev_ok), "group_pointer": acc(ptr_ok),
           "keyframe": acc(kf_ok), "arm_participation": acc(arm_ok.all(-1)),
           "arm_event": acc((pred["arm_event"] == targets["y_event"]).all(-1)),
           "acting_pos_cm": cm(act_err), "goal_cm": cm(goal_err),
           # A0 predicts no idle-arm position (NaN); never score its acting-arm guess there.
           "idle_pos_cm": None if np.isnan(pred["pos"]).any() else cm(idle_err)}
    if "rot" in pred and "y_rot" in targets:
        a = pred["rot"][rows, acting, :3]
        a = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-6)
        cos = np.clip((a * targets["y_rot"][rows, acting, :3]).sum(-1), -1, 1)
        out["acting_approach_deg"] = float(np.median(np.degrees(np.arccos(cos))[valid]))
    pairs = [(i, m["pair"]) for i, m in enumerate(metas) if m["pair"] > i]
    if pairs:
        out["contrast"] = float(np.mean([kf_ok[i] and kf_ok[j] for i, j in pairs]))
        out["contrast_pairs"] = len(pairs)
    return out


def main() -> int:
    import torch

    from oct_vla.prior.data import load_language
    from oct_vla.prior.shelf_adapter import load_prior
    from oct_vla.prior.time_row_batches import load_time_row_eval_set
    from oct_vla.prior.time_row_model import TimeRowPFN
    from oct_vla.prior.time_row_v1_bridge import KeyframePairedPFN

    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "composition", "shelf_family"])
    ap.add_argument("--runs", nargs="*", type=Path, default=[])
    ap.add_argument("--keyframe-runs", nargs="*", type=Path, default=[], help="A0: v1 KeyframePFN")
    ap.add_argument("--lang", type=Path, default=None)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    lang = load_language(args.lang) if args.lang else None

    models = []
    for run in args.runs:
        blob = torch.load(run / "model.pt", map_location="cpu")
        kind = blob.get("kind", "timerow")
        cls = KeyframePairedPFN if kind == "keyframe_paired" else TimeRowPFN
        cfg = {k: v for k, v in blob["config"].items() if k not in ("paired", "schema")}
        model = cls(**cfg)
        model.load_state_dict(blob["state"])
        uses_lang = bool(blob["config"].get("lang_dim"))
        models.append((run.name, kind, model.to(args.device).eval(), lang if uses_lang else None))
    for run in args.keyframe_runs:
        model = load_prior(run, args.device)
        models.append((run.name, "a0", model, lang if model.config.get("lang_dim") else None))

    results = {}
    for split in args.splits:
        inputs, targets, metas = load_time_row_eval_set(args.pack / f"{split}.npz")
        for condition in args.conditions:
            cond_in = apply_condition(inputs, condition)
            for name, kind, model, model_lang in models:
                pred = predict_paired(kind, model, cond_in, targets, model_lang, args.device)
                key = f"{split}/{name}/{condition}"
                results[key] = score(pred, targets, metas)
                print(key, {k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in results[key].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                    "results": results}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
