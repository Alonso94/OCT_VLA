#!/usr/bin/env python
"""Closed-loop pilot without a learned executor (plan stage 2b): a client of the running
evaluation server (scripts/robotwin_eval_server.py, started for the assigned rule).

    bpfn_online.py --controller pfn --gripper observable --run RUN --support EPISODE_DIR \\
        --seeds 800-809 --profile four_object --out OUT.json [--k 1] [--post-success-steps 30]
    bpfn_online.py --controller expert_window --gripper oracle ...   (privileged bridge check)

Every result names its controller, gripper source, support, checkpoint and whether
privileged server information was used. Traces (commanded and measured wrists, gripper
commands, objects) go to --trace-dir as one .npz per episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main() -> int:
    from oct_vla.prior.pose_window import online as On
    from oct_vla.serve.client import ShelfRestockEvalClient

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--controller", choices=("pfn", "expert_window", "hold"), required=True)
    ap.add_argument("--gripper", choices=("observable", "oracle"), required=True)
    ap.add_argument("--run", type=Path, help="PFN checkpoint directory (controller pfn)")
    ap.add_argument("--support", type=Path, help="recorded support episode (controller pfn)")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--profile", default="four_object")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--post-success-steps", type=int, default=30)
    ap.add_argument("--k", type=int, default=1, help="executed steps per prediction")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trace-dir", type=Path)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    privileged = args.controller == "expert_window" or args.gripper == "oracle"

    if args.controller == "pfn":
        if args.run is None or args.support is None:
            raise SystemExit("--controller pfn needs --run and --support")
        from oct_vla.prior.pose_window import train as Tr
        from oct_vla.prior.pose_window.real import read_recording

        model = Tr.load(args.run, args.device)
        controller = On.PFNController(model, read_recording(args.support), k=args.k,
                                      device=args.device)
    elif args.controller == "expert_window":
        controller = On.ExpertWindowController()
    else:
        controller = On.HoldController()

    results = []
    with ShelfRestockEvalClient(args.host, args.port) as client:
        for seed in _seeds(args.seeds):
            gripper = On.ObservableGripper() if args.gripper == "observable" else On.OracleGripper()
            r = On.run_episode(client, controller, gripper, seed=seed, profile=args.profile,
                               max_steps=args.max_steps,
                               post_success_steps=args.post_success_steps,
                               privileged=privileged)
            if args.trace_dir is not None:
                args.trace_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.trace_dir / f"{args.profile}_seed{seed}.npz",
                                    objects=r["objects"], **r.pop("trace"))
            else:
                r.pop("trace")
            r.pop("objects")
            if args.controller == "pfn":
                r["windows"] = len(controller.windows)
            results.append(r)
            print(f"seed {seed}: success {r['success']} transfers {r['transfers_completed']} "
                  f"reason {r['reason']} distractors_preserved {r['distractors_preserved']}",
                  flush=True)
    report = {"tag": args.tag, "controller": args.controller, "gripper": args.gripper,
              "privileged": privileged, "run": str(args.run) if args.run else None,
              "support": str(args.support) if args.support else None, "k": args.k,
              "profile": args.profile, "seeds": args.seeds, "max_steps": args.max_steps,
              "post_success_steps": args.post_success_steps,
              "success_rate": float(np.mean([r["success"] for r in results])),
              "episodes": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
