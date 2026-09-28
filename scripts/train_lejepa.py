#!/usr/bin/env python
"""Train the LeJEPA object-crop encoder (oct_vla.perception.lejepa.train).

Reads `<crops>/train` and `<crops>/val` from scripts/build_object_crops.py and
writes `lejepa.pt`, `config.json`, `pca32.npz`, `log.jsonl` and `probe.json`
to `--out`, which must not already hold a run.

    PYTHONPATH=src $OCTVLA_POLICY_PYTHON scripts/train_lejepa.py \\
        --out $HPCVAULT/octvla-lejepa/runs/s0 --steps 20000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from oct_vla.perception.lejepa.train import train


def main(argv: list[str] | None = None) -> int:
    vault = Path(os.environ.get("HPCVAULT", "")) / "octvla-lejepa"
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--crops", type=Path, default=vault / "crops_v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-sigreg", type=float, default=0.05)
    parser.add_argument("--target-views", type=int, default=3)
    parser.add_argument("--slices", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("--device cuda requested but no GPU is visible")
    train(
        args.crops,
        args.out,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        lambda_sigreg=args.lambda_sigreg,
        device=args.device,
        seed=args.seed,
        target_views=args.target_views,
        slices=args.slices,
        log_every=args.log_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
