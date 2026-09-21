#!/usr/bin/env python
"""Short train/save/reload/inference check on the real ACT implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature

    from oct_vla.policies.control_act import ControlACTConfig, ControlACTPolicy

    torch.manual_seed(71)
    config = ControlACTConfig(
        device=args.device,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=2,
        chunk_size=4,
        n_action_steps=2,
        use_vae=False,
        dropout=0,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.entity_tokens": PolicyFeature(type=FeatureType.STATE, shape=(8, 17)),
            "observation.entity_mask": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
    )
    model = ControlACTPolicy(config).to(args.device)
    batch = {
        "observation.state": torch.randn(2, 16, device=args.device),
        "observation.images.head": torch.rand(2, 3, 64, 64, device=args.device),
        "observation.entity_tokens": torch.randn(2, 8, 17, device=args.device),
        "observation.entity_mask": torch.ones(2, 8, dtype=torch.bool, device=args.device),
        "action": torch.randn(2, 4, 16, device=args.device),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool, device=args.device),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for _ in range(6):
        model.train()
        optimizer.zero_grad()
        loss, _ = model(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    expected = model.predict_action_chunk(batch)
    args.output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output / "checkpoint")
    loaded = ControlACTPolicy.from_pretrained(args.output / "checkpoint").to(args.device)
    torch.testing.assert_close(expected, loaded.predict_action_chunk(batch))
    altered = dict(batch)
    altered["observation.entity_tokens"] = batch["observation.entity_tokens"] + 2
    sensitivity = (expected - loaded.predict_action_chunk(altered)).abs().max().item()
    if sensitivity <= 1e-7:
        raise RuntimeError("trained policy is insensitive to changed entities")
    result = {
        "device": args.device,
        "losses": losses,
        "entity_sensitivity": sensitivity,
        "reload_parity": True,
        "scientific_performance_evaluation": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
