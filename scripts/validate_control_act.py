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
    def object_parameters():
        return {
            name: parameter
            for name, parameter in model.named_parameters()
            if "object_conditioning" in name
        }

    encoder_names = sorted(
        name for name in object_parameters() if ".embedding." in name
    )
    if not encoder_names:
        raise RuntimeError("no entity-encoder parameters found under object_conditioning")
    before = {name: p.detach().clone() for name, p in object_parameters().items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [n for n, p in object_parameters().items() if id(p) not in in_optimizer]
    if missing:
        raise RuntimeError(f"object parameters absent from the optimizer: {missing}")

    losses = []
    first_step_grads = None
    for step in range(6):
        model.train()
        optimizer.zero_grad()
        loss, _ = model(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        grads = {
            name: float(p.grad.abs().sum()) if p.grad is not None else 0.0
            for name, p in object_parameters().items()
        }
        for name, p in object_parameters().items():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(f"nonfinite gradient on {name} at step {step}")
        if first_step_grads is None:
            first_step_grads = grads
        optimizer.step()
        losses.append(loss.item())

    # The zero-init property, read off the first backward rather than asserted:
    # with V_z = 0 the branch output is zero, so only V_z can receive gradient.
    step0_moved = sorted(name for name, value in first_step_grads.items() if value > 0)
    if any(not name.endswith(("to_v.weight", "to_v.bias")) for name in step0_moved):
        raise RuntimeError(f"step 0 gradient reached more than V_z: {step0_moved}")

    # The check `entity_sensitivity` cannot make: the encoder could stay at its
    # random init forever while only K/V adapt, and the output would still move
    # when the entities did.
    final_grads = {
        name: float(p.grad.abs().sum()) if p.grad is not None else 0.0
        for name, p in object_parameters().items()
    }
    starved = [name for name in encoder_names if final_grads[name] == 0.0]
    if starved:
        raise RuntimeError(f"entity encoder received no gradient after 6 steps: {starved}")
    frozen = [
        name
        for name in encoder_names
        if torch.equal(before[name], object_parameters()[name].detach())
    ]
    if frozen:
        raise RuntimeError(f"entity encoder parameters never moved: {frozen}")
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
        "object_parameters": len(object_parameters()),
        "entity_encoder_parameters": len(encoder_names),
        "step0_gradient_reached": step0_moved,
        "entity_encoder_gradient_after_6_steps": {
            name: final_grads[name] for name in encoder_names
        },
        "entity_encoder_moved": True,
        "entity_sensitivity": sensitivity,
        "reload_parity": True,
        "scientific_performance_evaluation": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
