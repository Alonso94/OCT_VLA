#!/usr/bin/env python
"""Build a backbone end to end and run one batch through it, before any training.

The failures this catches are the expensive kind, because they surface late:

* a plugin without `processor_<type>.py` trains for hours and then cannot be
  evaluated, because `make_pre_post_processors` is what evaluation calls;
* a feature-name mismatch is silent, since every camera shares a shape and
  dtype, so the wrong view is fed and only the success rate says so;
* LoRA with no target modules attaches to nothing and still trains the object
  path, which looks like a working run;
* the object residual stays at its zero init if the conditioning module is
  wired to the wrong place, and the loss falls normally either way.

So this constructs the policy from its checkpoint, builds both processor
pipelines, runs a forward and a chunk prediction on one real batch, and reports
the object path's state -- in minutes, on one GPU, against the dataset the run
would actually use.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check(label: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if passed else 'FAIL'}] {label}{f' -- {detail}' if detail else ''}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-type", required=True,
                        help="Registered type, e.g. control_xvla or masked_pi05.")
    parser.add_argument("--pretrained", default="",
                        help="Base checkpoint. Omit to build from config alone.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--peft", action="store_true",
                        help="Also wrap with LoRA and check the targets attach.")
    parser.add_argument("--injection-mode", default="",
                        help="Override config.object_injection_mode "
                             "(controlvla or pooled) before the policy is built.")
    args = parser.parse_args()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    import oct_vla.policies  # noqa: F401  -- registers the plugin types
    from oct_vla.data.policy_inputs import rename_map_for

    print(f"policy     : {args.policy_type}")
    print(f"checkpoint : {args.pretrained or '(config only)'}")
    print(f"dataset    : {args.dataset_root}")

    ok = True
    ok &= check("type is registered", args.policy_type in PreTrainedConfig._choice_registry)
    if not ok:
        print(f"\nknown: {sorted(PreTrainedConfig._choice_registry)}")
        return 1

    rename_map = rename_map_for(args.policy_type)
    ok &= check("rename map registered", True, f"{len(rename_map)} entries")

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    action_dim = metadata.features["action"]["shape"][0]
    state_dim = metadata.features["observation.state"]["shape"][0]
    cameras = [k for k in metadata.features if k.startswith("observation.images.")]
    print(f"           : action {action_dim}d, state {state_dim}d, {len(cameras)} cameras")

    # Mirrors how training loads a plugin: `--policy.type` plus
    # `--policy.pretrained_path`, so features are built from *our* dataset and
    # the weights are loaded into that. Reading the config out of the checkpoint
    # instead would take another embodiment's camera names and action width --
    # and several of these checkpoints are not LeRobot policy configs at all.
    # X-VLA's config.json is a transformers-style one carrying `model_type`
    # rather than `type`, so `PreTrainedConfig.from_pretrained` refuses it.
    config = PreTrainedConfig._choice_registry[args.policy_type]()
    if args.injection_mode:
        if not hasattr(config, "object_injection_mode"):
            print(f"  [FAIL] {args.policy_type} has no object_injection_mode")
            return 1
        config.object_injection_mode = args.injection_mode
        config.validate_object_tokens()
        print(f"           : injection mode {args.injection_mode}")
    if args.pretrained:
        config.pretrained_path = args.pretrained

    policy = make_policy(cfg=config, ds_meta=metadata, rename_map=rename_map)
    policy.eval()
    params = sum(p.numel() for p in policy.parameters())
    ok &= check("policy constructed", True, f"{params / 1e9:.2f}B params")

    # The features and statistics must come from *our* dataset, not the
    # checkpoint's. A checkpoint trained on another embodiment carries its
    # action width in both: VLA-JEPA's are 7-d against our 16-d, and the
    # normalizer then fails inside `2 * (tensor - min) / denom - 1` rather than
    # anywhere informative. Training applies the same overrides for the same
    # reason (lerobot_train.py:505-520).
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.pretrained or None,
        dataset_stats=metadata.stats,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": rename_map},
            "normalizer_processor": {
                "features": {**config.input_features, **config.output_features},
                "norm_map": config.normalization_mapping,
                "stats": metadata.stats,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
                "stats": metadata.stats,
            },
        },
    )
    ok &= check("processors built", True)

    # LeRobot's own resolver, not a hand-built action-only dict. Policies differ
    # in what history they ask for, and the difference is not cosmetic:
    # VLA-JEPA's world-model loss requests `num_video_frames` observation
    # frames, so an action-only delta gives it one frame where it expects eight
    # and it fails deep inside a reshape. Reading the config's own
    # `*_delta_indices` is exactly what training does.
    delta = resolve_delta_timestamps(config, metadata, rename_map)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, delta_timestamps=delta)
    ok &= check("delta timestamps resolved", True,
                f"{ {k: len(v) for k, v in (delta or {}).items()} }")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    batch = next(iter(loader))

    # Required, not probed. `getattr(..., None)` would let a mis-wired plugin
    # skip both checks silently, which is the failure they exist to catch.
    from oct_vla.policies.object_conditioning import ObjectConditionedPolicyMixin

    if isinstance(policy, ObjectConditionedPolicyMixin):
        try:
            conditioning = policy.object_conditioning
            ok &= check("object module reachable", True,
                        f"at {policy.object_module_path}, width {conditioning.width}, "
                        f"mode {conditioning.mode}")
            ok &= check("object residual starts at zero", not conditioning.is_live)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            ok &= check("object module reachable", False, str(error))

    processed = preprocessor(batch)
    loss, _ = policy.forward(processed)
    ok &= check("forward pass", torch.isfinite(torch.as_tensor(loss)).all().item(),
                f"loss {float(torch.as_tensor(loss).mean()):.4f}")

    # requires_grad is not enough, and neither is a falling loss: a backbone
    # that freezes by *name* can leave the object path out of the optimiser
    # while everything else trains normally. GR00T's action head calls
    # set_trainable_parameters in its own __init__, before the conditioning
    # module is attached, so the module is never named there -- true today, and
    # exactly the kind of thing a LeRobot upgrade changes silently. So take a
    # real gradient and require it to be non-zero.
    if isinstance(policy, ObjectConditionedPolicyMixin):
        prefix = policy._object_state_prefix
        object_params = {
            name: parameter
            for name, parameter in policy.named_parameters()
            if name.startswith(prefix)
        }
        ok &= check("object parameters exist", bool(object_params),
                    f"{len(object_params)} tensors under {prefix}")
        frozen = [name for name, parameter in object_params.items() if not parameter.requires_grad]
        ok &= check("object parameters are trainable", not frozen,
                    f"{len(frozen)} frozen" if frozen else "all require grad")
        policy.zero_grad(set_to_none=True)
        torch.as_tensor(policy.forward(preprocessor(batch))[0]).mean().backward()
        moved = [
            name for name, parameter in object_params.items()
            if parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        ]
        # Not every tensor: the zero-init projections receive no gradient on the
        # very first step by construction, which is the mechanism working, not a
        # fault. What must not happen is *nothing* in the object path moving.
        ok &= check("object path receives gradient", bool(moved),
                    f"{len(moved)}/{len(object_params)} tensors with non-zero grad")
        policy.zero_grad(set_to_none=True)

    with torch.no_grad():
        actions = policy.predict_action_chunk(preprocessor(batch))
    ok &= check("action chunk predicted", actions.shape[-1] == action_dim,
                f"shape {tuple(actions.shape)}, dataset action {action_dim}d")

    if args.peft:
        # Overrides, not a prebuilt config: `wrap_with_peft(peft_config=...)`
        # hands the object straight to peft's `get_peft_model`, which wants one
        # of *its* configs. Training passes overrides and lets the policy build
        # the rest from `_get_default_peft_targets`, so that is what is tested.
        targets = policy._get_default_peft_targets()
        wrapped = policy.wrap_with_peft(
            peft_cli_overrides={"method_type": "LORA", "r": 16, "lora_alpha": 32}
        )
        trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
        ok &= check("LoRA attaches to something", bool(trainable), f"{len(trainable)} tensors")
        saved = targets.get("modules_to_save") or []
        if saved:
            hit = [n for n in trainable if any(m in n for m in saved)]
            ok &= check("object path is trainable", bool(hit),
                        f"{len(hit)} tensors under {saved}")

    print(f"\n{'PASS' if ok else 'FAIL'}: {args.policy_type}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("\nFAIL: raised")
        sys.exit(1)
