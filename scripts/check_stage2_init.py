#!/usr/bin/env python
"""Before a stage-2 VLA run: prove it starts as its stage-1 policy, and can learn.

Takes exactly the arguments `lerobot_train` takes (the training script runs
this instead of training when INIT_CHECK=1), and builds the dataset, policy,
PEFT wrap and processors the way `lerobot_train.train` does. Then, on one real
training batch:

1. **Starts at stage 1.** The stage-2 policy's predicted action chunk and loss
   match the stage-1 policy's, loaded independently from the same checkpoint,
   under the same noise. Layerwise and AdaLN arms must match to within bf16
   noise; the in-context arm is not identity at init, so its deviation is
   measured and reported, and fails only if it is gross.
2. **Can learn.** After one backward pass, the object branch -- and for the
   encoder-side arms the AdaLN projection or in-context tokens specifically --
   receives non-zero gradient. A branch the backbone never calls would train
   with a falling loss and stay at zero; this is the check that catches it.

Writes a JSON report to <output root>/run_metadata/<job>_init_check.json and exits
non-zero on failure.
"""

# No `from __future__ import annotations`: LeRobot's `parser.wrap` reads the
# wrapped function's annotation to find its config class, and a stringified
# annotation fails there -- which is how every INIT_CHECK in the first smoke
# run failed before it ran a single check.
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Relative change in the predicted chunk that still counts as "unchanged".
EXACT = 1e-2
#: The in-context arm's step-0 deviation above which something is broken, not
#: merely non-identity (the entity keys take some attention mass by design).
#: Behind the gate (init -4) it measures 0.36 % on pi0.5, 3.1 % on SmolVLA and
#: 0.16 % on GR00T; before the gate and position fix, pi0.5 moved 137 %.
GROSS = 0.1


def relative(a, b) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6))


def main() -> int:
    import torch
    from lerobot.configs import parser
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.factory import get_policy_class
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    import oct_vla.policies  # noqa: F401
    from oct_vla.policies.object_conditioning import unwrap_object_conditioning

    @parser.wrap()
    def check(cfg: TrainPipelineConfig) -> int:
        cfg.validate()
        stage1 = Path(str(cfg.policy.pretrained_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed or 0)
        dataset, _ = make_train_eval_datasets(cfg)
        policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
        if cfg.peft is not None:
            policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))
        stats = rename_stats(dataset.meta.stats, cfg.rename_map)
        inner = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=cfg.policy, pretrained_path=str(stage1), dataset_stats=stats,
            preprocessor_overrides={
                "device_processor": {"device": device.type},
                "normalizer_processor": {
                    "features": {**inner.config.input_features, **inner.config.output_features},
                    "norm_map": inner.config.normalization_mapping,
                    "stats": stats,
                },
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(cfg.batch_size, 4),
                                             shuffle=False, num_workers=0)
        raw = next(iter(loader))
        # Exactly as training prepares a batch: cameras to float32/255 *before*
        # the processor. Calling the processor on the raw batch fed uint8
        # frames -- SmolVLA crashed on them, pi0.5 and GR00T silently saw
        # inputs 255x out of range.
        batch = _preprocess_dataset_batch(raw, dataset.meta.camera_keys, cfg.rename_map,
                                          preprocessor)
        for key in dataset.meta.camera_keys:
            key = cfg.rename_map.get(key, key)
            if key in batch:
                assert batch[key].is_floating_point() and batch[key].max() <= 1.0 + 1e-6, key

        stage1_config = PreTrainedConfig.from_pretrained(stage1)
        reference = get_policy_class(stage1_config.type).from_pretrained(
            str(stage1), config=stage1_config
        )
        reference.to(device)

        def outputs(model):
            model.eval()
            torch.manual_seed(0)
            with torch.no_grad():
                chunk = model.predict_action_chunk(batch)
            torch.manual_seed(1)
            with torch.no_grad():
                loss = model.forward(batch)[0]
            return chunk, loss

        chunk_2, loss_2 = outputs(policy)
        chunk_1, loss_1 = outputs(reference)
        arm = getattr(inner.config, "object_conditioning", "rgb")
        report = {
            "stage1": str(stage1), "type": inner.config.type, "arm": arm,
            "chunk_relative_change": relative(chunk_2, chunk_1),
            "loss_stage1": float(loss_1), "loss_stage2": float(loss_2),
        }
        del reference
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # One backward: which parameters of the object branch receive gradient?
        policy.train()
        torch.manual_seed(1)
        policy.zero_grad(set_to_none=True)
        policy.forward(batch)[0].backward()
        grads = {}
        if arm != "rgb":
            control = unwrap_object_conditioning(inner.object_conditioning)
            for name in ("layers", "adaln", "incontext"):
                module = getattr(control, name, None)
                if module is None:
                    continue
                norms = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
                trainable = sum(p.requires_grad for p in module.parameters())
                grads[name] = {"trainable": trainable, "with_grad": len(norms),
                               "nonzero": sum(n > 0 for n in norms)}
        report["object_gradients"] = grads

        failures = []
        tolerance = GROSS if arm == "kv_tokens" else EXACT
        if report["chunk_relative_change"] > tolerance:
            failures.append(f"stage 2 does not start at stage 1: chunk change "
                            f"{report['chunk_relative_change']:.3e} > {tolerance}")
        # The branch each arm adds, by its state-dict name (object_conditioning.py).
        added = {"kv": "layers", "kv_adaln": "adaln", "kv_tokens": "incontext"}.get(arm)
        for name, g in grads.items():
            required = name in ("layers", added)
            if g["trainable"] == 0:
                failures.append(f"{name}: no trainable parameters (PEFT did not keep it)")
            elif required and g["nonzero"] == 0:
                failures.append(f"{name}: no gradient reached the {arm} branch")
        report["failures"] = failures
        target = Path(cfg.output_dir).parent / "run_metadata" / f"{cfg.job_name}_init_check.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 1 if failures else 0

    return check()


if __name__ == "__main__":
    raise SystemExit(main())
