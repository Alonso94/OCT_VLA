#!/usr/bin/env python
"""Fold a stage-1 LoRA adapter into its base weights, for a stage-2 run to start from.

pi0.5 and SmolVLA train stage 1 as a LoRA adapter over the public base model, so
their checkpoint holds only the adapter. Nothing in LeRobot loads an adapter
into a *different* policy class (the object-conditioned `control_*`) and then
adds new modules. So stage 2 starts instead from a full-weights checkpoint in
which the adapter is merged, and trains a fresh adapter over it -- the same
route the base model already takes (`--policy.pretrained_path`). The budget
control (`rgb_cont`) starts from the same merged checkpoint, so the two differ
only in the object path.

Three checks, each refusing to write a checkpoint that would silently be wrong:

1. the stage-1 checkpoint is an adapter, and its base is what its config says;
2. **functional**: the merged model predicts the same action chunk as the
   adapter model on a real observation from the training set, from the same
   noise (bf16 rounding in the merge is the only permitted difference);
3. **round trip**: the saved directory, reloaded through the class training
   will use, holds exactly the saved tensors and predicts the same chunk.

GPU node, policy environment:

    scripts/merge_stage1_adapter.py --stage1 RUN/checkpoints/last/pretrained_model \\
        --dataset-root DATASET --repo-id REPO --output RUN/merged
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Largest relative change in the predicted chunk the merge may cause. bf16
#: rounding of W + BA across a 10-step flow is ~1e-3 in practice; a wrong
#: adapter or a skipped merge moves it by O(1).
TOLERANCE = 5e-2

#: Stage-1 files a merged checkpoint needs beside its weights: the processor
#: pipelines (normalisation statistics fitted on stage 1's training split, which
#: stage 2 shares) and the tokenizer.
SKIP = {"adapter_config.json", "adapter_model.safetensors", "config.json",
        "model.safetensors", "README.md"}


def observation_batch(repo_id: str, root: Path, preprocessor, torch):
    """One training-split frame, shaped and processed exactly as rollouts are."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root, episodes=[0])
    item = dataset[len(dataset) // 2]
    batch = {}
    for key, value in item.items():
        if key.startswith("observation.") and isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
    batch["task"] = [item["task"]]
    return preprocessor(batch)


def chunk(policy, batch, torch, seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        return policy.predict_action_chunk(batch).float()


def relative_change(a, b) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", required=True, type=Path,
                        help="The stage-1 pretrained_model directory (an adapter)")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.factory import get_policy_class

    import oct_vla.policies  # noqa: F401 -- registers masked_pi05 and the plugins
    from oct_vla.data.policy_inputs import rename_map_for
    from oct_vla.policies.stage_loading import MODEL_FILE, verify_loaded_weights

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    stage1 = args.stage1.resolve()
    if not (stage1 / "adapter_config.json").is_file():
        raise SystemExit(
            f"{stage1} holds no adapter. A full-weights stage 1 (GR00T, ACT) needs no "
            "merge: point stage 2 at it directly."
        )
    config = PreTrainedConfig.from_pretrained(stage1)
    if not config.use_peft:
        raise SystemExit(f"{stage1}/config.json says use_peft=false beside an adapter file")
    config.pretrained_path = stage1
    adapter = json.loads((stage1 / "adapter_config.json").read_text())
    print(f"stage 1  : {stage1} ({config.type}, adapter over {adapter['base_model_name_or_path']})")

    rename_map = rename_map_for(config.type)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    policy = make_policy(cfg=config, ds_meta=metadata, rename_map=rename_map)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config, pretrained_path=stage1,
        preprocessor_overrides={
            "device_processor": {"device": str(config.device)},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    batch = observation_batch(args.repo_id, args.dataset_root, preprocessor, torch)

    # The adapter must have trained: a zero adapter passes every check below
    # while the "stage 1" being continued is just the public base model.
    lora_b = [p for n, p in policy.named_parameters() if "lora_B" in n]
    if not lora_b:
        raise SystemExit(f"{stage1} loaded with no LoRA parameters; the adapter did not attach")
    lora_norm = float(sum(p.detach().float().norm() ** 2 for p in lora_b) ** 0.5)
    print(f"adapter  : {len(lora_b)} lora_B tensors, norm {lora_norm:.3e}")
    if lora_norm == 0.0:
        raise SystemExit("every lora_B is zero: this adapter never trained")

    with_adapter = chunk(policy, batch, torch)
    merged = policy.merge_and_unload()
    merged.eval()
    after_merge = chunk(merged, batch, torch)
    merge_change = relative_change(after_merge, with_adapter)
    print(f"merge    : relative change in predicted chunk {merge_change:.2e}")
    if merge_change > TOLERANCE:
        raise SystemExit(f"merge changed the policy by {merge_change:.2e} > {TOLERANCE}")
    merged.config.use_peft = False
    merged.config.pretrained_path = None
    args.output.mkdir(parents=True)
    merged.save_pretrained(args.output)
    for item in stage1.iterdir():
        if item.name in SKIP:
            continue
        target = args.output / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    del policy, merged
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Round trip through the class training loads, not the object we saved.
    reloaded_config = PreTrainedConfig.from_pretrained(args.output)
    cls = get_policy_class(reloaded_config.type)
    reloaded = cls.from_pretrained(str(args.output), config=reloaded_config)
    reloaded.eval()
    report = verify_loaded_weights(reloaded, args.output / MODEL_FILE)
    round_trip = relative_change(chunk(reloaded, batch, torch), after_merge)
    print(f"reload   : {report['tensors']} tensors verified; "
          f"relative change in predicted chunk {round_trip:.2e}")
    if round_trip > 1e-3:
        raise SystemExit(f"reloaded checkpoint predicts differently ({round_trip:.2e})")

    provenance = {
        "stage1": str(stage1),
        "stage1_type": config.type,
        "base_model": adapter["base_model_name_or_path"],
        "lora_r": adapter.get("r"),
        "lora_alpha": adapter.get("lora_alpha"),
        "lora_B_norm": lora_norm,
        "merge_relative_change": merge_change,
        "reload_relative_change": round_trip,
        "tensors": report["tensors"],
        "dataset": str(args.dataset_root),
    }
    (args.output / "merge_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote    : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
