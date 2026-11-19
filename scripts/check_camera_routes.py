#!/usr/bin/env python
"""Prove each camera reaches the model slot it is named for (stage 3, before training).

Paints the three cameras of one dataset frame pure red (head), green (left wrist) and blue
(right wrist), runs the checkpoint's own preprocessor and a forward pass, and captures the
tensor entering every 3-channel input convolution (ACT's ResNet stem; GR00T's vision patch
embedding). Each captured image is identified by its dominant channel, which survives
per-channel normalisation. The script fails on a missing, duplicated or permuted stream
-- GR00T matches cameras to its video keys and, on no match, falls back to alphabetical
order with only a warning -- and writes a labelled contact sheet of what arrived.

    check_camera_routes.py --checkpoint RUN/checkpoints/last/pretrained_model \\
        --dataset-root VIEW --out OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PATTERNS = {"observation.images.head": 0, "observation.images.left_wrist": 1,
            "observation.images.right_wrist": 2}
NAMES = ("red", "green", "blue")


def main() -> int:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    import oct_vla.policies  # noqa: F401
    from oct_vla.data.policy_inputs import rename_map_for

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = args.checkpoint
    rename_map = rename_map_for(config.type)
    repo = f"local/{args.dataset_root.name}"
    meta = LeRobotDatasetMetadata(repo, root=args.dataset_root)
    dataset = LeRobotDataset(repo, root=args.dataset_root)
    policy = make_policy(cfg=config, ds_meta=meta, rename_map=rename_map).eval()
    pre, _ = make_pre_post_processors(
        policy_cfg=config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                "rename_observations_processor": {"rename_map": rename_map}})
    raw = torch.utils.data.default_collate([dataset[0]])
    cameras = [k for k in raw if k.startswith("observation.images.")]
    if sorted(cameras) != sorted(PATTERNS):
        raise SystemExit(f"the view's cameras are {cameras}, expected {sorted(PATTERNS)}")
    for key, channel in PATTERNS.items():
        img = torch.zeros_like(raw[key])
        img[:, channel] = 1.0 if img.is_floating_point() else 255
        raw[key] = img

    captured, warnings = [], []

    class Grab(logging.Handler):
        def emit(self, record):
            warnings.append(record.getMessage())

    logging.getLogger().addHandler(Grab(level=logging.WARNING))
    hooks = [m.register_forward_pre_hook(lambda mod, inp: captured.append(inp[0].detach().float().cpu()))
             for m in policy.modules()
             if isinstance(m, torch.nn.Conv2d) and m.in_channels == 3]
    if not hooks:
        raise SystemExit("no 3-channel input convolution found to observe")
    batch = _preprocess_dataset_batch(raw, meta.camera_keys, rename_map, pre)
    with torch.no_grad():
        policy.predict_action_chunk(batch)
    for h in hooks:
        h.remove()

    seen = []
    for t in captured:
        for img in t.reshape(-1, *t.shape[-3:]):
            seen.append(NAMES[int(img.mean(dim=(1, 2)).argmax())])
    expected = list(getattr(policy.config, "image_features", {}) or PATTERNS)
    report = {"checkpoint": str(args.checkpoint), "policy": config.type,
              "declared_camera_order": expected,
              "arrived_in_order": seen,
              "warnings": [w for w in warnings if "camera" in w.lower() or "alphabet" in w.lower()]}
    want = [NAMES[PATTERNS[k]] for k in expected if k in PATTERNS]
    counts = {n: seen.count(n) for n in NAMES}
    problems = []
    if any(v == 0 for v in counts.values()):
        problems.append(f"a camera never reached the model: {counts}")
    if len(set(counts.values())) > 1:
        problems.append(f"cameras arrive unevenly (duplicated or dropped): {counts}")
    if seen[:len(want)] != want and not report["warnings"]:
        problems.append(f"order {seen[:len(want)]} differs from the declared {want}")
    if report["warnings"]:
        problems.append("the processor warned about camera matching: " + "; ".join(report["warnings"]))
    report["problems"] = problems
    report["passed"] = not problems
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "camera_routes.json").write_text(json.dumps(report, indent=1) + "\n")
    _sheet(captured, seen, args.out / "camera_routes.png")
    print(json.dumps(report, indent=1))
    return 0 if not problems else 1


def _sheet(captured, seen, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs = [img for t in captured for img in t.reshape(-1, *t.shape[-3:])][:12]
    fig, axes = plt.subplots(1, len(imgs), figsize=(2 * len(imgs), 2.2), squeeze=False)
    for ax, img, name in zip(axes[0], imgs, seen):
        x = img.permute(1, 2, 0).numpy()
        x = (x - x.min()) / max(1e-6, float(x.max() - x.min()))
        ax.imshow(x)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.suptitle("what each model input convolution received, in call order", fontsize=8)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
