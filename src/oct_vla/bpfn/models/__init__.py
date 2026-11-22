"""The PFN models: ``axial`` (PoseWindowPFN, the first model) and ``table`` (TablePFN)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def build(config: dict) -> nn.Module:
    """A model from its config; a config without ``arch`` is the axial model."""
    from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN
    from oct_vla.bpfn.models.table_pfn import ARCH, TablePFN

    config = dict(config)
    arch = config.pop("arch", "axial")
    if arch == ARCH:
        return TablePFN(**config)
    if arch == "axial":
        return PoseWindowPFN(**config)
    raise ValueError(f"unknown model arch {arch!r}")


def load(run_dir: Path, device: str = "cpu") -> nn.Module:
    """The checkpoint of a training run (``run_dir/model.pt``), in eval mode."""
    checkpoint = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=False)
    model = build(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model
