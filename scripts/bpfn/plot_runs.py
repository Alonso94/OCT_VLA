#!/usr/bin/env python
"""Training curves of several PFN runs side by side, from their history.jsonl.

    plot_runs.py --out reports/sweep.png RUN [RUN ...]

Panels: training loss and gradient norm (window means), dev / held-out / count
both-correct, dev error at h1 / h8 / h16, moving median and 90th percentile, and the
held-out right-colour rate. Validation scores are for judging runs, never for picking a
checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run: Path) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    for line in (run / "history.jsonl").read_text().splitlines():
        record = json.loads(line)
        (val if "dev" in record else train).append(record)
    return train, val


def series(records: list[dict], *keys) -> tuple[list, list]:
    steps, values = [], []
    for record in records:
        value = record
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            steps.append(record["step"])
            values.append(value)
    return steps, values


PANELS = [
    ("training loss (window mean)", "train", ("loss",)),
    ("gradient norm before clipping", "train", ("grad_norm",)),
    ("dev both-correct", "val", ("dev", "both_correct")),
    ("held-out both-correct", "val", ("heldout_handover_colour", "both_correct")),
    ("count both-correct", "val", ("count_eval", "both_correct")),
    ("dev error at h8 (mm)", "val", ("dev", "h8_mm")),
    ("dev error at h16 (mm)", "val", ("dev", "h16_mm")),
    ("dev moving p90 (mm)", "val", ("dev", "moving_p90_mm")),
    ("held-out right colour", "val", ("heldout_handover_colour", "right_colour")),
]


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for run in args.runs:
        train, val = load(run)
        for ax, (title, source, keys) in zip(axes.flat, PANELS):
            steps, values = series(train if source == "train" else val, *keys)
            if steps:
                ax.plot(steps, values, label=run.name, marker="." if source == "val" else None)
            ax.set_title(title, fontsize=10)
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.set_xlabel("step", fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
