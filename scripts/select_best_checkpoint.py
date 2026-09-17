#!/usr/bin/env python
"""Point `checkpoints/best` at the checkpoint before the run overfits.

LeRobot always leaves `checkpoints/last`, and evaluation defaults to it. On this
dataset that is the wrong checkpoint to score: the ACT runs reach their minimum
validation loss well before the step budget ends and then climb, so `last` is a
model that has been memorising the training scenes for tens of thousands of
steps. Evaluating it measures how badly a run overfit, not how well the
conditioning works -- and the sweep compares arms, so an arm that happens to
overfit faster looks worse for a reason that has nothing to do with its inputs.

The validation loss lives only in the training log (`wandb.enable=false`), as
lines that LeRobot writes every `--eval_steps`:

    INFO ... ot_train.py:803 step 12000: eval_loss=0.3911

Selection is *restricted to steps that were actually saved*. The log is finer
grained than `--save_freq`, so the global minimum usually names a step with no
checkpoint behind it; silently rounding to the nearest saved step would report a
loss the selected weights do not have. This picks the saved checkpoint with the
lowest logged loss instead, and prints the global minimum alongside it so the
cost of the save granularity is visible rather than hidden.

The result is a symlink, not a copy: 1.2 GB per checkpoint, and a symlink keeps
`best` and the step it names obviously the same weights.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: `step 12000: eval_loss=0.3911`, as emitted by lerobot.scripts.lerobot_train.
EVAL_LINE = re.compile(r"step (\d+): eval_loss=([0-9.eE+-]+)")


def eval_losses(log_text: str) -> dict[int, float]:
    """Validation loss by step, last value winning.

    A requeued job appends to the same log, so a step can appear twice; the
    later line came from the run that actually produced the checkpoints on disk.
    """
    return {int(step): float(loss) for step, loss in EVAL_LINE.findall(log_text)}


def saved_steps(checkpoints: Path) -> list[int]:
    """Steps with a checkpoint on disk, ascending.

    `last` is skipped: it is a symlink onto one of the numbered directories, so
    counting it would let the same weights be considered twice under two names.
    """
    steps = []
    for entry in checkpoints.iterdir():
        if entry.name.isdigit() and (entry / "pretrained_model").is_dir():
            steps.append(int(entry.name))
    return sorted(steps)


def select(losses: dict[int, float], steps: list[int]) -> tuple[int, float]:
    """Saved step with the lowest validation loss.

    Ties go to the earlier step: two checkpoints that fit the validation set
    equally well are not equally good, and the earlier one has had less
    opportunity to memorise.
    """
    scored = [(step, losses[step]) for step in steps if step in losses]
    if not scored:
        raise SystemExit(
            f"No saved step has a logged eval_loss. Saved: {steps}; "
            f"logged: {sorted(losses)[:10]}... "
            "Check that --eval_steps divides --save_freq."
        )
    return min(scored, key=lambda pair: (pair[1], pair[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, help="Training output directory")
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Training stdout, the only place eval_loss is recorded when wandb is off.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report the choice without writing the symlink."
    )
    args = parser.parse_args()

    checkpoints = args.run / "checkpoints"
    if not checkpoints.is_dir():
        raise SystemExit(f"No checkpoints directory under {args.run}")
    losses = eval_losses(args.log.read_text())
    if not losses:
        raise SystemExit(f"No 'step N: eval_loss=' lines in {args.log}")
    steps = saved_steps(checkpoints)
    if not steps:
        raise SystemExit(f"No numbered checkpoints under {checkpoints}")

    step, loss = select(losses, steps)
    floor_step = min(losses, key=lambda s: (losses[s], s))
    print(f"saved checkpoints : {steps}")
    print(f"best saved        : step {step}, eval_loss {loss:.4f}")
    print(f"global minimum    : step {floor_step}, eval_loss {losses[floor_step]:.4f}")
    if floor_step != step:
        gap = (loss - losses[floor_step]) / losses[floor_step]
        print(f"cost of save granularity: {gap:+.1%} above the logged floor")
    final = max(losses)
    print(f"final             : step {final}, eval_loss {losses[final]:.4f} "
          f"({(losses[final] - loss) / loss:+.1%} vs best)")

    if args.dry_run:
        return 0

    link = checkpoints / "best"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(f"{step:06d}", target_is_directory=True)
    (args.run / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "step": step,
                "eval_loss": loss,
                "logged_minimum": {"step": floor_step, "eval_loss": losses[floor_step]},
                "final": {"step": final, "eval_loss": losses[final]},
                "saved_steps": steps,
                "log": str(args.log),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n{link} -> {step:06d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
