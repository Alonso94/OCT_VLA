"""Which checkpoint gets evaluated decides what the sweep reports, so the rule
is pinned here rather than left to whatever `last` happens to be."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "select_best_checkpoint", ROOT / "scripts/select_best_checkpoint.py"
)
select_best = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select_best)

LOG = """\
INFO 2026-09-17 15:14:29 /factory.py:220 Train/eval split: 225 train, 78 eval
INFO 2026-09-17 15:18:49 ot_train.py:803 step 2000: eval_loss=0.4796
INFO 2026-09-17 15:20:20 ot_train.py:803 step 4000: eval_loss=0.4012
INFO 2026-09-17 15:21:39 ot_train.py:803 step 5000: eval_loss=0.3911
INFO 2026-09-17 15:22:59 ot_train.py:803 step 6000: eval_loss=0.3950
INFO 2026-09-17 15:24:11 ot_train.py:803 step 8000: eval_loss=0.4400
"""


def make_run(tmp_path: Path, steps: list[int]) -> Path:
    for step in steps:
        (tmp_path / "checkpoints" / f"{step:06d}" / "pretrained_model").mkdir(parents=True)
    (tmp_path / "checkpoints" / "last").symlink_to(f"{max(steps):06d}")
    return tmp_path


# ---------------------------------------------------------------------- parsing


def test_eval_losses_are_read_from_the_training_log():
    """wandb is off in every job here, so stdout is the only record there is."""
    assert select_best.eval_losses(LOG) == {
        2000: 0.4796, 4000: 0.4012, 5000: 0.3911, 6000: 0.3950, 8000: 0.4400
    }


def test_a_requeued_job_appends_and_the_later_value_wins():
    text = LOG + "INFO ot_train.py:803 step 4000: eval_loss=0.3000\n"
    assert select_best.eval_losses(text)[4000] == 0.3000


# --------------------------------------------------------------------- selecting


def test_selection_is_restricted_to_steps_that_were_actually_saved():
    """The log is finer grained than --save_freq. The global minimum at 5000 has
    no checkpoint here, and reporting it would name a loss the selected weights
    do not have."""
    step, loss = select_best.select(select_best.eval_losses(LOG), [2000, 4000, 6000, 8000])
    assert (step, loss) == (6000, 0.3950)


def test_the_saved_minimum_is_taken_when_it_is_available():
    step, loss = select_best.select(select_best.eval_losses(LOG), [2000, 4000, 5000, 8000])
    assert (step, loss) == (5000, 0.3911)


def test_the_last_checkpoint_is_not_chosen_just_for_being_last():
    """The whole reason this script exists: on these runs `last` is tens of
    thousands of steps past the minimum."""
    step, _ = select_best.select(select_best.eval_losses(LOG), [2000, 4000, 5000, 6000, 8000])
    assert step != 8000


def test_a_tie_goes_to_the_earlier_checkpoint():
    losses = {2000: 0.40, 6000: 0.40}
    assert select_best.select(losses, [2000, 6000])[0] == 2000


def test_no_overlap_between_logged_and_saved_steps_fails_loudly():
    with pytest.raises(SystemExit, match="No saved step has a logged eval_loss"):
        select_best.select({1000: 0.5}, [2000, 4000])


# ------------------------------------------------------------------------ output


def test_saved_steps_ignores_the_last_symlink(tmp_path):
    run = make_run(tmp_path, [2000, 4000])
    assert select_best.saved_steps(run / "checkpoints") == [2000, 4000]


def test_a_directory_without_weights_is_not_a_checkpoint(tmp_path):
    run = make_run(tmp_path, [2000])
    (run / "checkpoints" / "004000").mkdir()  # interrupted mid-save
    assert select_best.saved_steps(run / "checkpoints") == [2000]


def test_best_is_a_symlink_and_records_what_it_gave_up(tmp_path, capsys, monkeypatch):
    run = make_run(tmp_path, [2000, 4000, 6000, 8000])
    log = tmp_path / "train.out"
    log.write_text(LOG)
    monkeypatch.setattr(
        "sys.argv", ["select", "--run", str(run), "--log", str(log)]
    )
    assert select_best.main() == 0
    link = run / "checkpoints" / "best"
    assert link.is_symlink() and link.resolve().name == "006000"
    record = json.loads((run / "best_checkpoint.json").read_text())
    assert record["step"] == 6000
    # The step the log actually minimised at, kept so the cost of saving every
    # 2000 steps is visible rather than quietly absorbed.
    assert record["logged_minimum"] == {"step": 5000, "eval_loss": 0.3911}
    assert record["final"] == {"step": 8000, "eval_loss": 0.44}


def test_rerunning_replaces_an_existing_best(tmp_path, monkeypatch):
    run = make_run(tmp_path, [2000, 4000, 6000, 8000])
    (run / "checkpoints" / "best").symlink_to("008000")
    log = tmp_path / "train.out"
    log.write_text(LOG)
    monkeypatch.setattr("sys.argv", ["select", "--run", str(run), "--log", str(log)])
    assert select_best.main() == 0
    assert (run / "checkpoints" / "best").resolve().name == "006000"


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    run = make_run(tmp_path, [2000, 4000])
    log = tmp_path / "train.out"
    log.write_text(LOG)
    monkeypatch.setattr(
        "sys.argv", ["select", "--run", str(run), "--log", str(log), "--dry-run"]
    )
    assert select_best.main() == 0
    assert not (run / "checkpoints" / "best").exists()
    assert not (run / "best_checkpoint.json").exists()
