from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from oct_vla.experiments import (
    STAGE_GPU_HOUR_LIMITS,
    ExperimentCell,
    ThroughputMeasurement,
    build_manifest,
    select_rollout_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def cell(
    *,
    stage: str = "stage_1",
    backbone: str = "act",
    representation: str = "rgb",
    seed: int = 1000,
    action_schema: dict | None = None,
    dataset_revision: str = "dataset-v1",
    base_revision: str = "base-v1",
    adaptation: str = "lora-r16",
    train_steps: int = 20_000,
    profile: str = "three_object",
    evaluation_split: str = "development",
) -> ExperimentCell:
    return ExperimentCell(
        stage=stage,
        backbone=backbone,
        representation=representation,
        seed=seed,
        action_schema=action_schema
        or {
            "space": "absolute_joint",
            "gripper": "binary_command",
            "n_action_steps": 25,
        },
        dataset_revision=dataset_revision,
        base_revision=base_revision,
        adaptation=adaptation,
        train_steps=train_steps,
        profile=profile,
        evaluation_split=evaluation_split,
    )


def measured(
    stage: str = "stage_1",
    hours: float = 4.5,
    *,
    steps: int = 20_000,
    backbone: str = "act",
    representation: str = "rgb",
    action_schema: dict | None = None,
    dataset_revision: str = "dataset-v1",
    base_revision: str = "base-v1",
    adaptation: str = "lora-r16",
    source: str = "sacct job 123",
    eval_overhead_gpu_hours: float = 0.0,
    **overrides,
) -> ThroughputMeasurement:
    """A measurement of `hours` GPU-hours spent over `steps` steps.

    A rate, not a total: the manifest extrapolates each cell's cost from it.
    The identity fields default to the same values `cell()` uses, because a
    measurement only funds cells it matches.
    """
    return ThroughputMeasurement(
        stage=stage,
        backbone=backbone,
        representation=representation,
        action_schema=action_schema
        or {
            "space": "absolute_joint",
            "gripper": "binary_command",
            "n_action_steps": 25,
        },
        dataset_revision=dataset_revision,
        base_revision=base_revision,
        adaptation=adaptation,
        measured_steps=steps,
        gpu_count=1,
        elapsed_seconds=hours * 3600.0,
        eval_overhead_gpu_hours=eval_overhead_gpu_hours,
        source=source,
        **overrides,
    )


def episode(
    seed: int,
    success: bool,
    transfers: int,
    *,
    profile: str = "three_object",
    split: str = "development",
    infeasible_steps: int = 0,
) -> dict:
    return {
        "seed": seed,
        "profile": profile,
        "evaluation_split": split,
        "success": success,
        "steps": 600,
        "transfers_completed": transfers,
        "infeasible_steps": infeasible_steps,
        "reason": "success" if success else "step_limit",
    }


def report(
    checkpoint: str,
    rows: list[dict],
    *,
    split: str | None = "development",
    summary: dict | None = None,
    max_steps: int = 600,
    steps_per_object: int = 200,
    n_action_steps: int = 25,
    eval_tag: str | None = None,
) -> dict:
    value = {
        "checkpoint": checkpoint,
        "episodes": rows,
        "summary": summary
        if summary is not None
        else {
            "three_object": {
                "episodes": len(rows),
                "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
                "mean_transfers": sum(row["transfers_completed"] for row in rows) / len(rows),
            }
        },
        "max_steps": max_steps,
        "steps_per_object": steps_per_object,
        "n_action_steps": n_action_steps,
    }
    if split is not None:
        value["evaluation_split"] = split
    if eval_tag is not None:
        value["eval_tag"] = eval_tag
    return value


def test_stage_budget_is_exactly_one_hundred_gpu_hours():
    assert list(STAGE_GPU_HOUR_LIMITS.values()) == [10, 15, 20, 25, 30]
    assert sum(STAGE_GPU_HOUR_LIMITS.values()) == 100


def test_manifest_is_deterministic_and_hashes_scientific_identity():
    first = build_manifest([cell()], [measured()])
    second = build_manifest([cell()], [measured()])
    # The measurement moves with the cell: a rate taken on a joint_delta run is
    # what funds a joint_delta cell, and `matches` now enforces that. Only
    # train_steps may differ, because extrapolating step count from a measured
    # per-step rate is the whole point.
    other_action = {"space": "joint_delta", "gripper": "binary_command"}
    changed_action = build_manifest(
        [cell(action_schema=other_action)], [measured(action_schema=other_action)]
    )
    changed_revision = build_manifest(
        [cell(dataset_revision="dataset-v2")], [measured(dataset_revision="dataset-v2")]
    )
    changed_steps = build_manifest([cell(train_steps=30_000)], [measured()])

    assert first == second
    assert first["manifest_hash"] != changed_action["manifest_hash"]
    assert first["manifest_hash"] != changed_revision["manifest_hash"]
    assert first["manifest_hash"] != changed_steps["manifest_hash"]
    assert first["cells"][0]["backbone"] == "act"
    assert first["cells"][0]["representation"] == "rgb"
    assert first["cells"][0]["seed"] == 1000
    assert first["cells"][0]["dataset_revision"] == "dataset-v1"
    assert first["cells"][0]["base_revision"] == "base-v1"
    assert first["cells"][0]["adaptation"] == "lora-r16"
    assert first["cells"][0]["train_steps"] == 20_000
    assert first["cells"][0]["cell_hash"] != changed_action["cells"][0]["cell_hash"]


def test_stage_budget_requires_measured_throughput_for_used_stages():
    with pytest.raises(ValueError, match="Measured throughput is required"):
        build_manifest([cell(stage="stage_2")], [measured(stage="stage_1")])


def test_estimated_or_nonfinite_throughput_records_are_rejected():
    with pytest.raises(ValueError, match="not measured"):
        ThroughputMeasurement.from_dict(
            {
                "stage": "stage_1",
                "measured_gpu_hours": 4.5,
                "source": "spreadsheet guess",
                "estimated": True,
            }
        )
    for value in [math.nan, math.inf, -1.0, 0.0]:
        with pytest.raises(ValueError, match="finite and positive"):
            build_manifest([cell()], [measured(hours=value)])


def test_duplicate_stage_measurements_are_rejected():
    with pytest.raises(ValueError, match="Duplicate throughput"):
        build_manifest([cell()], [measured(), measured(hours=4.6)])


def test_planning_is_bounded_to_three_object_development_cells():
    with pytest.raises(ValueError, match="three_object"):
        build_manifest([cell(profile="four_object")], [measured()])
    with pytest.raises(ValueError, match="development"):
        build_manifest([cell(evaluation_split="test")], [measured()])


def test_rollout_checkpoint_ranking_uses_success_transfers_then_infeasible_steps(tmp_path):
    reports = [
        (
            tmp_path / "a.json",
            report(
                "ckpt-a",
                [
                    episode(800, True, 3),
                    episode(801, False, 2),
                    episode(802, False, 1, infeasible_steps=4),
                ],
            ),
        ),
        (
            tmp_path / "b.json",
            report(
                "ckpt-b",
                [
                    episode(800, True, 3),
                    episode(801, True, 1),
                    episode(802, False, 0, infeasible_steps=2),
                ],
            ),
        ),
        (
            tmp_path / "c.json",
            report(
                "ckpt-c",
                [
                    episode(800, True, 3),
                    episode(801, True, 1),
                    episode(802, False, 0, infeasible_steps=0),
                ],
            ),
        ),
    ]

    selection = select_rollout_checkpoint(reports)

    assert selection["selected"]["checkpoint"] == "ckpt-c"
    assert [row["checkpoint"] for row in selection["ranked"]] == ["ckpt-c", "ckpt-b", "ckpt-a"]
    assert selection["selection_rule"] == [
        "successes desc",
        "transfers_completed desc",
        "infeasible_steps asc",
    ]
    assert selection["evaluation_split"] == "development"
    assert selection["execution_budget"] == {
        "max_steps": 600,
        "n_action_steps": 25,
        "steps_per_object": 200,
    }


def test_rollout_checkpoint_ranking_requires_explicit_development_split(tmp_path):
    with pytest.raises(ValueError, match="evaluation_split='development'"):
        select_rollout_checkpoint(
            [(tmp_path / "unlabeled.json", report("ckpt", [episode(800, True, 3)], split=None))]
        )
    legacy = report("ckpt", [episode(800, True, 3)], split=None)
    legacy["split"] = "development"
    assert select_rollout_checkpoint([(tmp_path / "legacy.json", legacy)])["selected"][
        "checkpoint"
    ] == "ckpt"


def test_rollout_checkpoint_ranking_rejects_count_shift_reports(tmp_path):
    rows = [episode(800, True, 3), episode(801, False, 1, profile="four_object")]
    with pytest.raises(ValueError, match="three_object"):
        select_rollout_checkpoint([(tmp_path / "countshift.json", report("ckpt", rows))])


def test_rollout_checkpoint_ranking_rejects_test_reports(tmp_path):
    with pytest.raises(ValueError, match="development"):
        select_rollout_checkpoint(
            [
                (
                    tmp_path / "test.json",
                    report("ckpt", [episode(800, True, 3, split="test")], split="test"),
                )
            ]
        )


def test_rollout_checkpoint_ranking_rejects_per_episode_split_contradictions(tmp_path):
    with pytest.raises(ValueError, match="contradicts report split"):
        select_rollout_checkpoint(
            [
                (
                    tmp_path / "bad.json",
                    report("ckpt", [episode(800, True, 3, split="test")]),
                )
            ]
        )


def test_rollout_checkpoint_ranking_rejects_duplicate_or_mismatched_seed_sets(tmp_path):
    with pytest.raises(ValueError, match="duplicate rollout seed"):
        select_rollout_checkpoint(
            [
                (
                    tmp_path / "dup.json",
                    report("dup", [episode(800, True, 3), episode(800, False, 1)]),
                )
            ]
        )
    with pytest.raises(ValueError, match="identical seed sets"):
        select_rollout_checkpoint(
            [
                (tmp_path / "a.json", report("a", [episode(800, True, 3)])),
                (tmp_path / "b.json", report("b", [episode(801, True, 3)])),
            ]
        )


def test_rollout_checkpoint_ranking_rejects_mismatched_execution_budgets(tmp_path):
    with pytest.raises(ValueError, match="identical execution budgets"):
        select_rollout_checkpoint(
            [
                (tmp_path / "a.json", report("a", [episode(800, True, 3)], max_steps=600)),
                (tmp_path / "b.json", report("b", [episode(800, True, 3)], max_steps=900)),
            ]
        )


def test_plan_experiments_cli_writes_the_same_manifest(tmp_path, monkeypatch):
    cells = tmp_path / "cells.json"
    throughput = tmp_path / "throughput.json"
    output = tmp_path / "manifest.json"
    cells.write_text(json.dumps([cell().to_manifest_record()]))
    throughput.write_text(json.dumps([measured().to_manifest_record()]))

    spec = importlib.util.spec_from_file_location(
        "plan_experiments", ROOT / "scripts/plan_experiments.py"
    )
    plan_experiments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan_experiments)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan",
            "--cells",
            str(cells),
            "--throughput",
            str(throughput),
            "--output",
            str(output),
        ],
    )

    assert plan_experiments.main() == 0
    assert json.loads(output.read_text()) == build_manifest([cell()], [measured()])


def test_a_measurement_only_funds_cells_it_was_taken_on():
    """The stage guard alone is decorative.

    It requires *a* measurement per stage, so a rate measured on a small ACT run
    would fund a pi0.5 cell sitting in the same stage and produce a budget
    number that looks authoritative and is not. `matches` is what binds the two,
    and it went unnoticed that it raised NameError because nothing called it.
    """
    with pytest.raises(ValueError, match="does not match the cell"):
        build_manifest([cell(backbone="pi05")], [measured(backbone="act")])
    with pytest.raises(ValueError, match="does not match the cell"):
        build_manifest([cell(representation="object")], [measured(representation="rgb")])
    with pytest.raises(ValueError, match="does not match the cell"):
        build_manifest([cell(adaptation="full-finetune")], [measured(adaptation="lora-r16")])
    # Seed and train_steps are not identity for this purpose: one measured rate
    # legitimately funds another seed, and extrapolating the step count is what
    # the per-step rate exists for.
    build_manifest([cell(seed=1001, train_steps=40_000)], [measured()])


def test_the_budget_extrapolates_each_cell_from_the_measured_rate():
    """Four hours measured over 20k steps is 8 hours for two 20k-step cells."""
    manifest = build_manifest(
        [cell(seed=1000), cell(seed=1001)], [measured(hours=4.0, steps=20_000)]
    )
    stage_1 = next(row for row in manifest["budget"] if row["stage"] == "stage_1")
    assert stage_1["cells"] == 2
    assert stage_1["planned_stage_gpu_hours"] == pytest.approx(8.0)
    assert stage_1["remaining_gpu_hours"] == pytest.approx(STAGE_GPU_HOUR_LIMITS["stage_1"] - 8.0)
    assert stage_1["fits"] is True

    # Over the ceiling must be visible, not silently rounded away.
    over = build_manifest([cell(train_steps=100_000)], [measured(hours=4.0, steps=20_000)])
    assert next(row for row in over["budget"] if row["stage"] == "stage_1")["fits"] is False


def test_eval_overhead_is_charged_once_per_cell():
    manifest = build_manifest([cell()], [measured(hours=1.0, eval_overhead_gpu_hours=0.5)])
    stage_1 = next(row for row in manifest["budget"] if row["stage"] == "stage_1")
    assert stage_1["planned_stage_gpu_hours"] == pytest.approx(1.5)
