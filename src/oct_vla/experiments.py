"""Bounded experiment planning and rollout-checkpoint selection.

This module is intentionally dry-run only. It creates deterministic manifests
for work that may be launched elsewhere, and it ranks already-produced rollout
evaluation reports. Nothing here submits a job, creates a checkpoint symlink, or
edits a training run directory.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGE_GPU_HOUR_LIMITS: dict[str, int] = {
    "stage_1": 10,
    "stage_2": 15,
    "stage_3": 20,
    "stage_4": 25,
    "stage_5": 30,
}
TOTAL_GPU_HOUR_LIMIT = 100
DEVELOPMENT_PROFILE = "three_object"
DEVELOPMENT_SPLIT = "development"
FORBIDDEN_EVAL_SCOPES = {"test", "countshift", "count_shift", "count-shift"}


def canonical_json(value: Any) -> str:
    """Stable JSON used for manifest and cell hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentCell:
    """One training/evaluation cell in the bounded development plan."""

    stage: str
    backbone: str
    representation: str
    seed: int
    action_schema: dict[str, Any]
    dataset_revision: str
    base_revision: str
    adaptation: str
    train_steps: int
    profile: str = DEVELOPMENT_PROFILE
    evaluation_split: str = DEVELOPMENT_SPLIT
    object_token_schema: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentCell:
        return cls(
            stage=str(data["stage"]),
            backbone=str(data["backbone"]),
            representation=str(data["representation"]),
            seed=int(data["seed"]),
            action_schema=dict(data["action_schema"]),
            dataset_revision=str(data["dataset_revision"]),
            base_revision=str(data["base_revision"]),
            adaptation=str(data["adaptation"]),
            train_steps=int(data["train_steps"]),
            profile=str(data.get("profile", DEVELOPMENT_PROFILE)),
            evaluation_split=str(
                data.get("evaluation_split", data.get("split", DEVELOPMENT_SPLIT))
            ),
            object_token_schema=data.get("object_token_schema"),
            notes=data.get("notes"),
        )

    def identity(self) -> dict[str, Any]:
        """Fields that define the cell's scientific identity."""
        identity = {
            "stage": self.stage,
            "backbone": self.backbone,
            "representation": self.representation,
            "seed": self.seed,
            "action_schema": self.action_schema,
            "dataset_revision": self.dataset_revision,
            "base_revision": self.base_revision,
            "adaptation": self.adaptation,
            "train_steps": self.train_steps,
            "profile": self.profile,
            "evaluation_split": self.evaluation_split,
            "object_token_schema": self.object_token_schema,
        }
        return {key: value for key, value in identity.items() if value is not None}

    def to_manifest_record(self) -> dict[str, Any]:
        record = self.identity()
        record["cell_hash"] = stable_hash(record)
        if self.notes:
            record["notes"] = self.notes
        return record

    def validate(self) -> None:
        if self.stage not in STAGE_GPU_HOUR_LIMITS:
            raise ValueError(f"Unknown stage {self.stage!r}")
        if not self.backbone:
            raise ValueError("Experiment cell is missing a backbone")
        if not self.representation:
            raise ValueError("Experiment cell is missing a representation")
        if not self.action_schema:
            raise ValueError("Experiment cell is missing an action_schema")
        if not self.dataset_revision:
            raise ValueError("Experiment cell is missing a dataset_revision")
        if not self.base_revision:
            raise ValueError("Experiment cell is missing a base_revision")
        if not self.adaptation:
            raise ValueError("Experiment cell is missing an adaptation")
        if self.train_steps <= 0:
            raise ValueError("Experiment cell train_steps must be positive")
        if self.profile != DEVELOPMENT_PROFILE:
            raise ValueError(
                f"Experiment planning is bounded to {DEVELOPMENT_PROFILE!r}, got {self.profile!r}"
            )
        if self.evaluation_split != DEVELOPMENT_SPLIT:
            raise ValueError(
                "Experiment planning is bounded to "
                f"{DEVELOPMENT_SPLIT!r}, got {self.evaluation_split!r}"
            )


@dataclass(frozen=True)
class ThroughputMeasurement:
    """Measured throughput for a matching training/evaluation schema."""

    stage: str
    backbone: str
    representation: str
    action_schema: dict[str, Any]
    dataset_revision: str
    base_revision: str
    adaptation: str
    measured_steps: int
    gpu_count: int
    elapsed_seconds: float
    eval_overhead_gpu_hours: float
    source: str
    batch_schema: dict[str, Any] | None = None
    full_workload_cell_hashes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThroughputMeasurement:
        if data.get("measured") is False or data.get("estimated") is True:
            raise ValueError(f"Throughput for {data.get('stage')!r} is not measured")
        return cls(
            stage=str(data["stage"]),
            backbone=str(data["backbone"]),
            representation=str(data["representation"]),
            action_schema=dict(data["action_schema"]),
            dataset_revision=str(data["dataset_revision"]),
            base_revision=str(data["base_revision"]),
            adaptation=str(data["adaptation"]),
            measured_steps=int(data["measured_steps"]),
            gpu_count=int(data["gpu_count"]),
            elapsed_seconds=float(data["elapsed_seconds"]),
            eval_overhead_gpu_hours=float(data.get("eval_overhead_gpu_hours", 0.0)),
            source=str(data["source"]),
            batch_schema=data.get("batch_schema"),
            full_workload_cell_hashes=tuple(data.get("full_workload_cell_hashes", ())),
        )

    def validate(self) -> None:
        if self.stage not in STAGE_GPU_HOUR_LIMITS:
            raise ValueError(f"Unknown throughput stage {self.stage!r}")
        if not self.backbone:
            raise ValueError("Throughput measurement is missing a backbone")
        if not self.representation:
            raise ValueError("Throughput measurement is missing a representation")
        if not self.action_schema:
            raise ValueError("Throughput measurement is missing an action_schema")
        if not self.dataset_revision:
            raise ValueError("Throughput measurement is missing a dataset_revision")
        if not self.base_revision:
            raise ValueError("Throughput measurement is missing a base_revision")
        if not self.adaptation:
            raise ValueError("Throughput measurement is missing an adaptation")
        if self.measured_steps <= 0:
            raise ValueError("Throughput measured_steps must be positive")
        if self.gpu_count <= 0:
            raise ValueError("Throughput gpu_count must be positive")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
            raise ValueError("Throughput elapsed_seconds must be finite and positive")
        if not math.isfinite(self.eval_overhead_gpu_hours) or self.eval_overhead_gpu_hours < 0:
            raise ValueError("Throughput eval_overhead_gpu_hours must be finite and nonnegative")
        if not self.source:
            raise ValueError(f"Throughput for {self.stage} needs a measurement source")

    @property
    def train_gpu_hours_per_step(self) -> float:
        return self.gpu_count * self.elapsed_seconds / 3600.0 / self.measured_steps

    def key(self) -> tuple[tuple[str, Any], ...]:
        value = {
            "stage": self.stage,
            "backbone": self.backbone,
            "representation": self.representation,
            "action_schema": self.action_schema,
            "dataset_revision": self.dataset_revision,
            "base_revision": self.base_revision,
            "adaptation": self.adaptation,
            "batch_schema": self.batch_schema,
        }
        return tuple(sorted((k, canonical_json(v)) for k, v in value.items() if v is not None))

    def matches(self, cell: ExperimentCell) -> bool:
        return self.key() == _measurement_key(cell)

    def planned_gpu_hours(self, cell: ExperimentCell) -> float:
        if cell.to_manifest_record()["cell_hash"] in self.full_workload_cell_hashes:
            train_hours = self.gpu_count * self.elapsed_seconds / 3600.0
        else:
            train_hours = cell.train_steps * self.train_gpu_hours_per_step
        return train_hours + self.eval_overhead_gpu_hours

    def to_manifest_record(self) -> dict[str, Any]:
        record = {
            "stage": self.stage,
            "backbone": self.backbone,
            "representation": self.representation,
            "action_schema": self.action_schema,
            "dataset_revision": self.dataset_revision,
            "base_revision": self.base_revision,
            "adaptation": self.adaptation,
            "batch_schema": self.batch_schema,
            "measured_steps": self.measured_steps,
            "gpu_count": self.gpu_count,
            "elapsed_seconds": self.elapsed_seconds,
            "train_gpu_hours_per_step": self.train_gpu_hours_per_step,
            "eval_overhead_gpu_hours": self.eval_overhead_gpu_hours,
            "source": self.source,
            "measured": True,
        }
        if self.full_workload_cell_hashes:
            record["full_workload_cell_hashes"] = list(self.full_workload_cell_hashes)
        return record


def build_manifest(
    cells: list[ExperimentCell], measurements: list[ThroughputMeasurement]
) -> dict[str, Any]:
    """Create a deterministic dry-run manifest with measured stage budgets."""
    if sum(STAGE_GPU_HOUR_LIMITS.values()) != TOTAL_GPU_HOUR_LIMIT:
        raise RuntimeError("Stage GPU-hour limits no longer sum to 100")

    for cell in cells:
        cell.validate()
    for measurement in measurements:
        measurement.validate()

    duplicate_stages = sorted(
        stage
        for stage in {measurement.stage for measurement in measurements}
        if sum(1 for measurement in measurements if measurement.stage == stage) > 1
    )
    if duplicate_stages:
        raise ValueError(
            "Duplicate throughput measurement for stage(s): " + ", ".join(duplicate_stages)
        )

    used_stages = sorted({cell.stage for cell in cells})
    measurement_by_stage = {measurement.stage: measurement for measurement in measurements}
    missing = [stage for stage in used_stages if stage not in measurement_by_stage]
    if missing:
        raise ValueError(
            "Measured throughput is required for every planned stage; missing "
            + ", ".join(missing)
        )

    budget_records = []
    for stage in sorted(STAGE_GPU_HOUR_LIMITS):
        measurement = measurement_by_stage.get(stage)
        measured_gpu_hours = measurement.measured_gpu_hours if measurement else 0.0
        limit = STAGE_GPU_HOUR_LIMITS[stage]
        budget_records.append(
            {
                "stage": stage,
                "limit_gpu_hours": limit,
                "measured_stage_gpu_hours": measured_gpu_hours,
                "remaining_gpu_hours": limit - measured_gpu_hours,
                "fits": measured_gpu_hours <= limit,
                "source": measurement.source if measurement else None,
            }
        )

    cell_records = sorted(
        (cell.to_manifest_record() for cell in cells),
        key=lambda record: (
            record["stage"],
            record["backbone"],
            record["representation"],
            record["seed"],
            record["cell_hash"],
        ),
    )
    throughput_records = sorted(
        (measurement.to_manifest_record() for measurement in measurements),
        key=lambda record: record["stage"],
    )
    manifest = {
        "schema_version": 1,
        "dry_run_only": True,
        "gpu_hour_limit_total": TOTAL_GPU_HOUR_LIMIT,
        "stage_gpu_hour_limits": STAGE_GPU_HOUR_LIMITS,
        "budget_scope": "measured stage workload only; no per-step extrapolation",
        "budget": budget_records,
        "throughput_measurements": throughput_records,
        "cells": cell_records,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


@dataclass(frozen=True)
class RolloutCandidate:
    checkpoint: str
    report_path: str
    seeds: tuple[int, ...]
    execution_budget: tuple[tuple[str, Any], ...]
    successes: int
    transfers_completed: int
    infeasible: int
    episodes: int

    @property
    def rank_key(self) -> tuple[int, int, int, str]:
        return (self.successes, self.transfers_completed, -self.infeasible, self.checkpoint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "report_path": self.report_path,
            "seeds": list(self.seeds),
            "execution_budget": dict(self.execution_budget),
            "episodes": self.episodes,
            "successes": self.successes,
            "transfers_completed": self.transfers_completed,
            "infeasible": self.infeasible,
        }


def load_eval_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def candidate_from_report(path: Path, report: dict[str, Any]) -> RolloutCandidate:
    _validate_development_report(path, report)
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{path}: evaluation report has no episodes")
    seeds = [int(row["seed"]) for row in episodes]
    duplicate_seeds = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
    if duplicate_seeds:
        raise ValueError(f"{path}: duplicate rollout seed(s): {duplicate_seeds}")
    transfers = sum(int(row.get("transfers_completed", 0)) for row in episodes)
    infeasible = sum(int(row.get("infeasible_steps", 0)) for row in episodes)
    return RolloutCandidate(
        checkpoint=str(report.get("checkpoint") or path),
        report_path=str(path),
        seeds=tuple(sorted(seeds)),
        execution_budget=_execution_budget(report),
        successes=sum(1 for row in episodes if bool(row.get("success"))),
        transfers_completed=transfers,
        infeasible=infeasible,
        episodes=len(episodes),
    )


def rank_rollout_candidates(reports: list[tuple[Path, dict[str, Any]]]) -> list[RolloutCandidate]:
    candidates = [candidate_from_report(path, report) for path, report in reports]
    if not candidates:
        raise ValueError("No rollout reports provided")
    seed_sets = {candidate.seeds for candidate in candidates}
    if len(seed_sets) != 1:
        pretty = {candidate.report_path: list(candidate.seeds) for candidate in candidates}
        raise ValueError(f"Rollout reports must use identical seed sets: {pretty}")
    execution_budgets = {candidate.execution_budget for candidate in candidates}
    if len(execution_budgets) != 1:
        pretty = {
            candidate.report_path: dict(candidate.execution_budget)
            for candidate in candidates
        }
        raise ValueError(f"Rollout reports must use identical execution budgets: {pretty}")
    return sorted(candidates, key=lambda candidate: candidate.rank_key, reverse=True)


def select_rollout_checkpoint(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    ranked = rank_rollout_candidates(reports)
    return {
        "selection_rule": [
            "successes desc",
            "transfers_completed desc",
            "infeasible_steps asc",
        ],
        "profile": DEVELOPMENT_PROFILE,
        "evaluation_split": DEVELOPMENT_SPLIT,
        "execution_budget": dict(ranked[0].execution_budget),
        "selected": ranked[0].to_dict(),
        "ranked": [candidate.to_dict() for candidate in ranked],
    }


def _validate_development_report(path: Path, report: dict[str, Any]) -> None:
    split = report.get("evaluation_split", report.get("split"))
    if split != DEVELOPMENT_SPLIT:
        raise ValueError(
            f"{path}: rollout checkpoint ranking requires "
            f"evaluation_split={DEVELOPMENT_SPLIT!r}; got {split!r}"
        )
    scopes = [
        split,
        report.get("scope"),
        report.get("eval_scope"),
        report.get("eval_tag"),
    ]
    lowered = {str(scope).lower() for scope in scopes if scope is not None}
    forbidden = lowered & FORBIDDEN_EVAL_SCOPES
    if forbidden:
        raise ValueError(f"{path}: rollout checkpoint ranking rejects {sorted(forbidden)}")

    summary = report.get("summary", {})
    summary_profiles = set(summary) if isinstance(summary, dict) else set()
    episodes = report.get("episodes", [])
    episode_profiles = {
        str(row.get("profile"))
        for row in episodes
        if row.get("profile") is not None
    }
    profiles = summary_profiles | episode_profiles
    if profiles != {DEVELOPMENT_PROFILE}:
        raise ValueError(
            f"{path}: rollout checkpoint ranking is only for {DEVELOPMENT_PROFILE}; "
            f"got {sorted(profiles)}"
        )
    for index, row in enumerate(episodes):
        row_split = row.get("evaluation_split", row.get("split", split))
        if row_split != split:
            raise ValueError(
                f"{path}: episode {index} evaluation_split {row_split!r} "
                f"contradicts report split {split!r}"
            )
        if row.get("profile") != DEVELOPMENT_PROFILE:
            raise ValueError(
                f"{path}: episode {index} profile {row.get('profile')!r} "
                f"contradicts {DEVELOPMENT_PROFILE!r}"
            )


def _execution_budget(report: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    budget = {
        "max_steps": report.get("max_steps"),
        "steps_per_object": report.get("steps_per_object"),
        "n_action_steps": report.get("n_action_steps"),
        "control_horizon": report.get("control_horizon"),
        "eval_tag": report.get("eval_tag"),
    }
    return tuple(sorted((key, value) for key, value in budget.items() if value is not None))
