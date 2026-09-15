#!/usr/bin/env python
"""Merge closed-loop evaluation JSONs into one comparison across sweep arms.

Two choices here do most of the work, and both exist because ~30 episodes is a
small sample:

*Wilson*, not Wald, intervals. At success rates near 0 or 1 -- exactly where an
undertrained arm sits -- the textbook interval runs past the ends of [0, 1] and
its coverage collapses. Wilson stays inside the interval and behaves at the
boundaries.

*Paired* differences, not a difference of two independent rates. Every arm is
evaluated on identical scene seeds, so an arm's score is mostly a statement
about which scenes it drew. Pairing on (training seed, profile, scene seed)
cancels scene difficulty, and with ~30 episodes an unpaired 95% interval is
roughly +/-0.18 -- wide enough to hide the whole effect being measured.

Stdlib only: this runs in either environment, and a comparison that cannot be
rerun because of a missing wheel is a comparison nobody checks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

#: 95% two-sided normal quantile.
Z = 1.959963984540054


# ------------------------------------------------------------------ statistics


def wilson_interval(successes: int, total: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (z / denominator) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for `b` and `c` discordant pairs.

    Conditional on b + c discordant pairs, b is Binomial(b + c, 1/2) under the
    null that the two arms are equally likely to win a pair. Exact rather than
    chi-squared because the discordant count is often well under 25, where the
    chi-squared approximation is unreliable in the direction that matters --
    it overstates significance.
    """
    n = b + c
    if n == 0:
        return 1.0
    observed = min(b, c)
    tail = sum(math.exp(_log_binom(n, k) - n * math.log(2.0)) for k in range(observed + 1))
    return min(1.0, 2.0 * tail)


def paired_difference(pairs: list[tuple[bool, bool]]) -> dict:
    """Difference in success rate over matched episodes, `treatment - baseline`.

    The interval is the standard one for paired proportions: its width is set
    by the *discordant* pairs alone, which is precisely why pairing helps --
    episodes both arms win, or both lose, carry no information about which is
    better and correctly contribute nothing.
    """
    n = len(pairs)
    if n == 0:
        return {"pairs": 0}
    # b: baseline won, treatment lost. c: treatment won, baseline lost.
    b = sum(1 for base, treat in pairs if base and not treat)
    c = sum(1 for base, treat in pairs if treat and not base)
    difference = (c - b) / n
    variance = (b + c - (c - b) ** 2 / n) / (n * n)
    half = Z * math.sqrt(max(variance, 0.0))
    return {
        "pairs": n,
        "baseline_rate": sum(base for base, _ in pairs) / n,
        "treatment_rate": sum(treat for _, treat in pairs) / n,
        "difference": difference,
        "ci95": [difference - half, difference + half],
        "baseline_only_wins": b,
        "treatment_only_wins": c,
        "mcnemar_p": mcnemar_exact_p(b, c),
    }


# ----------------------------------------------------------------- arm identity


def arm_of(report: dict, run_metadata: dict[str, dict]) -> tuple[str, int]:
    """Return (arm label, training seed) for one evaluation report.

    The run's own metadata is preferred over the checkpoint path: the path says
    where weights live, while the arm is a claim about how they were trained,
    and only the training job knows that.
    """
    run_name = _run_name(Path(report["checkpoint"]))
    meta = run_metadata.get(run_name, {})
    conditioning = meta.get("conditioning")
    token_mode = meta.get("token_mode") or report.get("token_mode")
    seed = meta.get("seed")
    if conditioning is None or seed is None:
        conditioning, token_mode, seed = _parse_run_name(run_name, report, token_mode)
    label = "rgb" if conditioning == "rgb" else f"object_{token_mode or 'full'}"
    if report.get("shuffled_tokens"):
        # A separate arm, not a variant of B: same weights, different inputs.
        label += "_shuffled"
    return label, int(seed)


def _run_name(checkpoint: Path) -> str:
    """The run directory name, from any checkpoint path inside the run.

    Paths look like <root>/<run>/checkpoints/<step>/pretrained_model, so walk
    up to whatever sits directly above `checkpoints`.
    """
    for parent in checkpoint.parents:
        if parent.name == "checkpoints":
            return parent.parent.name
    return checkpoint.name


def _parse_run_name(
    run_name: str, report: dict, token_mode: str | None
) -> tuple[str, str | None, int]:
    """Fall back to the `pi05_{variant}_{mode}_s{seed}` naming convention."""
    parts = run_name.split("_")
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
        middle = parts[1:-1]
        if middle:
            return middle[0], "_".join(middle[1:]) or token_mode, seed
        return ("object" if report.get("object_tokens") else "rgb"), token_mode, seed
    raise SystemExit(
        f"Cannot determine the arm for run {run_name!r}. Expected run metadata under "
        "$OCTVLA_OUTPUT_ROOT/run_metadata/, or a pi05_{variant}_{mode}_s{seed} directory name."
    )


# --------------------------------------------------------------------- loading


def load_run_metadata(roots: list[Path]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for root in roots:
        for path in sorted(root.glob("*.json")):
            try:
                metadata[path.stem] = json.loads(path.read_text())
            except json.JSONDecodeError as error:
                print(f"warning: skipping malformed {path}: {error}", file=sys.stderr)
    return metadata


def recomputed_summary(report: dict) -> dict:
    """Rebuild a report's `summary` from its own episode rows.

    Used by --verify. The doc's rule is to check this script reproduces numbers
    an existing eval JSON already contains before trusting anything new it
    says, because every later comparison is built on this same episode parsing.
    """
    summary: dict[str, dict] = {}
    for profile in dict.fromkeys(row["profile"] for row in report["episodes"]):
        rows = [row for row in report["episodes"] if row["profile"] == profile]
        summary[profile] = {
            "episodes": len(rows),
            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
            "mean_transfers": sum(row["transfers_completed"] for row in rows) / len(rows),
        }
    return summary


def verify(reports: list[tuple[Path, dict]]) -> int:
    failures = 0
    for path, report in reports:
        stored, rebuilt = report.get("summary", {}), recomputed_summary(report)
        for profile, expected in stored.items():
            actual = rebuilt.get(profile)
            if actual is None:
                print(f"FAIL {path.name}: {profile} missing from recomputed summary")
                failures += 1
                continue
            for key, value in expected.items():
                if not math.isclose(float(value), float(actual[key]), rel_tol=1e-9, abs_tol=1e-9):
                    print(f"FAIL {path.name}: {profile}.{key} stored {value} != {actual[key]}")
                    failures += 1
        print(f"{'FAIL' if failures else 'ok  '} {path.name}: {len(stored)} profile(s) checked")
    print(f"\n{'FAILED' if failures else 'OK'}: {failures} mismatch(es)")
    return 1 if failures else 0


# ------------------------------------------------------------------ aggregation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Evaluation JSONs, or directories")
    parser.add_argument(
        "--run-metadata",
        type=Path,
        action="append",
        default=[],
        help="Directory of run_metadata/*.json (repeatable)",
    )
    parser.add_argument("--baseline", default="rgb", help="Arm every other arm is compared to")
    parser.add_argument("--output", type=Path, help="Write the merged report here")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only check that each report's stored summary is reproduced from its episodes",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for entry in args.reports:
        paths.extend(sorted(entry.glob("*.json")) if entry.is_dir() else [entry])
    if not paths:
        raise SystemExit("No evaluation JSONs found")
    reports = [(path, json.loads(path.read_text())) for path in paths]

    if args.verify:
        return verify(reports)

    metadata = load_run_metadata(args.run_metadata)

    # (arm, training seed, profile, scene seed) -> success / transfers. Keyed
    # this tightly so pairing can match episodes exactly; a duplicate key means
    # the same cell was evaluated twice and silently averaging the two would
    # hide whichever run was broken.
    episodes: dict[tuple, dict] = {}
    arms: set[str] = set()
    for path, report in reports:
        arm, seed = arm_of(report, metadata)
        arms.add(arm)
        for row in report["episodes"]:
            key = (arm, seed, row["profile"], row["seed"])
            if key in episodes:
                print(f"warning: duplicate episode {key} (also in {path.name})", file=sys.stderr)
            episodes[key] = row

    def rows_for(arm: str, profile: str | None = None) -> list[dict]:
        return [
            row
            for (a, _, p, _), row in episodes.items()
            if a == arm and (profile is None or p == profile)
        ]

    profiles = sorted({key[2] for key in episodes})
    per_arm = {}
    for arm in sorted(arms):
        per_arm[arm] = {"overall": _describe(rows_for(arm))}
        # The count-shift breakdown: the same policy on 2/3/4 objects, which is
        # the sweep's free generalisation axis.
        per_arm[arm]["per_profile"] = {p: _describe(rows_for(arm, p)) for p in profiles}

    comparisons = {}
    if args.baseline in arms:
        for arm in sorted(arms - {args.baseline}):
            shared = [
                key[1:]
                for key in episodes
                if key[0] == arm and (args.baseline, *key[1:]) in episodes
            ]
            pairs = [
                (
                    bool(episodes[(args.baseline, *key)]["success"]),
                    bool(episodes[(arm, *key)]["success"]),
                )
                for key in sorted(shared)
            ]
            comparisons[f"{arm}_vs_{args.baseline}"] = paired_difference(pairs)
            unmatched = len(rows_for(arm)) - len(pairs)
            if unmatched:
                print(
                    f"warning: {arm} has {unmatched} episode(s) with no {args.baseline} "
                    "counterpart; they are excluded from the paired comparison",
                    file=sys.stderr,
                )
    elif arms:
        print(f"warning: baseline arm {args.baseline!r} not found; skipping comparisons",
              file=sys.stderr)

    merged = {
        "reports": [str(path) for path in paths],
        "arms": per_arm,
        "paired_comparisons": comparisons,
    }
    _print(merged)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(merged, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


def _describe(rows: list[dict]) -> dict:
    if not rows:
        return {"episodes": 0}
    successes = sum(bool(row["success"]) for row in rows)
    low, high = wilson_interval(successes, len(rows))
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "wilson95": [low, high],
        # Partial credit, already recorded per episode: a policy that moves two
        # of three objects is not the same as one that never grasps anything,
        # and success rate alone cannot tell them apart.
        "mean_transfers": sum(row["transfers_completed"] for row in rows) / len(rows),
    }


def _print(merged: dict) -> None:
    print(f"{'arm':<32} {'n':>4} {'success':>8}  {'95% Wilson':<18} {'transfers':>9}")
    print("-" * 78)
    for arm, stats in merged["arms"].items():
        overall = stats["overall"]
        if not overall.get("episodes"):
            continue
        low, high = overall["wilson95"]
        print(
            f"{arm:<32} {overall['episodes']:>4} {overall['success_rate']:>8.3f}  "
            f"[{low:.3f}, {high:.3f}]      {overall['mean_transfers']:>9.2f}"
        )
    for arm, stats in merged["arms"].items():
        per_profile = {p: s for p, s in stats["per_profile"].items() if s.get("episodes")}
        if len(per_profile) > 1:
            shift = "  ".join(f"{p}={s['success_rate']:.3f}" for p, s in per_profile.items())
            print(f"  count shift {arm:<20} {shift}")
    if merged["paired_comparisons"]:
        print(f"\n{'paired comparison':<32} {'pairs':>5} {'diff':>7}  {'95% CI':<18} {'p':>7}")
        print("-" * 78)
        for name, stats in merged["paired_comparisons"].items():
            if not stats.get("pairs"):
                continue
            low, high = stats["ci95"]
            print(
                f"{name:<32} {stats['pairs']:>5} {stats['difference']:>+7.3f}  "
                f"[{low:+.3f}, {high:+.3f}]    {stats['mcnemar_p']:>7.4f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
