"""Recorded demonstrations as corpora: runs grouped by the rule they were collected under."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from oct_vla.bpfn.data.real_data.robotwin import read_recording, recorded_metadata
from oct_vla.bpfn.schema import Episode


def find_runs(root: Path) -> list[Path]:
    """Every recorded run below ``root`` (``.../seed_*/episode_*`` with an episode.json)."""
    runs = []
    for path in Path(root).glob("**/seed_*/episode_*"):
        if (path / "episode.json").exists():
            runs.append(path)
    return sorted(runs)


class RealData:
    """Recorded runs per rule, read on demand with a bounded cache.

    Args:
        roots: rule -> collection roots whose runs were recorded under that rule.
        check: how many runs per rule to check against their recorded rule.
        cache_size: episodes kept in memory (bounded for the login nodes).
    """

    def __init__(self, roots: dict[str, list[Path]], check: int = 3, cache_size: int = 24) -> None:
        self.runs: dict[str, list[Path]] = {}
        for rule, rule_roots in roots.items():
            runs = []
            for root in rule_roots:
                runs.extend(find_runs(root))
            self._check_rule(rule, runs[:check])
            self.runs[rule] = runs
        self.cache_size = cache_size
        self._cache: OrderedDict[Path, Episode] = OrderedDict()

    @staticmethod
    def _check_rule(rule: str, runs: list[Path]) -> None:
        for path in runs:
            recorded = recorded_metadata(path).get("rule")
            if recorded != rule:
                raise SystemExit(f"{path} was collected under rule {recorded!r}, not {rule!r}")

    @property
    def rules(self) -> list[str]:
        return list(self.runs)

    def episode(self, path: Path) -> Episode:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        episode = read_recording(path)
        self._cache[path] = episode
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode
