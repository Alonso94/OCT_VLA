"""The representative video is defined by rule, never the best episode."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "select_rollout_videos", ROOT / "scripts/select_rollout_videos.py"
)
videos = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(videos)


def episode(seed, scene, transfers, lifted=0):
    return {"train_seed": seed, "eval_seed": scene, "transfers": float(transfers),
            "lifted": lifted, "success": transfers >= 3, "tier": "seen", "video": ""}


def test_the_median_seed_is_chosen_not_the_lucky_one():
    episodes = (
        [episode(1000, 800 + i, t) for i, t in enumerate([3, 3, 3, 1])]   # lucky seed
        + [episode(1001, 800 + i, t) for i, t in enumerate([0, 0, 0, 1])]
        + [episode(1002, 800 + i, t) for i, t in enumerate([1, 0, 1, 0])]
    )
    chosen, summary = videos.representative(episodes)
    assert summary["seed"] == 1002
    # Seed mean 0.5: 0 and 1 are equally close, so lifts then scene seed decide.
    assert chosen["train_seed"] == 1002 and chosen["eval_seed"] == 800


def test_the_typical_episode_matches_the_seed_mean():
    episodes = [episode(1000, 800 + i, t, lifted=l)
                for i, (t, l) in enumerate([(0, 0), (1, 2), (1, 1), (3, 3)])]
    episodes += [episode(1001, 800, 0), episode(1002, 800, 5)]
    chosen, _ = videos.representative(episodes)
    # Mean transfers 1.25 -> an episode with one; mean lifts 1.5 -> tie, lower scene.
    assert (chosen["transfers"], chosen["eval_seed"]) == (1.0, 801)


def test_two_seeds_take_the_lower_middle():
    episodes = [episode(1000, 800, 2), episode(1001, 800, 0)]
    _, summary = videos.representative(episodes)
    assert summary["seed"] == 1001
