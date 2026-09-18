"""The sweep's conclusion is whatever this script prints, so the statistics are
checked against values computed independently rather than against themselves."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("aggregate_eval", ROOT / "scripts/aggregate_eval.py")
aggregate_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aggregate_eval)


def episode(seed, profile, success, transfers=0):
    return {
        "seed": seed,
        "profile": profile,
        "success": success,
        "steps": 600,
        "transfers_completed": transfers,
        "reason": "success" if success else "step_limit",
    }


# ------------------------------------------------------------------- intervals


def test_wilson_matches_a_known_value():
    # 10 successes in 30 episodes -- the sweep's actual sample size. The
    # interval is asymmetric about the observed 0.333, which is the part Wald
    # gets wrong. Values cross-checked against the closed form by hand.
    low, high = aggregate_eval.wilson_interval(10, 30)
    assert low == pytest.approx(0.192305, abs=1e-6)
    assert high == pytest.approx(0.512199, abs=1e-6)


def test_wilson_is_wide_enough_to_justify_pairing():
    """The doc's warning, as a test: an unpaired interval at this sample size
    is about +/-0.18, wide enough to hide the effect the sweep measures."""
    low, high = aggregate_eval.wilson_interval(15, 30)
    assert (high - low) / 2 == pytest.approx(0.18, abs=0.02)


def test_wilson_stays_inside_the_unit_interval_at_the_boundary():
    """The reason for preferring Wilson: Wald gives [0, 0] for 0/30, claiming
    certainty from a sample that has merely never succeeded."""
    low, high = aggregate_eval.wilson_interval(0, 30)
    assert low == 0.0
    assert 0.0 < high < 0.2
    low, high = aggregate_eval.wilson_interval(30, 30)
    assert high <= 1.0 and high == pytest.approx(1.0)
    assert 0.8 < low < 1.0


def test_wilson_of_an_empty_sample_is_maximally_uncertain():
    assert aggregate_eval.wilson_interval(0, 0) == (0.0, 1.0)


# ---------------------------------------------------------------------- paired


def test_mcnemar_is_symmetric_and_exact():
    # b=0, c=6: all six discordant pairs fall one way. Two-sided exact p is
    # 2 * (1/2)^6 = 0.03125.
    assert aggregate_eval.mcnemar_exact_p(0, 6) == pytest.approx(0.03125)
    assert aggregate_eval.mcnemar_exact_p(6, 0) == pytest.approx(0.03125)


def test_mcnemar_without_discordant_pairs_is_uninformative():
    assert aggregate_eval.mcnemar_exact_p(0, 0) == 1.0


def test_paired_difference_counts_only_discordant_pairs():
    # 10 pairs: 3 where only the treatment wins, 1 where only the baseline
    # does, 6 concordant. The difference is (3 - 1) / 10.
    pairs = [(False, True)] * 3 + [(True, False)] + [(True, True)] * 4 + [(False, False)] * 2
    result = aggregate_eval.paired_difference(pairs)
    assert result["pairs"] == 10
    assert result["difference"] == pytest.approx(0.2)
    assert result["treatment_only_wins"] == 3
    assert result["baseline_only_wins"] == 1
    assert result["baseline_rate"] == pytest.approx(0.5)
    assert result["treatment_rate"] == pytest.approx(0.7)


def test_concordant_pairs_narrow_the_interval():
    """Pairing's whole benefit: episodes both arms win carry no information
    about which is better, so adding them must not widen the interval."""
    discordant = [(False, True)] * 3 + [(True, False)]
    narrow = aggregate_eval.paired_difference(discordant + [(True, True)] * 20)
    wide = aggregate_eval.paired_difference(discordant)
    assert (narrow["ci95"][1] - narrow["ci95"][0]) < (wide["ci95"][1] - wide["ci95"][0])


def test_paired_difference_of_nothing_is_reported_not_crashed():
    assert aggregate_eval.paired_difference([]) == {"pairs": 0}


# ------------------------------------------------------------------- identity


def test_run_name_is_taken_from_above_the_checkpoints_directory():
    path = Path("/vault/out/pi05_object_role_stripped_s1001/checkpoints/006000/pretrained_model")
    assert aggregate_eval._run_name(path) == "pi05_object_role_stripped_s1001"


def test_arm_prefers_run_metadata_over_the_path():
    report = {"checkpoint": "/out/renamed_dir/checkpoints/002000/pretrained_model"}
    metadata = {"renamed_dir": {"conditioning": "object", "token_mode": "full", "seed": 1002}}
    assert aggregate_eval.arm_of(report, metadata) == ("object_full", 1002)


def test_arm_falls_back_to_the_naming_convention():
    checkpoint = "/out/pi05_object_role_stripped_s1001/checkpoints/last/pretrained_model"
    assert aggregate_eval.arm_of({"checkpoint": checkpoint}, {}) == ("object_role_stripped", 1001)


def test_shuffled_control_is_a_separate_arm_from_the_weights_it_reuses():
    """B and its shuffled control share a checkpoint, so nothing in the path
    distinguishes them; merging the two would dilute both."""
    checkpoint = "/out/pi05_object_full_s1000/checkpoints/last/pretrained_model"
    plain = aggregate_eval.arm_of({"checkpoint": checkpoint}, {})
    shuffled = aggregate_eval.arm_of({"checkpoint": checkpoint, "shuffled_tokens": True}, {})
    assert plain == ("object_full", 1000)
    assert shuffled == ("object_full_shuffled", 1000)


def test_an_unparseable_run_name_fails_loudly():
    with pytest.raises(SystemExit, match="Cannot determine the arm"):
        aggregate_eval.arm_of({"checkpoint": "/out/some-run/checkpoints/last"}, {})


# --------------------------------------------------------------------- verify


def test_recomputed_summary_reproduces_a_real_report(tmp_path):
    """The doc's rule: this script must reproduce numbers an eval JSON already
    contains before anything new it says is trusted."""
    episodes = [
        episode(800, "three_object", True, 3),
        episode(801, "three_object", False, 1),
        episode(800, "two_object", True, 2),
    ]
    report = {
        "episodes": episodes,
        "summary": {
            "three_object": {"episodes": 2, "success_rate": 0.5, "mean_transfers": 2.0},
            "two_object": {"episodes": 1, "success_rate": 1.0, "mean_transfers": 2.0},
        },
    }
    assert aggregate_eval.recomputed_summary(report) == report["summary"]
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(report))
    assert aggregate_eval.verify([(path, report)]) == 0


def test_verify_reports_a_summary_that_does_not_match_its_episodes(tmp_path):
    report = {
        "episodes": [episode(800, "three_object", False)],
        "summary": {"three_object": {"episodes": 1, "success_rate": 1.0, "mean_transfers": 0.0}},
    }
    assert aggregate_eval.verify([(tmp_path / "eval.json", report)]) == 1


def test_a_privileged_baseline_is_not_relabelled_as_an_object_arm():
    """An earlier version mapped everything that was not 'rgb' onto 'object_*',
    which would have merged the vision-free baseline into the object arm."""
    report = {"checkpoint": "/out/act_three_object_privileged_s1000/checkpoints/last/pm"}
    meta = {"act_three_object_privileged_s1000": {
        "conditioning": "privileged", "token_mode": "full", "seed": 1000, "backbone": "act"}}
    assert aggregate_eval.arm_of(report, meta) == ("act_privileged", 1000)


def test_the_backbone_appears_in_the_label_when_it_is_not_pi05():
    report = {"checkpoint": "/out/act_three_object_rgb_s1000/checkpoints/last/pm"}
    meta = {"act_three_object_rgb_s1000": {
        "conditioning": "rgb", "token_mode": "full", "seed": 1000, "backbone": "act"}}
    assert aggregate_eval.arm_of(report, meta) == ("act_rgb", 1000)


def test_pi05_arms_keep_their_bare_labels():
    """The sweep's own arms must not be renamed, or they stop matching the
    baseline used for the paired comparison."""
    report = {"checkpoint": "/out/pi05_object_full_s1000/checkpoints/last/pm"}
    meta = {"pi05_object_full_s1000": {
        "conditioning": "object", "token_mode": "full", "seed": 1000, "backbone": "pi05"}}
    assert aggregate_eval.arm_of(report, meta) == ("object_full", 1000)


def test_a_different_execution_horizon_is_a_separate_arm():
    """An open-loop horizon of 50 and a closed-loop 1 are not the same policy in
    the environment, even though they share weights. Averaging them would hide
    the effect the horizon sweep exists to measure."""
    checkpoint = "/out/act_three_object_rgb_r75_abs_s1000/checkpoints/best/pm"
    meta = {"act_three_object_rgb_r75_abs_s1000": {
        "conditioning": "rgb", "token_mode": "full", "seed": 1000, "backbone": "act"}}
    plain = aggregate_eval.arm_of({"checkpoint": checkpoint}, meta)
    closed = aggregate_eval.arm_of({"checkpoint": checkpoint, "eval_tag": "n1"}, meta)
    assert plain == ("act_rgb", 1000)
    assert closed == ("act_rgb__n1", 1000)


def test_an_empty_tag_stays_comparable_with_older_reports():
    """Reports written before the horizon override existed carry no eval_tag,
    and a re-run at defaults must still merge with them."""
    checkpoint = "/out/act_three_object_rgb_r75_abs_s1000/checkpoints/best/pm"
    meta = {"act_three_object_rgb_r75_abs_s1000": {
        "conditioning": "rgb", "token_mode": "full", "seed": 1000, "backbone": "act"}}
    assert (aggregate_eval.arm_of({"checkpoint": checkpoint, "eval_tag": ""}, meta)
            == aggregate_eval.arm_of({"checkpoint": checkpoint}, meta))
