"""The G1b reliability table (scripts/prior_shelf_g1b.py)."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prior_shelf_g1b", Path(__file__).parents[1] / "scripts/prior_shelf_g1b.py")
g1b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1b)


def test_a_calibrated_source_has_zero_error_and_an_overconfident_one_does_not():
    calibrated = [(0.9, True)] * 9 + [(0.9, False)] + [(0.6, True)] * 6 + [(0.6, False)] * 4
    table = g1b._calibration(calibrated)
    assert table["ece"] == pytest.approx(0.0)
    assert [b["n"] for b in table["bins"]] == [10, 10]
    overconfident = [(0.99, False)] * 8 + [(0.99, True)] * 2
    assert g1b._calibration(overconfident)["ece"] == pytest.approx(0.79)
    assert g1b._calibration([]) == {}


def test_stages_split_the_compaction_out():
    assert g1b._stage("compact push") == g1b._stage("park compactor") == "compaction"
    assert g1b._stage("close") == "to_grasp" and g1b._stage("release") == "carry"
