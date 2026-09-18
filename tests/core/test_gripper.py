"""The binary column has to mean exactly what the aperture column meant. If it
does not, the 5/6 oracle replay ceiling stops being a valid reference for
anything trained on the re-encoded data."""

from __future__ import annotations

import pytest

from oct_vla.core.gripper import (
    GRIPPER_OPEN_THRESHOLD,
    binary_gripper_command,
    decode_gripper_command,
)

#: Spans both observed clusters and the empty gap between them.
APERTURES = [0.0, 0.1667, 0.30, 0.5301, 0.70, 0.8219, 0.822521, 0.8231, 0.8300, 0.8333, 1.0]


def test_binarising_then_decoding_reproduces_the_legacy_decode():
    """The round trip, which is the whole safety argument for re-encoding."""
    for aperture in APERTURES:
        legacy = decode_gripper_command(aperture, "measured_aperture")
        through_binary = decode_gripper_command(
            binary_gripper_command(aperture), "binary_command"
        )
        assert through_binary == legacy, aperture


def test_the_open_class_is_the_top_of_the_range():
    assert binary_gripper_command(0.8333) == 1.0
    assert binary_gripper_command(0.5301) == 0.0


def test_the_boundary_belongs_to_closed_on_both_paths():
    """Strictly greater, matching the bridge, so a value landing exactly on the
    threshold is treated the same by the exporter and the server."""
    assert binary_gripper_command(GRIPPER_OPEN_THRESHOLD) == 0.0
    assert decode_gripper_command(GRIPPER_OPEN_THRESHOLD, "measured_aperture") == 0.0


def test_the_binary_decision_margin_is_far_larger_than_the_aperture_one():
    """Why this exists. Against the trained policies' measured error of 0.0138
    (left) and 0.0377 (right) raw, the aperture margin of 0.0108 is smaller than
    the error on both arms; the binary margin of 0.5 is not."""
    aperture_margin = 0.8333 - GRIPPER_OPEN_THRESHOLD
    assert aperture_margin == pytest.approx(0.0108, abs=1e-4)
    assert 0.0138 / aperture_margin > 1.0        # left  gripper: decision unreliable
    assert 0.0377 / aperture_margin > 1.0        # right gripper: worse
    assert 0.0377 / 0.5 < 0.1                    # binary: error is a tenth of the margin


def test_a_noisy_prediction_still_decodes_correctly_when_binary():
    """A policy emitting 0.8 for 'open' and 0.2 for 'closed' -- far from the
    targets by aperture standards -- is unambiguous once the target is binary."""
    assert decode_gripper_command(0.8, "binary_command") == 1.0
    assert decode_gripper_command(0.2, "binary_command") == 0.0


def test_an_unknown_encoding_is_refused():
    with pytest.raises(ValueError, match="gripper_encoding must be one of"):
        decode_gripper_command(0.5, "whatever")
