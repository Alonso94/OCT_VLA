"""The gripper is a binary actuator; the recordings hold a measured aperture.

Collection records what the fingers *were*, not what the oracle *asked for* --
the commanded state is never written to disk. The aperture is bimodal, so the
command is recoverable by thresholding, and the bridge has always done that at
execution time.

Doing it only at execution time is the problem. The measured aperture stalls at
whatever the grasped object is wide, so the "open" class is a band 0.0108 wide
at the very top of the range while "closed" spreads over 0.29 -- and 37-73 % of
frames sit in that thin band. A policy regressing the aperture under MEAN_STD
then has to land inside 0.0108 to command "open". Measured on the trained
checkpoints, its error against that margin is:

    left gripper   0.0138 raw vs 0.0108 margin  = 1.28x
    right gripper  0.0377 raw vs 0.0108 margin  = 3.48x

Above 1.0 on both arms, so the open/close decision is wrong more often than
right -- which is what `mean_transfers = 0.00` looks like when the arm reaches
the object.

The fix is to threshold at *export* instead, so the policy regresses a
well-separated {0, 1} target and the decode threshold sits at 0.5, far from
both modes. The classes are then about 1.0 sigma from the boundary against the
same 0.39 sigma error.

Thresholding with the bridge's own constant, not a new one: the binary column
must decode to exactly the command the aperture column already decoded to, or
the 5/6 oracle replay ceiling in docs/control_space_comparison.md stops being a
valid reference for anything trained on it.
"""

from __future__ import annotations

#: Midpoint of the widest empty interval in the observed aperture distribution
#: (0.821989 to 0.823055). Measured, not chosen: the gap is where the bimodal
#: distribution separates, and picking a round number by eye put a real sample
#: within 1e-4 of the boundary, which decoded a full grip as half-open.
GRIPPER_OPEN_THRESHOLD = 0.822522

#: How a dataset's gripper *action* column is encoded, recorded in
#: meta/info.json beside control_space. Absent means "measured_aperture": every
#: dataset built before this existed holds the raw measurement.
GRIPPER_ENCODINGS = ("measured_aperture", "binary_command")


def binary_gripper_command(aperture: float) -> float:
    """The open/close command a measured aperture implies: 1.0 open, 0.0 closed.

    Strictly greater than the threshold, matching the bridge's decode, so a
    value landing exactly on it is treated the same way by both.
    """
    return 1.0 if aperture > GRIPPER_OPEN_THRESHOLD else 0.0


def decode_gripper_command(requested: float, encoding: str = "measured_aperture") -> float:
    """Turn a policy's gripper output into a drive command in [0, 1].

    For `binary_command` the value is already a command, so the boundary sits at
    the midpoint of the two classes rather than at the top of an aperture range.
    For `measured_aperture` the legacy threshold applies, which keeps every
    checkpoint trained before this change executing exactly as it did.
    """
    if encoding == "binary_command":
        return 1.0 if requested >= 0.5 else 0.0
    if encoding != "measured_aperture":
        raise ValueError(
            f"gripper_encoding must be one of {GRIPPER_ENCODINGS}, got {encoding!r}"
        )
    return 1.0 if requested > GRIPPER_OPEN_THRESHOLD else 0.0
