"""Stock ACT with training-time image augmentation aimed at the wrist cameras.

Measured on the full-run policy (scripts/diagnose_camera_reliance.py): the
left arm's predicted grasp misses the oracle's by 11-13 mm in the horizontal
plane, and moves 170-200 mm when the head frame is swapped against 7-34 mm for
its own wrist camera. The policy grasps from the head camera, where a 5-7 cm
box spans ~10-15 px (~5 mm per pixel), while the gripper leaves a few
millimetres of clearance. Two ways to push the alignment onto the wrists:

* `head_dropout`: per training sample, with this probability the cameras in
  `dropout_cameras` are replaced by the dataset mean (zero after MEAN_STD
  normalisation), so the grasp has to be solvable from the wrist views;
* `shift_pad`: DrQ-style random shift of every camera by up to this many
  pixels, so a memorised head-image pixel position stops predicting the grasp.

Both are training-only; evaluation sees every camera, unshifted.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("aug_act")
@dataclass
class AugACTConfig(ACTConfig):
    head_dropout: float = 0.0
    dropout_cameras: list[str] = field(default_factory=lambda: ["observation.images.head"])
    shift_pad: int = 0

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")
        if self.shift_pad < 0:
            raise ValueError("shift_pad must be non-negative")
