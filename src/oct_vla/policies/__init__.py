"""Optional LeRobot policy plugins.

Importing this package registers the lightweight configuration classes.  Model
imports remain lazy until LeRobot resolves the selected policy type.
"""

from .aug_act.configuration_aug_act import AugACTConfig
from .control_act.configuration_control_act import ControlACTConfig
from .control_groot.configuration_control_groot import ControlGrootConfig
from .control_pi05.configuration_control_pi05 import ControlPI05Config
from .control_smolvla.configuration_control_smolvla import ControlSmolVLAConfig
from .control_vla_jepa.configuration_control_vla_jepa import ControlVLAJEPAConfig
from .history_act.configuration_history_act import HistoryACTConfig
from .masked_pi05.configuration_masked_pi05 import MaskedPI05Config
from .rel_act.configuration_rel_act import RelACTConfig
from .slim_groot.configuration_slim_groot import SlimGrootConfig

__all__ = [
    "AugACTConfig",
    "ControlACTConfig",
    "ControlGrootConfig",
    "ControlPI05Config",
    "ControlSmolVLAConfig",
    "ControlVLAJEPAConfig",
    "HistoryACTConfig",
    "MaskedPI05Config",
    "RelACTConfig",
    "SlimGrootConfig",
]
