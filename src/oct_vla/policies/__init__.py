"""Optional LeRobot policy plugins.

Importing this package registers the lightweight configuration classes.  Model
imports remain lazy until LeRobot resolves the selected policy type.
"""

from .control_pi05.configuration_control_pi05 import ControlPI05Config
from .control_smolvla.configuration_control_smolvla import ControlSmolVLAConfig
from .control_vla_jepa.configuration_control_vla_jepa import ControlVLAJEPAConfig
from .control_xvla.configuration_control_xvla import ControlXVLAConfig
from .masked_pi05.configuration_masked_pi05 import MaskedPI05Config

__all__ = [
    "ControlPI05Config",
    "ControlSmolVLAConfig",
    "ControlVLAJEPAConfig",
    "ControlXVLAConfig",
    "MaskedPI05Config",
]
