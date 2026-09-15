"""Optional LeRobot policy plugins.

Importing this package registers the lightweight configuration classes.  Model
imports remain lazy until LeRobot resolves the selected policy type.
"""

from .control_pi05.configuration_control_pi05 import ControlPI05Config

__all__ = ["ControlPI05Config"]
