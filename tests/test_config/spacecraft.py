"""
Spacecraft configuration for unit tests.
"""

import numpy as np

# ==========================================================
# Default Spacecraft Properties
# ==========================================================

DEFAULT_MASS = 250.0

DEFAULT_INERTIA = np.diag(
    [
        100.0,
        120.0,
        80.0,
    ]
)

# ==========================================================
# Default State
# ==========================================================

DEFAULT_QUATERNION = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
)

DEFAULT_ANGULAR_VELOCITY = np.zeros(3)

DEFAULT_POSITION = np.zeros(3)

DEFAULT_VELOCITY = np.zeros(3)