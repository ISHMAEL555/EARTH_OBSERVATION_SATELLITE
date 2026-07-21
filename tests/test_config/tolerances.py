"""
Numerical tolerances for verification and validation.
"""

# ==========================================================
# Absolute Tolerances
# ==========================================================

ATOL_POSITION = 1e-5          # metres
ATOL_VELOCITY = 1e-5          # m/s
ATOL_ACCELERATION = 1e-8      # m/s²

ATOL_ANGULAR_VELOCITY = 1e-10
ATOL_ANGULAR_ACCELERATION = 1e-10

ATOL_QUATERNION = 1e-12
ATOL_DCM = 1e-10

ATOL_FORCE = 1e-12
ATOL_TORQUE = 1e-12

ATOL_DOT_PRODUCT = 1e-5

# ==========================================================
# Relative Tolerances
# ==========================================================

RTOL_DEFAULT = 1e-10
RTOL_MONTE_CARLO = 1e-2