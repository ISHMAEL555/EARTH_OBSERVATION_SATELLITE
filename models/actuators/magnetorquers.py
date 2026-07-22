"""
models/actuators/magnetorquers.py
=================================

Three-Axis Magnetorquer Assembly

This module implements an ideal three-axis magnetorquer assembly for
spacecraft attitude control simulations.

The assembly generates magnetic control torque according to

    τ = m × B

where

    τ : Generated body torque [N·m]
    m : Magnetic dipole moment [A·m²]
    B : Geomagnetic field in the body frame [Tesla]

Current Model
-------------
✓ Ideal actuator
✓ Dipole saturation
✓ Internal actuator state
✓ Configuration-driven
✓ Ready for future first-order actuator dynamics

Future Extensions
-----------------
- First-order coil dynamics
- Electrical current model
- Voltage model
- Power consumption
- Thermal model
- Coil failures
- Axis failures
- Misalignment errors

References
----------
Markley & Crassidis
Wie
ECSS-E-ST-60-30C
"""

import numpy as np

from config import ACTUATORS


class Magnetorquers:
    """
    Three-axis orthogonal magnetorquer assembly.
    """

    def __init__(self):
        """
        Initialize the magnetorquer assembly.
        """

        cfg = ACTUATORS["magnetorquers"]

        # =====================================================
        # Configuration
        # =====================================================

        self.num_rods = cfg["num_rods"]
        self.axes = np.asarray(cfg["axes"], dtype=float)
        self.max_dipole = float(cfg["max_dipole"])

        # =====================================================
        # Internal States
        # =====================================================

        self.commanded_dipole = np.zeros(3)
        self.actual_dipole = np.zeros(3)
        self.body_torque = np.zeros(3)

    # =========================================================
    # Public Interface
    # =========================================================

    def update(
        self,
        commanded_dipole: np.ndarray,
        magnetic_field_body: np.ndarray,
        dt: float,
    ) -> None:
        """
        Advance the actuator by one simulation step.

        Parameters
        ----------
        commanded_dipole : ndarray(3,)
            Desired magnetic dipole [A·m²].

        magnetic_field_body : ndarray(3,)
            Earth's magnetic field in the body frame [Tesla].

        dt : float
            Simulation time step [s].
            Reserved for future actuator dynamics.
        """

        commanded_dipole = self._validate_vector(
            commanded_dipole,
            "commanded_dipole"
        )

        magnetic_field_body = self._validate_vector(
            magnetic_field_body,
            "magnetic_field_body"
        )

        # Store command

        self.commanded_dipole = commanded_dipole.copy()

        # Apply saturation

        self.actual_dipole = np.clip(
            self.commanded_dipole,
            -self.max_dipole,
            self.max_dipole,
        )

        # Compute magnetic torque

        self.body_torque = np.cross(
            self.actual_dipole,
            magnetic_field_body,
        )

    # =========================================================
    # Utility Methods
    # =========================================================

    def reset(self) -> None:
        """
        Reset actuator states.
        """

        self.commanded_dipole.fill(0.0)
        self.actual_dipole.fill(0.0)
        self.body_torque.fill(0.0)

    @staticmethod
    def _validate_vector(
        vector: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """
        Validate a 3-element vector.
        """

        vector = np.asarray(vector, dtype=float)

        if vector.shape != (3,):
            raise ValueError(
                f"{name} must be a 3-element vector."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} contains invalid values."
            )

        return vector