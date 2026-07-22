"""
models/disturbances/atmospheric_drag.py

Atmospheric Drag Force Model

Computes the aerodynamic drag force acting on the spacecraft.

Current Model
-------------
- Constant drag coefficient
- Constant reference area
- Neglects Earth rotation

Future Models
-------------
- Free molecular flow
- Aerodynamic coefficient lookup tables
- Earth rotation
- Relative atmospheric wind
"""

import numpy as np

from models.disturbances.disturbance_state import DisturbanceState


class AtmosphericDrag:
    """
    Atmospheric drag force model.
    """

    def __init__(
        self,
        drag_coefficient: float,
        reference_area: float,
    ):

        if drag_coefficient <= 0.0:
            raise ValueError(
                "drag_coefficient must be positive."
            )

        if reference_area <= 0.0:
            raise ValueError(
                "reference_area must be positive."
            )

        self.drag_coefficient = float(
            drag_coefficient
        )

        self.reference_area = float(
            reference_area
        )

    # ==========================================================
    # Drag Force
    # ==========================================================

    def compute(
        self,
        state: DisturbanceState,
    ) -> np.ndarray:
        """
        Compute aerodynamic drag force.

        Parameters
        ----------
        state : DisturbanceState
            Current spacecraft/environment state.

        Returns
        -------
        ndarray (3,)
            Aerodynamic drag force expressed in the ECI frame [N].
        """

        density = float(state.atmospheric_density)

        velocity_eci = np.asarray(
            state.velocity_eci,
            dtype=float,
        )

        if density < 0.0:
            raise ValueError(
                "Atmospheric density must be non-negative."
            )

        if velocity_eci.shape != (3,):
            raise ValueError(
                "velocity_eci must have shape (3,)."
            )

        velocity_norm = np.linalg.norm(
            velocity_eci
        )

        if velocity_norm < 1e-12:
            return np.zeros(3)

        velocity_unit = (
            velocity_eci
            / velocity_norm
        )

        dynamic_pressure = (
            0.5
            * density
            * velocity_norm**2
        )

        drag_force = (
            dynamic_pressure
            * self.drag_coefficient
            * self.reference_area
        )

        return (
            -drag_force
            * velocity_unit
        )