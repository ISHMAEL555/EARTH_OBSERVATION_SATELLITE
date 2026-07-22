"""
models/disturbances/atmospheric_drag.py

Atmospheric Drag Force Model

Computes the aerodynamic drag force acting on the spacecraft.

This model is independent of:

- Orbit propagation
- Atmospheric model
- Spacecraft attitude
- Controllers
- Sensors
- Actuators

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


class AtmosphericDrag:
    """
    Atmospheric drag force model.

    Parameters
    ----------
    drag_coefficient : float
        Spacecraft drag coefficient.

    reference_area : float
        Effective aerodynamic reference area [m²].
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
        density: float,
        spacecraft_velocity_eci: np.ndarray,
    ) -> np.ndarray:
        """
        Compute aerodynamic drag force.

        Parameters
        ----------
        density : float
            Atmospheric density [kg/m³].

        spacecraft_velocity_eci : ndarray (3,)
            Spacecraft velocity in the ECI frame [m/s].

        Returns
        -------
        drag_force_eci : ndarray (3,)
            Aerodynamic drag force expressed in the
            ECI frame [N].
        """

        if density < 0.0:
            raise ValueError(
                "density must be non-negative."
            )

        spacecraft_velocity_eci = np.asarray(
            spacecraft_velocity_eci,
            dtype=float,
        )

        if spacecraft_velocity_eci.shape != (3,):
            raise ValueError(
                "spacecraft_velocity_eci must have shape (3,)."
            )

        velocity_magnitude = np.linalg.norm(
            spacecraft_velocity_eci
        )

        if velocity_magnitude < 1e-12:
            return np.zeros(3)

        velocity_unit_vector = (
            spacecraft_velocity_eci
            / velocity_magnitude
        )

        dynamic_pressure = (
            0.5
            * density
            * velocity_magnitude**2
        )

        drag_force = (
            dynamic_pressure
            * self.drag_coefficient
            * self.reference_area
        )

        drag_force_eci = (
            -drag_force
            * velocity_unit_vector
        )

        return drag_force_eci