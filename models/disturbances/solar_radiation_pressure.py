"""
models/disturbances/solar_radiation_pressure.py

Solar Radiation Pressure Force Model

Computes the solar radiation pressure force acting on the spacecraft.

This model is independent of:

- Orbit propagation
- Spacecraft attitude
- Controllers
- Sensors
- Actuators

Current Model
-------------
- Constant solar radiation pressure at 1 AU
- Constant reflectivity coefficient
- Constant illuminated area
- Fixed Sun direction
- No eclipse modelling

Future Models
-------------
- Eclipse model
- Variable solar flux
- Surface optical properties
- Self-shadowing
- Articulated solar arrays

References
----------
- Wertz, Spacecraft Attitude Determination and Control
- Fortescue, Spacecraft Systems Engineering
"""

import numpy as np


class SolarRadiationPressure:
    """
    Solar Radiation Pressure (SRP) force model.

    Parameters
    ----------
    solar_radiation_pressure : float
        Solar radiation pressure at 1 AU [N/m²].

    reflectivity_coefficient : float
        Surface reflectivity coefficient.

    reference_area : float
        Effective illuminated area [m²].

    sun_direction_eci : ndarray (3,)
        Unit vector pointing from Earth toward the Sun
        expressed in the ECI frame.
    """

    def __init__(
        self,
        solar_radiation_pressure: float,
        reflectivity_coefficient: float,
        reference_area: float,
        sun_direction_eci: np.ndarray,
    ):

        if solar_radiation_pressure <= 0.0:
            raise ValueError(
                "solar_radiation_pressure must be positive."
            )

        if reflectivity_coefficient <= 0.0:
            raise ValueError(
                "reflectivity_coefficient must be positive."
            )

        if reference_area <= 0.0:
            raise ValueError(
                "reference_area must be positive."
            )

        self.solar_radiation_pressure = float(
            solar_radiation_pressure
        )

        self.reflectivity_coefficient = float(
            reflectivity_coefficient
        )

        self.reference_area = float(
            reference_area
        )

        self.sun_direction_eci = np.asarray(
            sun_direction_eci,
            dtype=float,
        )

        if self.sun_direction_eci.shape != (3,):
            raise ValueError(
                "sun_direction_eci must have shape (3,)."
            )

        norm = np.linalg.norm(
            self.sun_direction_eci
        )

        if norm < 1e-12:
            raise ValueError(
                "sun_direction_eci cannot be the zero vector."
            )

        self.sun_direction_eci /= norm

    # ==========================================================
    # Solar Radiation Force
    # ==========================================================

    def compute(self) -> np.ndarray:
        """
        Compute the solar radiation pressure force.

        Returns
        -------
        solar_force_eci : ndarray (3,)
            Solar radiation pressure force expressed
            in the ECI frame [N].
        """

        force_magnitude = (

            self.solar_radiation_pressure

            * self.reflectivity_coefficient

            * self.reference_area

        )

        solar_force_eci = (

            -force_magnitude

            * self.sun_direction_eci

        )

        return solar_force_eci