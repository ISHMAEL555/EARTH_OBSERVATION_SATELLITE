"""
models/disturbances/solar_radiation_pressure.py

Solar Radiation Pressure Force Model

Computes the solar radiation pressure force acting on the spacecraft.

Current Model
-------------
- Constant solar radiation pressure at 1 AU
- Constant reflectivity coefficient
- Constant illuminated area
- Sun direction provided by DisturbanceState
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

from models.disturbances.disturbance_state import DisturbanceState


class SolarRadiationPressure:
    """
    Solar Radiation Pressure (SRP) force model.
    """

    def __init__(
        self,
        solar_radiation_pressure: float,
        reflectivity_coefficient: float,
        reference_area: float,
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

    # ==========================================================
    # Solar Radiation Pressure Force
    # ==========================================================

    def compute(
        self,
        state: DisturbanceState,
    ) -> np.ndarray:
        """
        Compute the solar radiation pressure force.

        Parameters
        ----------
        state : DisturbanceState
            Current spacecraft/environment state.

        Returns
        -------
        ndarray (3,)
            Solar radiation pressure force expressed
            in the ECI frame [N].
        """

        sun_direction_eci = np.asarray(
            state.solar_vector_eci,
            dtype=float,
        )

        if sun_direction_eci.shape != (3,):
            raise ValueError(
                "solar_vector_eci must have shape (3,)."
            )

        norm = np.linalg.norm(sun_direction_eci)

        if norm < 1e-12:
            return np.zeros(3)

        sun_direction_eci /= norm

        force_magnitude = (

            self.solar_radiation_pressure

            * self.reflectivity_coefficient

            * self.reference_area

        )

        solar_force_eci = (

            -force_magnitude

            * sun_direction_eci

        )

        return solar_force_eci