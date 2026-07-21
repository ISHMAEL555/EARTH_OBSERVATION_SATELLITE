"""
models/disturbances/solar_radiation_pressure.py

Solar Radiation Pressure (SRP) Disturbance Model

Computes the disturbance torque acting on the spacecraft due to
solar radiation pressure using a simplified constant-pressure model.

Current Model
-------------
- Constant solar radiation pressure at 1 AU
- Constant reflectivity coefficient
- Constant illuminated area
- Constant center of pressure
- Fixed Sun direction in ECI
- No eclipse modelling
- No articulated solar arrays

References
----------
- Wertz, Spacecraft Attitude Determination and Control
- Fortescue, Spacecraft Systems Engineering
"""

import numpy as np


class SolarRadiationPressure:
    """
    Solar Radiation Pressure disturbance model.
    """

    def __init__(
        self,
        solar_radiation_pressure: float,
        reflectivity_coefficient: float,
        reference_area: float,
        center_of_pressure: np.ndarray,
        sun_direction_eci: np.ndarray,
    ):
        """
        Initialize the SRP disturbance model.

        Parameters
        ----------
        solar_radiation_pressure : float
            Solar radiation pressure at 1 AU [N/m²].

        reflectivity_coefficient : float
            Surface reflectivity coefficient.

        reference_area : float
            Effective illuminated area [m²].

        center_of_pressure : ndarray (3,)
            Center of pressure expressed in the spacecraft body frame [m].

        sun_direction_eci : ndarray (3,)
            Unit vector pointing from Earth toward the Sun
            expressed in the ECI frame.
        """

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

        self.solar_radiation_pressure = solar_radiation_pressure
        self.reflectivity_coefficient = reflectivity_coefficient
        self.reference_area = reference_area

        self.center_of_pressure = np.asarray(
            center_of_pressure,
            dtype=float,
        )

        self.sun_direction_eci = np.asarray(
            sun_direction_eci,
            dtype=float,
        )

        if self.center_of_pressure.shape != (3,):
            raise ValueError(
                "center_of_pressure must have shape (3,)."
            )

        if self.sun_direction_eci.shape != (3,):
            raise ValueError(
                "sun_direction_eci must have shape (3,)."
            )

        norm = np.linalg.norm(self.sun_direction_eci)

        if norm < 1e-12:
            raise ValueError(
                "sun_direction_eci cannot be the zero vector."
            )

        self.sun_direction_eci /= norm

    # ==========================================================
    # Solar Radiation Force
    # ==========================================================

    def _compute_solar_force(self) -> np.ndarray:
        """
        Compute the solar radiation force in the ECI frame.

        Returns
        -------
        solar_force_eci : ndarray (3,)
            Solar radiation force expressed in the ECI frame [N].
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

    # ==========================================================
    # Solar Radiation Pressure Torque
    # ==========================================================

    def compute(
        self,
        body_to_eci_dcm: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the solar radiation pressure disturbance torque.

        Parameters
        ----------
        body_to_eci_dcm : ndarray (3,3)
            Direction Cosine Matrix transforming vectors
            from the body frame to the ECI frame.

        Returns
        -------
        solar_radiation_pressure_torque : ndarray (3,)
            Solar radiation pressure disturbance torque
            expressed in the spacecraft body frame [N·m].
        """

        body_to_eci_dcm = np.asarray(
            body_to_eci_dcm,
            dtype=float,
        )

        if body_to_eci_dcm.shape != (3, 3):
            raise ValueError(
                "body_to_eci_dcm must have shape (3,3)."
            )

        solar_force_eci = self._compute_solar_force()

        eci_to_body_dcm = body_to_eci_dcm.T

        solar_force_body = (
            eci_to_body_dcm
            @ solar_force_eci
        )

        solar_radiation_pressure_torque = np.cross(
            self.center_of_pressure,
            solar_force_body,
        )

        return solar_radiation_pressure_torque