"""
models/disturbances/atmospheric_drag.py

Atmospheric Drag Disturbance Model

Computes the aerodynamic disturbance torque acting on the spacecraft
using a simple exponential atmosphere model.

Current Model
-------------
- Exponential atmosphere
- Constant drag coefficient
- Constant reference area
- Constant center of pressure
- Neglects Earth rotation

References
----------
- Wertz, Spacecraft Attitude Determination and Control
- Curtis, Orbital Mechanics for Engineering Students
"""

import numpy as np


class AtmosphericDrag:
    """
    Atmospheric drag disturbance model.
    """

    def __init__(
        self,
        earth_radius: float,
        atmosphere_surface_density: float,
        atmosphere_scale_height: float,
        drag_coefficient: float,
        reference_area: float,
        center_of_pressure: np.ndarray,
    ):
        """
        Initialize atmospheric drag model.

        Parameters
        ----------
        earth_radius : float
            Earth's mean radius [m].

        atmosphere_surface_density : float
            Atmospheric density at sea level [kg/m³].

        atmosphere_scale_height : float
            Atmospheric scale height [m].

        drag_coefficient : float
            Spacecraft drag coefficient.

        reference_area : float
            Effective aerodynamic reference area [m²].

        center_of_pressure : ndarray (3,)
            Center of pressure expressed in the spacecraft body frame [m].
        """

        self.earth_radius = earth_radius
        self.atmosphere_surface_density = atmosphere_surface_density
        self.atmosphere_scale_height = atmosphere_scale_height

        self.drag_coefficient = drag_coefficient
        self.reference_area = reference_area
        self.center_of_pressure = np.asarray(
            center_of_pressure,
            dtype=float,
        )

    # ==========================================================
    # Atmospheric Density
    # ==========================================================

    def _compute_density(
        self,
        altitude: float,
    ) -> float:
        """
        Compute atmospheric density using an exponential model.

        Parameters
        ----------
        altitude : float
            Spacecraft altitude above Earth's surface [m].

        Returns
        -------
        density : float
            Atmospheric density [kg/m³].
        """

        if altitude < 0.0:
            altitude = 0.0

        density = (
            self.atmosphere_surface_density
            * np.exp(
                -altitude /
                self.atmosphere_scale_height
            )
        )

        return density

    # ==========================================================
    # Drag Force
    # ==========================================================

    def _compute_drag_force(
        self,
        density: float,
        spacecraft_velocity_eci: np.ndarray,
    ) -> np.ndarray:
        """
        Compute aerodynamic drag force in the ECI frame.

        Parameters
        ----------
        density : float
            Atmospheric density [kg/m³].

        spacecraft_velocity_eci : ndarray (3,)
            Spacecraft velocity in ECI frame [m/s].

        Returns
        -------
        drag_force_eci : ndarray (3,)
            Aerodynamic drag force expressed in ECI frame [N].
        """

        velocity_magnitude = np.linalg.norm(
            spacecraft_velocity_eci
        )

        if velocity_magnitude < 1e-12:
            return np.zeros(3)

        velocity_unit_vector = (
            spacecraft_velocity_eci /
            velocity_magnitude
        )

        dynamic_pressure = (
            0.5
            * density
            * velocity_magnitude**2
        )

        drag_force_magnitude = (
            dynamic_pressure
            * self.drag_coefficient
            * self.reference_area
        )

        drag_force_eci = (
            -drag_force_magnitude
            * velocity_unit_vector
        )

        return drag_force_eci

    # ==========================================================
    # Atmospheric Drag Torque
    # ==========================================================

    def compute(
        self,
        body_to_eci_dcm: np.ndarray,
        spacecraft_position_eci: np.ndarray,
        spacecraft_velocity_eci: np.ndarray,
    ) -> np.ndarray:
        """
        Compute atmospheric drag disturbance torque.

        Parameters
        ----------
        body_to_eci_dcm : ndarray (3,3)
            Direction Cosine Matrix transforming vectors
            from body frame to ECI frame.

        spacecraft_position_eci : ndarray (3,)
            Spacecraft position in ECI frame [m].

        spacecraft_velocity_eci : ndarray (3,)
            Spacecraft velocity in ECI frame [m/s].

        Returns
        -------
        atmospheric_drag_torque : ndarray (3,)
            Aerodynamic disturbance torque expressed
            in the spacecraft body frame [N·m].
        """

        body_to_eci_dcm = np.asarray(
            body_to_eci_dcm,
            dtype=float,
        )

        spacecraft_position_eci = np.asarray(
            spacecraft_position_eci,
            dtype=float,
        )

        spacecraft_velocity_eci = np.asarray(
            spacecraft_velocity_eci,
            dtype=float,
        )

        if body_to_eci_dcm.shape != (3, 3):
            raise ValueError(
                "body_to_eci_dcm must be a 3×3 matrix."
            )

        if spacecraft_position_eci.shape != (3,):
            raise ValueError(
                "spacecraft_position_eci must have shape (3,)."
            )

        if spacecraft_velocity_eci.shape != (3,):
            raise ValueError(
                "spacecraft_velocity_eci must have shape (3,)."
            )

        orbit_radius = np.linalg.norm(
            spacecraft_position_eci
        )

        altitude = (
            orbit_radius -
            self.earth_radius
        )

        density = self._compute_density(
            altitude
        )

        drag_force_eci = self._compute_drag_force(
            density,
            spacecraft_velocity_eci,
        )

        eci_to_body_dcm = body_to_eci_dcm.T

        drag_force_body = (
            eci_to_body_dcm
            @ drag_force_eci
        )

        atmospheric_drag_torque = np.cross(
            self.center_of_pressure,
            drag_force_body,
        )

        return atmospheric_drag_torque