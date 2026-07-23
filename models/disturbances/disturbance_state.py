"""
Common disturbance input state.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DisturbanceState:
    """
    Common input state shared by all disturbance models.

    Notes
    -----
    All vectors are expressed in the Earth-Centered Inertial (ECI)
    frame unless otherwise specified.

    Vector Shapes
    -------------
    position_eci            : (3,)
    velocity_eci            : (3,)
    radial_unit_vector_eci  : (3,)
    body_to_eci_dcm         : (3,3)
    inertia_matrix          : (3,3)
    magnetic_field_eci      : (3,)
    magnetic_field_body     : (3,)
    solar_vector_eci        : (3,)
    """

    # ======================================================
    # Time
    # ======================================================

    time: float

    # ======================================================
    # Orbit
    # ======================================================

    position_eci: np.ndarray
    velocity_eci: np.ndarray

    orbit_radius: float
    radial_unit_vector_eci: np.ndarray

    # ======================================================
    # Spacecraft
    # ======================================================

    body_to_eci_dcm: np.ndarray
    inertia_matrix: np.ndarray

    # ======================================================
    # Environment
    # ======================================================

    magnetic_field_eci: np.ndarray
    magnetic_field_body: np.ndarray

    atmospheric_density: float

    solar_vector_eci: np.ndarray

    # ======================================================
    # Validation
    # ======================================================

    def __post_init__(self):
        """
        Validate state dimensions.
        """

        vector_fields = {
            "position_eci": self.position_eci,
            "velocity_eci": self.velocity_eci,
            "radial_unit_vector_eci": self.radial_unit_vector_eci,
            "magnetic_field_eci": self.magnetic_field_eci,
            "magnetic_field_body": self.magnetic_field_body,
            "solar_vector_eci": self.solar_vector_eci,
        }

        matrix_fields = {
            "body_to_eci_dcm": self.body_to_eci_dcm,
            "inertia_matrix": self.inertia_matrix,
        }

        for name, value in vector_fields.items():

            array = np.asarray(
                value,
                dtype=float,
            )

            if array.shape != (3,):
                raise ValueError(
                    f"{name} must have shape (3,)."
                )

            setattr(
                self,
                name,
                array,
            )

        for name, value in matrix_fields.items():

            array = np.asarray(
                value,
                dtype=float,
            )

            if array.shape != (3, 3):
                raise ValueError(
                    f"{name} must have shape (3,3)."
                )

            setattr(
                self,
                name,
                array,
            )

        self.time = float(self.time)
        self.orbit_radius = float(self.orbit_radius)
        self.atmospheric_density = float(
            self.atmospheric_density
        )