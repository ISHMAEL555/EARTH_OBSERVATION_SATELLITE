"""
models/environment/magnetic_field.py

Earth Magnetic Field Model

Computes the Earth's magnetic field at the spacecraft location and
expresses it in the spacecraft body frame.

The magnetic field model is independent of the spacecraft dynamics,
actuators, and controllers.

Current Model
-------------
- Centered dipole approximation

Future Models
-------------
- IGRF13
- WMM
"""

import numpy as np

from config import EARTH_MAGNETIC_DIPOLE_MOMENT


class MagneticField:
    """
    Earth's Magnetic Field Model.

    Notes
    -----
    Computes the Earth's magnetic field using a centered dipole model.
    The returned magnetic field is expressed in the spacecraft body frame.
    """

    def __init__(self):
        """Initialize magnetic field model."""

        self.earth_magnetic_dipole_moment = (
            EARTH_MAGNETIC_DIPOLE_MOMENT
        )

    # ==========================================================
    # Magnetic Field
    # ==========================================================

    def compute(
        self,
        body_to_eci_dcm: np.ndarray,
        spacecraft_position_eci: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Earth's magnetic field.

        Parameters
        ----------
        body_to_eci_dcm : ndarray (3,3)
            Direction Cosine Matrix transforming vectors from the
            body frame to the ECI frame.

        spacecraft_position_eci : ndarray (3,)
            Spacecraft position expressed in the ECI frame [m].

        Returns
        -------
        magnetic_field_body : ndarray (3,)
            Earth's magnetic field expressed in the spacecraft
            body frame [Tesla].
        """

        body_to_eci_dcm = np.asarray(
            body_to_eci_dcm,
            dtype=float,
        )

        spacecraft_position_eci = np.asarray(
            spacecraft_position_eci,
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

        orbit_radius = np.linalg.norm(
            spacecraft_position_eci
        )

        if orbit_radius <= 0.0:
            raise ValueError(
                "Spacecraft position cannot be zero."
            )

        # ------------------------------------------------------
        # Unit radial direction
        # ------------------------------------------------------

        radial_unit_vector_eci = (
            spacecraft_position_eci / orbit_radius
        )

        # ------------------------------------------------------
        # Centered Dipole Magnetic Field (ECI)
        # ------------------------------------------------------

        magnetic_field_eci = (
            self.earth_magnetic_dipole_moment
            / orbit_radius**3
        ) * (
            3.0
            * np.dot(
                self.earth_magnetic_dipole_moment,
                radial_unit_vector_eci,
            )
            * radial_unit_vector_eci
            - self.earth_magnetic_dipole_moment
        )

        # ------------------------------------------------------
        # Transform to Body Frame
        # ------------------------------------------------------

        eci_to_body_dcm = body_to_eci_dcm.T

        magnetic_field_body = (
            eci_to_body_dcm @ magnetic_field_eci
        )

        return magnetic_field_body