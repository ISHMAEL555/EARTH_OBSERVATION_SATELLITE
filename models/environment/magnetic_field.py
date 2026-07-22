"""
models/environment/magnetic_field.py

Earth Magnetic Field Model

Computes the Earth's magnetic field using a centered dipole
approximation.

The model is completely independent of:

- Spacecraft
- Sensors
- Actuators
- Controllers
- Simulation scenarios

Input
-----
Spacecraft position expressed in the ECI frame.

Output
------
Earth magnetic field expressed in the ECI frame.

Current Model
-------------
- Centered Dipole Approximation

Future Models
-------------
- IGRF13
- WMM
"""

import numpy as np


class MagneticField:
    """
    Earth's Magnetic Field Model.

    Parameters
    ----------
    magnetic_dipole_moment : ndarray (3,)
        Earth's magnetic dipole moment expressed in the
        Earth-Centered Inertial (ECI) frame.
    """

    def __init__(
        self,
        magnetic_dipole_moment: np.ndarray,
    ):

        magnetic_dipole_moment = np.asarray(
            magnetic_dipole_moment,
            dtype=float,
        )

        if magnetic_dipole_moment.shape != (3,):
            raise ValueError(
                "magnetic_dipole_moment must have shape (3,)."
            )

        self.magnetic_dipole_moment = magnetic_dipole_moment

    # ==========================================================
    # Magnetic Field
    # ==========================================================

    def compute(
        self,
        spacecraft_position_eci: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Earth's magnetic field using the centered
        dipole approximation.

        Parameters
        ----------
        spacecraft_position_eci : ndarray (3,)
            Spacecraft position expressed in the ECI frame [m].

        Returns
        -------
        magnetic_field_eci : ndarray (3,)
            Earth's magnetic field expressed in the
            ECI frame [Tesla].
        """

        spacecraft_position_eci = np.asarray(
            spacecraft_position_eci,
            dtype=float,
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
                "Spacecraft position magnitude must be greater than zero."
            )

        # ------------------------------------------------------
        # Unit radial vector
        # ------------------------------------------------------

        radial_unit_vector = (
            spacecraft_position_eci /
            orbit_radius
        )

        # ------------------------------------------------------
        # Centered Dipole Magnetic Field
        # ------------------------------------------------------

        magnetic_field_eci = (
            self.magnetic_dipole_moment /
            orbit_radius**3
        ) * (
            3.0
            * np.dot(
                self.magnetic_dipole_moment,
                radial_unit_vector,
            )
            * radial_unit_vector
            - self.magnetic_dipole_moment
        )

        return magnetic_field_eci