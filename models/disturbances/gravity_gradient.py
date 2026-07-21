"""
models/environment/gravity_gradient.py

Gravity Gradient Torque Model

Computes the gravity-gradient disturbance torque acting on a rigid
spacecraft.

The model is independent of the orbit propagator and only requires
the current radial direction expressed in the inertial frame.

References
----------
- Markley & Crassidis, Fundamentals of Spacecraft Attitude Determination
  and Control
- Wie, Space Vehicle Dynamics and Control
"""

import numpy as np

from config import MU


class GravityGradient:
    """
    Gravity Gradient Torque Model.

    Computes the gravity-gradient disturbance torque

        τ_gg = (3μ/R³) (r̂ × J r̂)

    where

        μ   : Earth's gravitational parameter
        R   : Orbit radius
        r̂  : Radial unit vector expressed in the body frame
        J   : Spacecraft inertia matrix
    """

    def __init__(self):
        """Initialize the gravity-gradient model."""

        self.mu = MU

    # ==========================================================
    # Gravity Gradient Torque
    # ==========================================================

    def compute(
        self,
        body_to_eci_dcm: np.ndarray,
        inertia_matrix: np.ndarray,
        radial_unit_vector_eci: np.ndarray,
        orbit_radius: float,
    ) -> np.ndarray:
        """
        Compute the gravity-gradient disturbance torque.

        Parameters
        ----------
        body_to_eci_dcm : ndarray (3,3)
            Direction Cosine Matrix transforming vectors from the
            body frame to the ECI frame.

        inertia_matrix : ndarray (3,3)
            Spacecraft inertia matrix [kg·m²].

        radial_unit_vector_eci : ndarray (3,)
            Unit vector from the Earth's center to the spacecraft
            expressed in the ECI frame.

        orbit_radius : float
            Distance from the Earth's center to the spacecraft [m].

        Returns
        -------
        gravity_gradient_torque : ndarray (3,)
            Gravity-gradient disturbance torque expressed in the
            spacecraft body frame [N·m].
        """

        body_to_eci_dcm = np.asarray(body_to_eci_dcm, dtype=float)
        inertia_matrix = np.asarray(inertia_matrix, dtype=float)
        radial_unit_vector_eci = np.asarray(
            radial_unit_vector_eci,
            dtype=float,
        )

        if body_to_eci_dcm.shape != (3, 3):
            raise ValueError(
                "body_to_eci_dcm must be a 3×3 matrix."
            )

        if inertia_matrix.shape != (3, 3):
            raise ValueError(
                "inertia_matrix must be a 3×3 matrix."
            )

        if radial_unit_vector_eci.shape != (3,):
            raise ValueError(
                "radial_unit_vector_eci must have shape (3,)."
            )

        if orbit_radius <= 0.0:
            raise ValueError(
                "orbit_radius must be positive."
            )

        # Ensure unit-length radial vector
        radial_unit_vector_eci /= np.linalg.norm(
            radial_unit_vector_eci
        )

        # Transform radial direction into the body frame
        eci_to_body_dcm = body_to_eci_dcm.T

        radial_unit_vector_body = (
            eci_to_body_dcm @ radial_unit_vector_eci
        )

        # Compute gravity-gradient disturbance torque
        gravity_gradient_torque = (
            (3.0 * self.mu / orbit_radius**3)
            * np.cross(
                radial_unit_vector_body,
                inertia_matrix @ radial_unit_vector_body,
            )
        )

        return gravity_gradient_torque