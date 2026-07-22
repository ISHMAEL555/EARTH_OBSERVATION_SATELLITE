"""
models/disturbances/gravity_gradient.py

Gravity Gradient Torque Model

Computes the gravity-gradient disturbance torque acting on a rigid
spacecraft.

Current Model
-------------
- Two-body gravity
- Rigid spacecraft
- Constant inertia matrix

Future Models
-------------
- Higher-order gravity field
- Flexible spacecraft
- Time-varying inertia

References
----------
- Markley & Crassidis, Fundamentals of Spacecraft Attitude
  Determination and Control
- Wie, Space Vehicle Dynamics and Control
"""

import numpy as np

from models.disturbances.disturbance_state import DisturbanceState


class GravityGradient:
    """
    Gravity-gradient disturbance torque model.
    """

    def __init__(
        self,
        gravitational_parameter: float,
    ):

        if gravitational_parameter <= 0.0:
            raise ValueError(
                "gravitational_parameter must be positive."
            )

        self.gravitational_parameter = float(
            gravitational_parameter
        )

    # ==========================================================
    # Gravity Gradient Torque
    # ==========================================================

    def compute(
        self,
        state: DisturbanceState,
    ) -> np.ndarray:
        """
        Compute gravity-gradient disturbance torque.

        Parameters
        ----------
        state : DisturbanceState
            Current spacecraft/environment state.

        Returns
        -------
        ndarray (3,)
            Gravity-gradient torque expressed in the body frame.
        """

        body_to_eci_dcm = np.asarray(
            state.body_to_eci_dcm,
            dtype=float,
        )

        inertia_matrix = np.asarray(
            state.inertia_matrix,
            dtype=float,
        )

        radial_unit_vector_eci = np.asarray(
            state.radial_unit_vector_eci,
            dtype=float,
        )

        orbit_radius = float(
            state.orbit_radius
        )

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        if body_to_eci_dcm.shape != (3, 3):
            raise ValueError(
                "body_to_eci_dcm must have shape (3,3)."
            )

        if inertia_matrix.shape != (3, 3):
            raise ValueError(
                "inertia_matrix must have shape (3,3)."
            )

        if radial_unit_vector_eci.shape != (3,):
            raise ValueError(
                "radial_unit_vector_eci must have shape (3,)."
            )

        if orbit_radius <= 0.0:
            raise ValueError(
                "orbit_radius must be positive."
            )

        radial_norm = np.linalg.norm(
            radial_unit_vector_eci
        )

        if radial_norm < 1e-12:
            raise ValueError(
                "radial_unit_vector_eci cannot be zero."
            )

        radial_unit_vector_eci = (
            radial_unit_vector_eci / radial_norm
        )

        # ------------------------------------------------------
        # Convert radial direction to body frame
        # ------------------------------------------------------

        eci_to_body_dcm = body_to_eci_dcm.T

        radial_unit_vector_body = (
            eci_to_body_dcm
            @ radial_unit_vector_eci
        )

        # ------------------------------------------------------
        # Gravity-gradient torque
        # ------------------------------------------------------

        gravity_gradient_torque = (

            3.0
            * self.gravitational_parameter
            / orbit_radius**3

            * np.cross(

                radial_unit_vector_body,

                inertia_matrix
                @ radial_unit_vector_body,

            )

        )

        return gravity_gradient_torque