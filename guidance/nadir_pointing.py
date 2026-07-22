"""
guidance/nadir_pointing.py

LVLH Nadir Pointing Guidance Law
"""

from __future__ import annotations

import numpy as np

from guidance.base import Guidance


class NadirPointingGuidance(Guidance):
    """
    LVLH Nadir Pointing Guidance.

    Body Frame
    ----------
    +X : Velocity direction
    +Y : Orbit normal
    +Z : Earth (Nadir)

    Reference frame:
        ECI -> LVLH
    """

    def compute(
        self,
        position_eci: np.ndarray,
        velocity_eci: np.ndarray,
    ):
        """
        Compute reference quaternion and angular velocity.
        """

        r = np.asarray(position_eci, dtype=float)
        v = np.asarray(velocity_eci, dtype=float)

        # --------------------------------------------------
        # LVLH axes
        # --------------------------------------------------

        z_hat = -r / np.linalg.norm(r)

        h = np.cross(r, v)
        y_hat = -h / np.linalg.norm(h)

        x_hat = np.cross(y_hat, z_hat)
        x_hat /= np.linalg.norm(x_hat)

        # Re-orthogonalize
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)

        # --------------------------------------------------
        # Rotation matrix
        #
        # Columns = body axes expressed in ECI
        # --------------------------------------------------

        C = np.column_stack(
            (
                x_hat,
                y_hat,
                z_hat,
            )
        )

        # --------------------------------------------------
        # DCM -> Quaternion
        # Quaternion format:
        #
        # [w, x, y, z]
        # --------------------------------------------------

        q_ref = self.dcm_to_quaternion(C)

        # --------------------------------------------------
        # Reference body rates
        #
        # Circular orbit approximation
        # --------------------------------------------------

        mu = 3.986004418e14

        orbit_rate = np.sqrt(
            mu / np.linalg.norm(r) ** 3
        )

        omega_ref = np.array(
            [
                0.0,
                orbit_rate,
                0.0,
            ]
        )

        return q_ref, omega_ref

    @staticmethod
    def dcm_to_quaternion(C):
        """
        Convert DCM to quaternion.

        Quaternion ordering:
            [w, x, y, z]
        """

        q = np.zeros(4)

        tr = np.trace(C)

        if tr > 0:

            s = np.sqrt(tr + 1.0) * 2.0

            q[0] = 0.25 * s
            q[1] = (C[2, 1] - C[1, 2]) / s
            q[2] = (C[0, 2] - C[2, 0]) / s
            q[3] = (C[1, 0] - C[0, 1]) / s

        elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:

            s = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0

            q[0] = (C[2, 1] - C[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (C[0, 1] + C[1, 0]) / s
            q[3] = (C[0, 2] + C[2, 0]) / s

        elif C[1, 1] > C[2, 2]:

            s = np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0

            q[0] = (C[0, 2] - C[2, 0]) / s
            q[1] = (C[0, 1] + C[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (C[1, 2] + C[2, 1]) / s

        else:

            s = np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0

            q[0] = (C[1, 0] - C[0, 1]) / s
            q[1] = (C[0, 2] + C[2, 0]) / s
            q[2] = (C[1, 2] + C[2, 1]) / s
            q[3] = 0.25 * s

        q /= np.linalg.norm(q)

        return q