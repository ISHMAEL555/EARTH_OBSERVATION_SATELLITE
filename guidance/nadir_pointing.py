"""
guidance/nadir_pointing.py

LVLH Nadir Pointing Guidance

Body Frame
----------
+X : Along-track (Velocity)
+Y : Cross-track
+Z : Nadir (Toward Earth)

Quaternion Convention
---------------------
Scalar-first

Quaternion represents

    Body -> ECI
"""

from __future__ import annotations

import numpy as np

from guidance.base import Guidance
from models.dynamics.quaternion import dcm_to_quaternion


class NadirPointingGuidance(Guidance):
    """
    LVLH Nadir Pointing Guidance.
    """

    def compute(
        self,
        position_eci: np.ndarray,
        velocity_eci: np.ndarray,
    ):

        r = np.asarray(position_eci, dtype=float)
        v = np.asarray(velocity_eci, dtype=float)

        # --------------------------------------------------
        # Unit vectors
        # --------------------------------------------------

        r_hat = r / np.linalg.norm(r)
        v_hat = v / np.linalg.norm(v)

        # --------------------------------------------------
        # LVLH Frame
        # --------------------------------------------------

        # +Z : Toward Earth
        z_hat = -r_hat

        # +X : Along-track
        x_hat = v_hat

        # +Y : Complete right-handed frame
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)

        # Re-orthogonalize X
        x_hat = np.cross(y_hat, z_hat)
        x_hat /= np.linalg.norm(x_hat)

        # --------------------------------------------------
        # Body -> ECI DCM
        # --------------------------------------------------

        C_bi = np.column_stack((x_hat,y_hat,z_hat,))

        # Trial 1
        q_ref = dcm_to_quaternion(C_bi.T)

        # --------------------------------------------------
        # Reference Body Rates
        #
        # TEMPORARY:
        #
        # The controller regulation has already been verified
        # using zero reference body rates.
        #
        # Until the LVLH angular velocity is derived
        # consistently from the reference frame,
        # command zero body rates.
        # --------------------------------------------------

        omega_ref = np.zeros(3)

        return q_ref, omega_ref