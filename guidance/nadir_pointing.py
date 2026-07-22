"""
guidance/nadir_pointing.py

Nadir Pointing Guidance Law.
"""

from __future__ import annotations

import numpy as np

from guidance.base import Guidance


class NadirPointingGuidance(Guidance):
    """
    Nadir-pointing guidance.

    Currently returns a fixed reference attitude.

    This class will later compute the desired
    LVLH attitude quaternion.
    """

    def compute(
        self,
        position_eci: np.ndarray,
        velocity_eci: np.ndarray,
    ):

        # --------------------------------------------------
        # Temporary placeholder
        # --------------------------------------------------

        q_ref = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        omega_ref = np.zeros(3)

        return q_ref, omega_ref