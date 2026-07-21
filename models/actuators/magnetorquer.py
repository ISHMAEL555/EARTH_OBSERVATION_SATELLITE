"""
models/actuators/magnetorquer.py

3-Axis Magnetorquer Model for Momentum Dumping
Strictly follows assignment: 18.28 Am² per axis
"""

import numpy as np


class Magnetorquer:
    """
    3-Axis Orthogonal Magnetorquer Assembly
    """

    def __init__(self, m_max: float = 18.28):
        self.m_max = m_max                    # Maximum dipole moment per axis [Am²]
        self.dipole_cmd = np.zeros(3)         # Current commanded dipole [Am²]

    def compute_torque(self, m_cmd: np.ndarray, B_body: np.ndarray) -> np.ndarray:
        """
        Compute magnetic torque: T = m × B

        Parameters
        ----------
        m_cmd : np.ndarray (3,)
            Commanded magnetic dipole moment [Am²]
        B_body : np.ndarray (3,)
            Magnetic field vector in body frame [Tesla]

        Returns
        -------
        torque : np.ndarray (3,)
            Produced torque [Nm]
        """
        # Saturate dipole command
        self.dipole_cmd = np.clip(m_cmd, -self.m_max, self.m_max)

        # T = m × B
        torque = np.cross(self.dipole_cmd, B_body)
        return torque

    def reset(self):
        """Reset commands"""
        self.dipole_cmd = np.zeros(3)