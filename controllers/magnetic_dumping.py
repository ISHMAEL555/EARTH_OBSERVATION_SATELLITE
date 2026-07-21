"""
controllers/magnetic_dumping.py

Magnetic Momentum Dumping Law
"""

import numpy as np
from config import K_DUMP, H_DUMP_THRESHOLD, M_MAX
from models.actuators.magnetorquer import Magnetorquer


class MagneticDumper:
    """
    B × h type momentum dumping while attitude control is active
    """

    def __init__(self):
        self.mtq = Magnetorquer()
        self.k_dump = K_DUMP

    def compute_dumping(self, h_total: np.ndarray, B_body: np.ndarray) -> np.ndarray:
        """
        Standard cross-product dumping law
        m = -k (B × h) / |B|^2
        """
        B_norm2 = np.dot(B_body, B_body)

        if B_norm2 < 1e-12:
            return np.zeros(3)   # No field → no torque

        # Basic dumping command
        m_cmd = -self.k_dump * np.cross(B_body, h_total) / B_norm2

        # Threshold to avoid unnecessary dumping
        if np.linalg.norm(h_total) < H_DUMP_THRESHOLD:
            m_cmd *= 0.3   # Reduce gain when momentum is low

        return m_cmd

    def get_torque(self, h_total: np.ndarray, B_body: np.ndarray) -> np.ndarray:
        m_cmd = self.compute_dumping(h_total, B_body)
        return self.mtq.compute_torque(m_cmd, B_body)

    def reset(self):
        self.mtq.reset()