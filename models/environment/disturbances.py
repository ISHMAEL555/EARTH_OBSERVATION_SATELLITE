"""
models/environment/disturbances.py

Disturbance Torques
"""

import numpy as np
from models.environment.gravity_gradient import gravity_gradient_torque


class Disturbances:
    def __init__(self, J):
        self.J = J

    def compute_total_disturbance(self, q: np.ndarray, omega: np.ndarray,
                                  pos_eci: np.ndarray, t: float) -> np.ndarray:
        """
        Total disturbance torque
        """
        R = np.linalg.norm(pos_eci)

        # Mandatory: Gravity Gradient
        T_gg = gravity_gradient_torque(q, self.J, R)

        # Optional: Aero + SRP (simplified)
        T_aero_srp = np.random.normal(0, 1e-6, 3)   # Small random for now

        return T_gg + T_aero_srp