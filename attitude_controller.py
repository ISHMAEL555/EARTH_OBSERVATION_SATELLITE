"""
controllers/attitude_controller.py

Outer-loop attitude controller (Quaternion + Rate Feedback + Integral)
"""

import numpy as np
from utils.quaternion import quat_error
from config import KP, KD, KI


class AttitudeController:
    """
    Quaternion-based PD+I Attitude Controller
    """

    def __init__(self):
        self.Kp = KP
        self.Kd = KD
        self.Ki = KI
        self.integral_error = np.zeros(3)   # Integral of vector error

    def compute_torque(self, q_target: np.ndarray, q_current: np.ndarray,
                       omega_current: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute commanded body torque.

        Parameters
        ----------
        q_target : Desired attitude quaternion
        q_current : Current attitude quaternion
        omega_current : Current body rates
        dt : time step

        Returns
        -------
        tau_cmd : Commanded torque [Nm]
        """
        # Quaternion error (vector part)
        q_err = quat_error(q_target, q_current)
        err_vec = q_err[1:]                     # 3-element vector

        # Integral action (anti-windup via simple clipping)
        self.integral_error += err_vec * dt
        self.integral_error = np.clip(self.integral_error, -0.5, 0.5)

        # PD + I control law
        tau_cmd = -self.Kp @ err_vec - self.Kd @ omega_current - self.Ki @ self.integral_error

        return tau_cmd

    def reset(self):
        self.integral_error = np.zeros(3)