"""
models/actuators/rw_pyramid.py

Pyramidal Reaction Wheel Cluster (4 wheels, 45° skew)
Strictly follows assignment: momentum capacity 1.4 Nms, peak torque 0.475 Nm per wheel.
"""

import numpy as np


class RWPyramid:
    """
    4-wheel pyramidal reaction wheel cluster.
    Skew angle = 45° (aligned to pitch-yaw plane).
    """

    def __init__(self, h_max: float = 1.4, tau_max: float = 0.475):
        self.h_max = h_max          # Nms per wheel
        self.tau_max = tau_max      # Nm per wheel

        # Wheel spin axes in body frame (unit vectors)
        # Pyramid configuration, 45° skew
        s = np.sin(np.deg2rad(45.0))
        c = np.cos(np.deg2rad(45.0))

        # Wheel directions (standard pyramid mounting)
        self.wheel_axes = np.array([
            [ c,  0,  s],   # Wheel 1
            [ 0,  c,  s],   # Wheel 2
            [-c,  0,  s],   # Wheel 3
            [ 0, -c,  s]    # Wheel 4
        ]).T  # Shape: (3, 4)

        # Allocation matrix: torque_body = A @ tau_wheels
        self.A = self.wheel_axes.copy()           # (3 x 4)
        self.A_pinv = np.linalg.pinv(self.A)      # Moore-Penrose pseudo-inverse (4 x 3)

        # State
        self.omega_rw = np.zeros(4)               # Wheel speeds (rad/s) - not needed if using momentum directly
        self.h_rw = np.zeros(4)                   # Angular momentum of each wheel [Nms]

    def allocate_torque(self, tau_cmd: np.ndarray) -> np.ndarray:
        """
        Allocate commanded body torque to individual wheel torques.

        Parameters
        ----------
        tau_cmd : np.ndarray (3,)   Desired body torque [Nm]

        Returns
        -------
        tau_wheels : np.ndarray (4,)   Torque command for each wheel [Nm]
        """
        tau_wheels = self.A_pinv @ tau_cmd

        # Saturate wheel torques
        tau_wheels = np.clip(tau_wheels, -self.tau_max, self.tau_max)

        return tau_wheels

    def update_momentum(self, tau_wheels: np.ndarray, dt: float):
        """
        Integrate wheel momentum.
        h_dot = tau_wheels
        """
        self.h_rw += tau_wheels * dt

        # Saturate momentum
        self.h_rw = np.clip(self.h_rw, -self.h_max, self.h_max)

    def get_total_momentum(self) -> np.ndarray:
        """Return total RW momentum in body frame [Nms]"""
        return self.wheel_axes @ self.h_rw

    def get_momentum_capacity(self) -> float:
        """Approximate total scalar momentum capacity"""
        return 4 * self.h_max

    def reset(self):
        """Reset wheels to zero momentum"""
        self.h_rw = np.zeros(4)
        self.omega_rw = np.zeros(4)