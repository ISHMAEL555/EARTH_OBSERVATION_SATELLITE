"""
lqr.py

Linear Quadratic Regulator (LQR) Attitude Controller

Author
------
Kowluri Ishmael

Description
-----------
Continuous-time LQR controller for spacecraft attitude control.

State Vector
------------
x = [attitude_error_x,
     attitude_error_y,
     attitude_error_z,
     omega_x,
     omega_y,
     omega_z]

Control Output
--------------
Desired body control torque [N·m]
"""

import numpy as np
from scipy.linalg import solve_continuous_are

from config import CONTROLLERS, SPACECRAFT


class LQR:
    """
    Continuous-Time Linear Quadratic Regulator
    """

    def __init__(self):

        cfg = CONTROLLERS["lqr"]

        self.Q = cfg["Q"]
        self.R = cfg["R"]

        self.inertia = SPACECRAFT["inertia"]

        self.A = self._build_A()
        self.B = self._build_B()

        self.K = self._compute_gain()

        self.state = np.zeros(6)

        self.commanded_torque = np.zeros(3)

        self.attitude_error = np.zeros(3)
        self.rate_error = np.zeros(3)

    # ======================================================
    # Public Interface
    # ======================================================

    def update(
        self,
        current_quaternion,
        desired_quaternion,
        body_rates,
    ):
        """
        Compute desired body control torque.

        Parameters
        ----------
        current_quaternion : ndarray (4,)
        desired_quaternion : ndarray (4,)
        body_rates : ndarray (3,)

        Returns
        -------
        ndarray (3,)
            Desired body control torque.
        """

        # ------------------------------------------
        # Convert to floating-point arrays
        # ------------------------------------------

        current_quaternion = np.asarray(
            current_quaternion,
            dtype=float
        )

        desired_quaternion = np.asarray(
            desired_quaternion,
            dtype=float
        )

        body_rates = np.asarray(
            body_rates,
            dtype=float
        )

        # ------------------------------------------
        # Validate
        # ------------------------------------------

        self._validate_quaternion(current_quaternion)
        self._validate_quaternion(desired_quaternion)
        self._validate_vector(body_rates)

        # ------------------------------------------
        # Normalize
        # ------------------------------------------

        current_quaternion /= np.linalg.norm(current_quaternion)
        desired_quaternion /= np.linalg.norm(desired_quaternion)

        # ------------------------------------------
        # Quaternion Error
        # ------------------------------------------

        q_error = self._quaternion_error(
            current_quaternion,
            desired_quaternion
        )

        # Shortest Rotation

        if q_error[0] < 0.0:
            q_error = -q_error

        # ------------------------------------------
        # Small-angle approximation
        # ------------------------------------------

        self.attitude_error = 2.0 * q_error[1:]

        self.rate_error = body_rates.copy()

        # ------------------------------------------
        # State Vector
        # ------------------------------------------

        self.state = np.hstack((
            self.attitude_error,
            self.rate_error
        ))

        # ------------------------------------------
        # LQR Control Law
        # ------------------------------------------

        self.commanded_torque = -self.K @ self.state

        return self.commanded_torque.copy()

    # ======================================================
    # Linearized Dynamics
    # ======================================================

    def _build_A(self):

        A = np.zeros((6, 6))

        A[0:3, 3:6] = 0.5 * np.eye(3)

        return A

    def _build_B(self):

        B = np.zeros((6, 3))

        B[3:6, :] = np.linalg.inv(self.inertia)

        return B

    def _compute_gain(self):

        P = solve_continuous_are(
            self.A,
            self.B,
            self.Q,
            self.R,
        )

        return np.linalg.inv(self.R) @ self.B.T @ P

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.state[:] = 0.0

        self.commanded_torque[:] = 0.0

        self.attitude_error[:] = 0.0

        self.rate_error[:] = 0.0

    # ======================================================
    # Quaternion Utilities
    # ======================================================

    @staticmethod
    def _quaternion_error(current, desired):

        return LQR._quat_multiply(
            desired,
            LQR._quat_conjugate(current)
        )

    @staticmethod
    def _quat_conjugate(q):

        return np.array([
            q[0],
            -q[1],
            -q[2],
            -q[3],
        ])

    @staticmethod
    def _quat_multiply(q1, q2):

        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    # ======================================================
    # Validation
    # ======================================================

    @staticmethod
    def _validate_vector(v):

        v = np.asarray(v, dtype=float)

        if v.shape != (3,):
            raise ValueError(
                "Vector must have shape (3,)"
            )

        if not np.all(np.isfinite(v)):
            raise ValueError(
                "Vector contains invalid values."
            )

    @staticmethod
    def _validate_quaternion(q):

        q = np.asarray(q, dtype=float)

        if q.shape != (4,):
            raise ValueError(
                "Quaternion must have shape (4,)"
            )

        if not np.all(np.isfinite(q)):
            raise ValueError(
                "Quaternion contains invalid values."
            )

        if np.linalg.norm(q) < 1e-10:
            raise ValueError(
                "Quaternion norm cannot be zero."
            )