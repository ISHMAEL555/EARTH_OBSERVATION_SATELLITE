"""
quaternion_pd.py

Quaternion Proportional-Derivative (PD) Attitude Controller

Author
------
Kowluri Ishmael

Description
-----------
Computes the commanded body torque required to drive the spacecraft
from the current attitude to the desired attitude using quaternion
feedback and body-rate damping.

This controller outputs the desired body torque only.
Actuator allocation is performed separately.
"""

import numpy as np

from config import CONTROLLERS


class QuaternionPD:
    """
    Quaternion Proportional-Derivative Controller.
    """

    def __init__(self):

        cfg = CONTROLLERS["pd"]

        self.Kp = cfg["Kp"]
        self.Kd = cfg["Kd"]

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
        Compute commanded body torque.

        Parameters
        ----------
        current_quaternion : ndarray, shape (4,)
            Current spacecraft attitude quaternion.

        desired_quaternion : ndarray, shape (4,)
            Desired attitude quaternion.

        body_rates : ndarray, shape (3,)
            Current body angular velocity [rad/s].

        Returns
        -------
        ndarray
            Desired body torque [N·m].
        """

        self._validate_quaternion(current_quaternion)
        self._validate_quaternion(desired_quaternion)
        self._validate_vector(body_rates)

        q_error = self._quaternion_error(
            current_quaternion,
            desired_quaternion,
        )

        # Ensure shortest rotation
        if q_error[0] < 0.0:
            q_error = -q_error

        self.attitude_error = q_error[1:]
        self.rate_error = body_rates

        self.commanded_torque = (
            -self.Kp @ self.attitude_error
            -self.Kd @ self.rate_error
        )

        return self.commanded_torque.copy()

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.commanded_torque[:] = 0.0
        self.attitude_error[:] = 0.0
        self.rate_error[:] = 0.0

    # ======================================================
    # Quaternion Utilities
    # ======================================================

    @staticmethod
    def _quaternion_error(current, desired):
        """
        Error quaternion.

        q_error = q_desired * conj(q_current)
        """

        return QuaternionPD._quat_multiply(
            desired,
            QuaternionPD._quat_conjugate(current),
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
    def _validate_vector(vector):

        vector = np.asarray(vector)

        if vector.shape != (3,):
            raise ValueError(
                "Input vector must have shape (3,)"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "Input vector contains invalid values."
            )

    @staticmethod
    def _validate_quaternion(q):

        q = np.asarray(q)

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
                "Quaternion magnitude cannot be zero."
            )

        q /= np.linalg.norm(q)