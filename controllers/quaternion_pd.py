"""
controllers/quaternion_pd.py

Quaternion Proportional-Derivative Tracking Controller

Implements

    τ = -Kp qe - Kd (ω - ωref)

References
----------
- Markley & Crassidis
- Wie
"""

import numpy as np


class QuaternionPD:
    """
    Quaternion PD attitude tracking controller.
    """

    def __init__(
        self,
        proportional_gain: np.ndarray,
        derivative_gain: np.ndarray,
    ):

        self.Kp = self._validate_gain(
            proportional_gain,
            "proportional_gain",
        )

        self.Kd = self._validate_gain(
            derivative_gain,
            "derivative_gain",
        )

    # ======================================================
    # Public Interface
    # ======================================================

    def compute(
        self,
        current_quaternion: np.ndarray,
        desired_quaternion: np.ndarray,
        body_rates: np.ndarray,
        desired_body_rates: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute commanded body torque.

        Parameters
        ----------
        current_quaternion : ndarray (4,)
        desired_quaternion : ndarray (4,)
        body_rates : ndarray (3,)
        desired_body_rates : ndarray (3,), optional

        Returns
        -------
        ndarray (3,)
            Commanded body torque.
        """

        current_quaternion = self._validate_quaternion(
            current_quaternion
        )

        desired_quaternion = self._validate_quaternion(
            desired_quaternion
        )

        body_rates = self._validate_vector(
            body_rates,
            "body_rates",
        )

        if desired_body_rates is None:

            desired_body_rates = np.zeros(3)

        desired_body_rates = self._validate_vector(
            desired_body_rates,
            "desired_body_rates",
        )

        # ---------------------------------------------
        # Quaternion Error
        # ---------------------------------------------

        q_error = self._quaternion_error(
            current_quaternion,
            desired_quaternion,
        )

        if q_error[0] < 0.0:
            q_error = -q_error

        attitude_error = q_error[1:]

        # ---------------------------------------------
        # Angular Velocity Error
        # ---------------------------------------------

        rate_error = (
            body_rates
            - desired_body_rates
        )

        # ---------------------------------------------
        # PD Tracking Law
        # ---------------------------------------------

        commanded_torque = (

            -self.Kp @ attitude_error

            -self.Kd @ rate_error

        )

        return commanded_torque

    # ======================================================
    # Quaternion Utilities
    # ======================================================

    @staticmethod
    def _quaternion_error(current, desired):

        return QuaternionPD._quat_multiply(

            desired,

            QuaternionPD._quat_conjugate(current),

        )

    @staticmethod
    def _quat_conjugate(q):

        return np.array(
            [
                q[0],
                -q[1],
                -q[2],
                -q[3],
            ],
            dtype=float,
        )

    @staticmethod
    def _quat_multiply(q1, q2):

        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array(
            [
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ],
            dtype=float,
        )

    # ======================================================
    # Validation
    # ======================================================

    @staticmethod
    def _validate_gain(gain, name):

        gain = np.asarray(gain, dtype=float)

        if gain.shape != (3, 3):
            raise ValueError(
                f"{name} must have shape (3,3)."
            )

        if not np.all(np.isfinite(gain)):
            raise ValueError(
                f"{name} contains invalid values."
            )

        return gain

    @staticmethod
    def _validate_vector(vector, name):

        vector = np.asarray(vector, dtype=float)

        if vector.shape != (3,):
            raise ValueError(
                f"{name} must have shape (3,)."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} contains invalid values."
            )

        return vector

    @staticmethod
    def _validate_quaternion(q):

        q = np.asarray(q, dtype=float)

        if q.shape != (4,):
            raise ValueError(
                "Quaternion must have shape (4,)."
            )

        if not np.all(np.isfinite(q)):
            raise ValueError(
                "Quaternion contains invalid values."
            )

        norm = np.linalg.norm(q)

        if norm < 1e-12:
            raise ValueError(
                "Quaternion norm cannot be zero."
            )

        return q / norm