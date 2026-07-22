"""
controllers/lqr.py

Continuous-Time Linear Quadratic Regulator (LQR)
for Spacecraft Attitude Control.

Author
------
Kowluri Ishmael

Description
-----------
Continuous-time LQR controller using quaternion attitude
error and body-rate feedback.

State Vector
------------
x = [
    attitude_error_x,
    attitude_error_y,
    attitude_error_z,
    rate_error_x,
    rate_error_y,
    rate_error_z
]

Control Law
-----------
u = -Kx

This controller is memoryless.
"""

import numpy as np
from scipy.linalg import solve_continuous_are


class LQR:
    """
    Continuous-Time Linear Quadratic Regulator.
    """

    def __init__(
        self,
        inertia_matrix: np.ndarray,
        state_weight: np.ndarray,
        control_weight: np.ndarray,
    ):

        self.inertia = self._validate_inertia(
            inertia_matrix
        )

        self.Q = self._validate_state_weight(
            state_weight
        )

        self.R = self._validate_control_weight(
            control_weight
        )

        self.A = self._build_A()

        self.B = self._build_B()

        self.K = self._compute_gain()

    # ======================================================
    # Public Interface
    # ======================================================

    def compute(
        self,
        current_quaternion: np.ndarray,
        desired_quaternion: np.ndarray,
        body_rates: np.ndarray,
        desired_body_rates: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute desired body control torque.

        Parameters
        ----------
        current_quaternion : ndarray (4,)
        desired_quaternion : ndarray (4,)
        body_rates : ndarray (3,)
        desired_body_rates : ndarray (3,), optional

        Returns
        -------
        ndarray (3,)
            Desired body torque [N·m]
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

        q_error = self._quaternion_error(
            current_quaternion,
            desired_quaternion,
        )

        # Shortest rotation

        if q_error[0] < 0.0:

            q_error = -q_error

        # Small-angle approximation

        attitude_error = 2.0 * q_error[1:]

        rate_error = body_rates - desired_body_rates

        state = np.concatenate(

            (
                attitude_error,
                rate_error,
            )

        )

        torque = -self.K @ state

        return torque

    # ======================================================
    # Linearized Dynamics
    # ======================================================

    def _build_A(self):

        A = np.zeros((6, 6))

        A[0:3, 3:6] = 0.5 * np.eye(3)

        return A

    def _build_B(self):

        B = np.zeros((6, 3))

        B[3:6, :] = np.linalg.inv(
            self.inertia
        )

        return B

    def _compute_gain(self):

        P = solve_continuous_are(

            self.A,

            self.B,

            self.Q,

            self.R,

        )

        return (

            np.linalg.inv(self.R)

            @ self.B.T

            @ P

        )

    # ======================================================
    # Quaternion Utilities
    # ======================================================

    @staticmethod
    def _quaternion_error(
        current,
        desired,
    ):

        return LQR._quat_multiply(

            desired,

            LQR._quat_conjugate(current),

        )

    @staticmethod
    def _quat_conjugate(q):

        return np.array(

            [
                q[0],
                -q[1],
                -q[2],
                -q[3],
            ]

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

            ]

        )

    # ======================================================
    # Validation
    # ======================================================

    @staticmethod
    def _validate_vector(
        vector,
        name,
    ):

        vector = np.asarray(
            vector,
            dtype=float,
        )

        if vector.shape != (3,):

            raise ValueError(
                f"{name} must have shape (3,)."
            )

        if not np.all(
            np.isfinite(vector)
        ):

            raise ValueError(
                f"{name} contains invalid values."
            )

        return vector

    @staticmethod
    def _validate_quaternion(q):

        q = np.asarray(
            q,
            dtype=float,
        )

        if q.shape != (4,):

            raise ValueError(
                "Quaternion must have shape (4,)."
            )

        if not np.all(
            np.isfinite(q)
        ):

            raise ValueError(
                "Quaternion contains invalid values."
            )

        norm = np.linalg.norm(q)

        if norm < 1e-12:

            raise ValueError(
                "Quaternion norm cannot be zero."
            )

        return q / norm

    @staticmethod
    def _validate_inertia(J):

        J = np.asarray(
            J,
            dtype=float,
        )

        if J.shape != (3, 3):

            raise ValueError(
                "inertia_matrix must have shape (3,3)."
            )

        if not np.allclose(
            J,
            J.T,
        ):

            raise ValueError(
                "Inertia matrix must be symmetric."
            )

        if np.any(
            np.linalg.eigvals(J) <= 0.0
        ):

            raise ValueError(
                "Inertia matrix must be positive definite."
            )

        return J

    @staticmethod
    def _validate_state_weight(Q):

        Q = np.asarray(
            Q,
            dtype=float,
        )

        if Q.shape != (6, 6):

            raise ValueError(
                "Q must have shape (6,6)."
            )

        if not np.allclose(
            Q,
            Q.T,
        ):

            raise ValueError(
                "Q must be symmetric."
            )

        if np.any(
            np.linalg.eigvals(Q) <= 0.0
        ):

            raise ValueError(
                "Q must be positive definite."
            )

        return Q

    @staticmethod
    def _validate_control_weight(R):

        R = np.asarray(
            R,
            dtype=float,
        )

        if R.shape != (3, 3):

            raise ValueError(
                "R must have shape (3,3)."
            )

        if not np.allclose(
            R,
            R.T,
        ):

            raise ValueError(
                "R must be symmetric."
            )

        if np.any(
            np.linalg.eigvals(R) <= 0.0
        ):

            raise ValueError(
                "R must be positive definite."
            )

        return R