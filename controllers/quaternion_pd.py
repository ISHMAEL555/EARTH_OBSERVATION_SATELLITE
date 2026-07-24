"""
controllers/quaternion_pd.py

Quaternion Proportional-Derivative Attitude Tracking Controller.

Implements the nonlinear quaternion PD control law

    τ = -Kp qe - Kd (ω - ωref)

where

    qe = q_current ⊗ q_ref⁻¹

This error definition is consistent with the Body-to-ECI quaternion
convention used throughout the spacecraft dynamics model.

References
----------
- Markley & Crassidis,
  Fundamentals of Spacecraft Attitude Determination and Control

- Wie,
  Space Vehicle Dynamics and Control
"""

from __future__ import annotations

import numpy as np

from models.dynamics.quaternion import (
    normalize,
    enforce_unique,
    multiply,
    inverse,
)


class QuaternionPD:
    """
    Quaternion Proportional-Derivative attitude tracking controller.
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

        # Debug counter
        self._debug_counter = 0

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
        Compute the commanded body torque.

        Parameters
        ----------
        current_quaternion : ndarray (4,)
            Current spacecraft attitude quaternion
            (Body → ECI).

        desired_quaternion : ndarray (4,)
            Desired reference quaternion
            (Body → ECI).

        body_rates : ndarray (3,)
            Current body angular velocity [rad/s].

        desired_body_rates : ndarray (3,), optional
            Desired body angular velocity [rad/s].

        Returns
        -------
        ndarray (3,)
            Commanded control torque [N·m].
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

        # --------------------------------------------------
        # Quaternion Error
        # --------------------------------------------------

        q_error = multiply(
            current_quaternion,
            inverse(desired_quaternion),
        )

        q_error = enforce_unique(
            q_error
        )

        attitude_error = q_error[1:]

        # --------------------------------------------------
        # Angular Velocity Error
        # --------------------------------------------------

        rate_error = (
            body_rates
            - desired_body_rates
        )

        # ======================================================
        # DEBUG (Print first 10 controller calls)
        # ======================================================

        if self._debug_counter < 10:

            print("\n======================================")
            print(f"Controller Step {self._debug_counter}")

            print("\nCurrent Quaternion")
            print(current_quaternion)

            print("\nDesired Quaternion")
            print(desired_quaternion)

            print("\nQuaternion Error")
            print(q_error)

            print("\nAttitude Error")
            print(attitude_error)

            print("\nCurrent Body Rates")
            print(body_rates)

            print("\nDesired Body Rates")
            print(desired_body_rates)

            print("\nRate Error")
            print(rate_error)

        self._debug_counter += 1

        # --------------------------------------------------
        # Quaternion PD Control Law
        # --------------------------------------------------

        commanded_torque = (

            -self.Kp @ attitude_error

            -self.Kd @ rate_error

        )

        if self._debug_counter <= 10:

            print("\nCommanded Torque")
            print(commanded_torque)
            print("======================================")

        return commanded_torque

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

        if np.linalg.norm(q) < 1e-12:
            raise ValueError(
                "Quaternion norm cannot be zero."
            )

        return enforce_unique(
            normalize(q)
        )