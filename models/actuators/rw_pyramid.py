"""
models/actuators/rw_pyramid.py

Four-Wheel Pyramidal Reaction Wheel Assembly
"""

import numpy as np

from config import ACTUATORS


class RWPyramid:
    """
    Four-wheel pyramidal reaction wheel assembly.
    """

    def __init__(self):

        cfg = ACTUATORS["reaction_wheels"]

        self.num_wheels = cfg["num_wheels"]
        self.wheel_axes = cfg["wheel_axes"]

        self.max_torque = cfg["max_torque"]
        self.max_momentum = cfg["max_momentum"]

        # Allocation matrix
        self.allocation_matrix = self.wheel_axes
        self.allocation_matrix_pinv = np.linalg.pinv(self.allocation_matrix)

        # -------------------------------------------------
        # States
        # -------------------------------------------------

        self.commanded_body_torque = np.zeros(3)

        self.commanded_wheel_torque = np.zeros(self.num_wheels)

        self.actual_wheel_torque = np.zeros(self.num_wheels)

        self.wheel_momentum = np.zeros(self.num_wheels)

        self.body_torque = np.zeros(3)

    # =====================================================
    # Public Interface
    # =====================================================

    def update(
        self,
        commanded_body_torque: np.ndarray,
        dt: float,
    ):
        """
        Advance the reaction wheel assembly by one simulation step.

        Parameters
        ----------
        commanded_body_torque : ndarray (3,)
            Desired spacecraft body torque [Nm]

        dt : float
            Simulation time step [s]
        """

        commanded_body_torque = self._validate_vector(
            commanded_body_torque,
            "commanded_body_torque",
            3,
        )

        self.commanded_body_torque = commanded_body_torque.copy()

        # -------------------------------------------------
        # Torque Allocation
        # -------------------------------------------------

        self.commanded_wheel_torque = (
            self.allocation_matrix_pinv
            @ self.commanded_body_torque
        )

        # -------------------------------------------------
        # Torque Saturation
        # -------------------------------------------------

        self.actual_wheel_torque = np.clip(
            self.commanded_wheel_torque,
            -self.max_torque,
            self.max_torque,
        )

        # -------------------------------------------------
        # Wheel Momentum Integration
        # -------------------------------------------------

        self.wheel_momentum += self.actual_wheel_torque * dt

        self.wheel_momentum = np.clip(
            self.wheel_momentum,
            -self.max_momentum,
            self.max_momentum,
        )

        # -------------------------------------------------
        # Actual Body Torque
        # -------------------------------------------------

        self.body_torque = (
            self.allocation_matrix
            @ self.actual_wheel_torque
        )

    # =====================================================
    # Utility Functions
    # =====================================================

    def get_total_momentum(self) -> np.ndarray:
        """
        Total spacecraft angular momentum stored in the wheel cluster.

        Returns
        -------
        ndarray (3,)
            Total body-frame momentum [N·m·s]
        """

        return self.wheel_axes @ self.wheel_momentum

    def get_total_momentum_capacity(self) -> float:
        """
        Total scalar momentum capacity.

        Returns
        -------
        float
            Total momentum capacity [N·m·s]
        """

        return self.num_wheels * self.max_momentum

    def reset(self):
        """
        Reset reaction wheel states.
        """

        self.commanded_body_torque.fill(0.0)

        self.commanded_wheel_torque.fill(0.0)

        self.actual_wheel_torque.fill(0.0)

        self.wheel_momentum.fill(0.0)

        self.body_torque.fill(0.0)

    # =====================================================
    # Private Functions
    # =====================================================

    @staticmethod
    def _validate_vector(
        vector: np.ndarray,
        name: str,
        size: int,
    ) -> np.ndarray:
        """
        Validate vector inputs.
        """

        vector = np.asarray(vector, dtype=float)

        if vector.shape != (size,):
            raise ValueError(
                f"{name} must be a ({size},) vector."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} contains invalid values."
            )

        return vector