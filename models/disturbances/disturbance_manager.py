"""
disturbances/disturbance_manager.py

Disturbance Manager

Author
------
Kowluri Ishmael

Description
-----------
Aggregates all environmental disturbance models and computes
the total disturbance torque acting on the spacecraft.

Each disturbance model implements

    compute(state)

where state is an instance of DisturbanceState.

Adding a new disturbance only requires implementing the
same interface and registering it with this manager.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .disturbance_state import DisturbanceState


class DisturbanceManager:
    """
    Manager for spacecraft disturbance models.
    """

    def __init__(self):

        self._models: list[Any] = []

    # =====================================================
    # Registration
    # =====================================================

    def add(
        self,
        model,
    ) -> None:
        """
        Register a disturbance model.

        Parameters
        ----------
        model
            Any class implementing

                compute(state) -> ndarray(3,)
        """

        if not hasattr(
            model,
            "compute",
        ):
            raise TypeError(
                f"{type(model).__name__} "
                "does not implement compute()."
            )

        self._models.append(
            model
        )

    # =====================================================
    # Public Interface
    # =====================================================

    def compute(
        self,
        state: DisturbanceState,
    ) -> np.ndarray:
        """
        Compute total disturbance torque.

        Parameters
        ----------
        state : DisturbanceState

        Returns
        -------
        ndarray, shape (3,)
            Total disturbance torque [N·m]
        """

        total = np.zeros(
            3,
            dtype=float,
        )

        for model in self._models:

            torque = model.compute(
                state
            )

            if torque is None:
                raise ValueError(
                    f"{type(model).__name__} "
                    "returned None."
                )

            torque = np.asarray(
                torque,
                dtype=float,
            )

            if torque.shape != (3,):
                raise ValueError(
                    f"{type(model).__name__} "
                    "returned an invalid torque vector."
                )

            total += torque

        return total

    # =====================================================
    # Utilities
    # =====================================================

    def clear(
        self,
    ) -> None:
        """
        Remove all registered models.
        """

        self._models.clear()

    @property
    def models(
        self,
    ) -> tuple:
        """
        Registered disturbance models.
        """

        return tuple(
            self._models
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self._models
        )

    def __iter__(
        self,
    ):

        return iter(
            self._models
        )

    def __repr__(
        self,
    ) -> str:

        return (
            f"{type(self).__name__}"
            f"(models={len(self)})"
        )