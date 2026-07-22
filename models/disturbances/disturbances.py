"""
models/disturbances/disturbances.py

Disturbance Manager

Registers and evaluates spacecraft disturbance models.

Each disturbance model is responsible for computing a single physical
quantity (force or torque). The disturbance manager simply evaluates
all registered models and returns their contributions.
"""

from typing import List


class Disturbances:
    """
    Disturbance model manager.
    """

    def __init__(self):
        """Initialize an empty disturbance manager."""

        self._disturbance_models: List[object] = []

    # ==========================================================
    # Registration
    # ==========================================================

    def add(self, disturbance_model):
        """
        Register a disturbance model.

        Parameters
        ----------
        disturbance_model : object
            Object implementing a compute() method.
        """

        if not callable(getattr(disturbance_model, "compute", None)):
            raise TypeError(
                "Disturbance model must implement a compute() method."
            )

        self._disturbance_models.append(disturbance_model)

    def remove(self, disturbance_model):
        """Remove a disturbance model."""

        self._disturbance_models.remove(disturbance_model)

    def clear(self):
        """Remove all disturbance models."""

        self._disturbance_models.clear()

    # ==========================================================
    # Evaluation
    # ==========================================================

    def compute(self, **kwargs):
        """
        Evaluate all registered disturbance models.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to each disturbance model.

        Returns
        -------
        disturbances : list
            List containing the output from each registered
            disturbance model.
        """

        return [
            disturbance.compute(**kwargs)
            for disturbance in self._disturbance_models
        ]