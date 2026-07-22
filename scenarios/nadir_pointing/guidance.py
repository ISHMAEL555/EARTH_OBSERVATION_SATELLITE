"""
scenarios/nadir_pointing/guidance.py

Mission guidance wrapper.

Responsible for computing the desired spacecraft attitude
and body-rate references.

The actual guidance algorithm is implemented in the
top-level guidance package.
"""

from __future__ import annotations

import numpy as np


# ==========================================================
# Public Interface
# ==========================================================

def update_guidance(scenario) -> None:
    """
    Update the mission guidance solution.

    Parameters
    ----------
    scenario : NadirPointingScenario
    """

    (
        scenario.q_ref,
        scenario.omega_ref,
    ) = scenario.guidance.compute(

        position_eci=scenario.position_eci,

        velocity_eci=scenario.velocity_eci,

    )


# ==========================================================
# Convenience Functions
# ==========================================================

def reference_quaternion(scenario) -> np.ndarray:
    """
    Return the desired attitude quaternion.
    """

    return scenario.q_ref


def reference_body_rates(scenario) -> np.ndarray:
    """
    Return the desired body rates.
    """

    return scenario.omega_ref