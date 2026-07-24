"""
scenarios/nadir_pointing/run.py

Mission execution entry point.
"""

from __future__ import annotations

from typing import Any


def run(
    scenario,
    monitor: Any | None = None,
):
    """
    Execute the complete mission.

    Parameters
    ----------
    scenario : NadirPointingScenario
        Mission scenario.

    monitor : object, optional
        Verification monitor executed after every
        simulation step. If None, verification is skipped.

    Returns
    -------
    dict
        Mission telemetry.
    """

    # ======================================================
    # Initialize Mission
    # ======================================================

    scenario.initialize()

    # ======================================================
    # Mission Loop
    # ======================================================

    while not scenario.simulator.finished:

        scenario.update()

        # --------------------------------------------------
        # Verification
        # --------------------------------------------------

        if monitor is not None:

            monitor.inspect(scenario)

    # ======================================================
    # Finalize
    # ======================================================

    telemetry = scenario.finalize()

    return telemetry