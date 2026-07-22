"""
scenarios/nadir_pointing/run.py

Mission execution entry point.
"""

from __future__ import annotations


def run(scenario):
    """
    Execute the complete mission.

    Parameters
    ----------
    scenario : NadirPointingScenario

    Returns
    -------
    dict
        Mission telemetry.
    """

    scenario.initialize()

    while not scenario.simulator.finished:

        scenario.update()

    telemetry = scenario.finalize()

    return telemetry