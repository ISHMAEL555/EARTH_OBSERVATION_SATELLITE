"""
verification/verify_closed_loop.py

Closed-Loop Verification Driver

Runs the complete nadir-pointing mission while the
Verification Monitor inspects every subsystem.

Author
------
Ishmael
"""

from __future__ import annotations

from simulator.builder import Builder
from simulator.simulator import Simulator

from guidance.nadir_pointing import (
    NadirPointingGuidance,
)

from scenarios.nadir_pointing.config import (
    MISSION,
    CONTROLLER,
    INITIAL_CONDITIONS,
)

from scenarios.nadir_pointing.scenario import (
    NadirPointingScenario,
)

from scenarios.nadir_pointing.run import run

from verification.monitor import (
    VerificationMonitor,
)


def main():
    """
    Execute the complete mission with verification enabled.
    """

    # ==========================================================
    # Build Complete Simulation
    # ==========================================================

    builder = Builder(
        mission=MISSION,
        controller=CONTROLLER,
        initial_conditions=INITIAL_CONDITIONS,
    )

    simulation = builder.build()

    # ==========================================================
    # Create Simulator
    # ==========================================================

    simulator = Simulator(
        simulation=simulation,
    )

    # ==========================================================
    # Guidance Law
    # ==========================================================

    guidance = NadirPointingGuidance()

    # ==========================================================
    # Mission Scenario
    # ==========================================================

    scenario = NadirPointingScenario(
        simulation=simulation,
        simulator=simulator,
        guidance=guidance,
    )

    # ==========================================================
    # Verification Monitor
    # ==========================================================

    monitor = VerificationMonitor(
        first_n_steps=20,
    )

    print("=" * 70)
    print("Closed-Loop Verification")
    print("=" * 70)

    telemetry = run(
        scenario=scenario,
        monitor=monitor,
    )

    print("\n" + "=" * 70)
    print("Verification Completed Successfully")
    print("=" * 70)

    return telemetry


if __name__ == "__main__":
    main()

