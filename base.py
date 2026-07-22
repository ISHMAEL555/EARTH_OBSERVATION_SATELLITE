"""
base.py

Earth Observation Satellite Simulation Launcher.
"""

from simulator.builder import Builder
from simulator.simulator import Simulator

from scenarios.nadir_pointing.run import run
from scenarios.nadir_pointing.scenario import NadirPointingScenario

from scenarios.nadir_pointing.config import (
    MISSION,
    CONTROLLER,
    INITIAL_CONDITIONS,
)

from guidance.nadir_pointing import NadirPointingGuidance


# ==========================================================
# Builders
# ==========================================================

def build_simulation():

    builder = Builder(
        mission=MISSION,
        controller=CONTROLLER,
        initial_conditions=INITIAL_CONDITIONS,
    )

    return builder.build()


def build_guidance():

    return NadirPointingGuidance()


# ==========================================================
# Main
# ==========================================================

def main():

    simulation = build_simulation()

    simulator = Simulator(simulation)

    guidance = build_guidance()

    scenario = NadirPointingScenario(
        simulation=simulation,
        simulator=simulator,
        guidance=guidance,
    )

    telemetry = run(scenario)

    return telemetry


if __name__ == "__main__":

    telemetry = main()

    print("Simulation completed successfully.\n")

    print(f"Samples : {len(telemetry['time'])}")

    print(f"Final Time : {telemetry['time'][-1]:.2f} s")

    print("\nTelemetry Channels")

    for key in telemetry.keys():

        print(f" - {key}")