"""
base.py

Earth Observation Satellite Simulation Launcher.
"""

import numpy as np

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

from analysis import generate_report

from models.dynamics.quaternion import (
    multiply,
    inverse,
)


# ==========================================================
# Builders
# ==========================================================

def build_simulation():
    """
    Construct the spacecraft simulation.
    """

    builder = Builder(
        mission=MISSION,
        controller=CONTROLLER,
        initial_conditions=INITIAL_CONDITIONS,
    )

    return builder.build()


def build_guidance():
    """
    Construct the guidance law.
    """

    return NadirPointingGuidance()


# ==========================================================
# Simulation
# ==========================================================

def main():
    """
    Execute the complete spacecraft simulation.
    """

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


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    telemetry = main()

    # ======================================================
    # Debug
    # ======================================================

    print("\n" + "=" * 70)
    print("DEBUG INFORMATION")
    print("=" * 70)

    print("\nINITIAL STATE")
    print("-" * 70)

    print("Current Quaternion:")
    print(telemetry["quaternion"][0])

    print("\nReference Quaternion:")
    print(telemetry["reference_quaternion"][0])

    print("\nCurrent Body Rates [rad/s]:")
    print(telemetry["body_rates"][0])

    print("\nReference Body Rates [rad/s]:")
    print(telemetry["reference_body_rates"][0])

    print("\nFINAL STATE")
    print("-" * 70)

    print("Current Quaternion:")
    print(telemetry["quaternion"][-1])

    print("\nReference Quaternion:")
    print(telemetry["reference_quaternion"][-1])

    print("\nCurrent Body Rates [rad/s]:")
    print(telemetry["body_rates"][-1])

    print("\nReference Body Rates [rad/s]:")
    print(telemetry["reference_body_rates"][-1])

    # ------------------------------------------------------
    # Quaternion Error
    # ------------------------------------------------------

    q_error = multiply(
        telemetry["reference_quaternion"][-1],
        inverse(telemetry["quaternion"][-1]),
    )

    print("\nQuaternion Error:")
    print(q_error)

    angle_error = np.degrees(
        2.0 * np.arccos(
            np.clip(abs(q_error[0]), -1.0, 1.0)
        )
    )

    print(f"\nComputed Angle Error : {angle_error:.6f} deg")

    print("=" * 70)

    # ======================================================

    print("\n" + "=" * 70)
    print("EARTH OBSERVATION SATELLITE SIMULATION")
    print("=" * 70)

    print("\nSimulation completed successfully.")

    print(f"\nSamples      : {len(telemetry['time'])}")
    print(f"Final Time   : {telemetry['time'][-1]:.2f} s")

    print("\nTelemetry Channels")
    print("-" * 70)

    for key in telemetry.keys():
        print(f"• {key}")

    print("\n" + "=" * 70)
    print("Generating Engineering Report...")
    print("=" * 70)

    metrics = generate_report(
        telemetry,
        output_directory="analysis/results",
    )

    print("\nEngineering Performance")
    print("-" * 70)

    print(f"Maximum Attitude Error : {metrics.maximum_attitude_error:.6f} deg")
    print(f"RMS Attitude Error     : {metrics.rms_attitude_error:.6f} deg")
    print(f"Settling Time          : {metrics.settling_time:.3f} s")

    print(f"Maximum Body Rate      : {metrics.maximum_body_rate:.6e} rad/s")
    print(f"RMS Body Rate          : {metrics.rms_body_rate:.6e} rad/s")

    print(f"Maximum Control Torque : {metrics.maximum_control_torque:.6e} Nm")
    print(f"RMS Control Torque     : {metrics.rms_control_torque:.6e} Nm")

    print(f"Maximum RW Momentum    : {metrics.maximum_wheel_momentum:.6f} N·m·s")

    print(f"Quaternion Norm Error  : {metrics.quaternion_norm_error:.6e}")

    print("-" * 70)
    print("Analysis complete.")
    print("=" * 70)