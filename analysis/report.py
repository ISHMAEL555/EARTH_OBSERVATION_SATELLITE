"""
analysis/report.py

Automatic engineering report generation for the
Earth Observation Satellite simulator.
"""

from __future__ import annotations

from pathlib import Path

from analysis.metrics import compute_metrics

# -----------------------------
# Plotting Modules
# -----------------------------

from analysis.plotting.orbit import plot_orbit

from analysis.plotting.attitude import plot_attitude

from analysis.plotting.attitude_error import plot_attitude_error

from analysis.plotting.quaternion_norm import plot_quaternion_norm

from analysis.plotting.body_rates import plot_body_rates

from analysis.plotting.body_rate_norm import plot_body_rate_norm

from analysis.plotting.control_torque import plot_control_torque

from analysis.plotting.control_effort import plot_control_effort

from analysis.plotting.disturbance_torque import plot_disturbance_torque

from analysis.plotting.magnetic_field import plot_magnetic_field

from analysis.plotting.wheel_momentum import plot_wheel_momentum

from analysis.plotting.simulation_dashboard import (
    plot_simulation_dashboard,
)


# ==========================================================
# Generate Report
# ==========================================================

def generate_report(
    telemetry,
    output_directory="analysis/results",
):
    """
    Generate a complete engineering report.
    """

    output_directory = Path(output_directory)

    figures = output_directory / "figures"

    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()

    print("="*60)

    print("Generating Engineering Report")

    print("="*60)

    # ------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------

    metrics = compute_metrics(
        telemetry
    )

    # ------------------------------------------------------
    # Generate Figures
    # ------------------------------------------------------

    plot_orbit(
        telemetry,
        figures,
    )

    plot_attitude(
        telemetry,
        figures,
    )

    plot_attitude_error(
        telemetry,
        figures,
    )

    plot_quaternion_norm(
        telemetry,
        figures,
    )

    plot_body_rates(
        telemetry,
        figures,
    )

    plot_body_rate_norm(
        telemetry,
        figures,
    )

    plot_control_torque(
        telemetry,
        figures,
    )

    plot_control_effort(
        telemetry,
        figures,
    )

    plot_disturbance_torque(
        telemetry,
        figures,
    )

    plot_magnetic_field(
        telemetry,
        figures,
    )

    plot_wheel_momentum(
        telemetry,
        figures,
    )

    plot_simulation_dashboard(
        telemetry,
        figures,
    )

    # ------------------------------------------------------
    # Metrics Summary
    # ------------------------------------------------------

    report_file = output_directory / "summary.txt"

    with open(
        report_file,
        "w",
    ) as f:

        f.write(
            "EARTH OBSERVATION SATELLITE\n"
        )

        f.write(
            "Engineering Verification Summary\n\n"
        )

        f.write(
            f"Maximum Attitude Error     : {metrics.maximum_attitude_error:.6f} deg\n"
        )

        f.write(
            f"RMS Attitude Error         : {metrics.rms_attitude_error:.6f} deg\n"
        )

        f.write(
            f"Settling Time              : {metrics.settling_time:.3f} s\n"
        )

        f.write(
            f"Maximum Body Rate          : {metrics.maximum_body_rate:.6e} rad/s\n"
        )

        f.write(
            f"RMS Body Rate              : {metrics.rms_body_rate:.6e} rad/s\n"
        )

        f.write(
            f"Maximum Control Torque     : {metrics.maximum_control_torque:.6e} Nm\n"
        )

        f.write(
            f"RMS Control Torque         : {metrics.rms_control_torque:.6e} Nm\n"
        )

        f.write(
            f"Maximum Wheel Momentum     : {metrics.maximum_wheel_momentum:.6f} Nms\n"
        )

        f.write(
            f"Quaternion Norm Error      : {metrics.quaternion_norm_error:.6e}\n"
        )

    print()

    print("✓ Figures saved to")

    print(figures)

    print()

    print("✓ Metrics summary saved to")

    print(report_file)

    print()

    print("="*60)

    print("Report generation complete")

    print("="*60)

    return metrics