"""
analysis/report.py

Automatic simulation report generation.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import compute_metrics

from .plotting import (
    plot_attitude,
    plot_body_rates,
    plot_control_torque,
    plot_disturbance_torque,
    plot_wheel_momentum,
    plot_magnetic_field,
    plot_orbit,
)


def generate_report(
    telemetry: dict,
    output_directory: str = "analysis",
):
    """
    Generate all figures and simulation report.
    """

    figures_directory = Path(output_directory) / "figures"
    results_directory = Path(output_directory) / "results"

    figures_directory.mkdir(parents=True, exist_ok=True)
    results_directory.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")

    plot_attitude(
        telemetry,
        figures_directory,
    )

    plot_body_rates(
        telemetry,
        figures_directory,
    )

    plot_control_torque(
        telemetry,
        figures_directory,
    )

    plot_disturbance_torque(
        telemetry,
        figures_directory,
    )

    plot_wheel_momentum(
        telemetry,
        figures_directory,
    )

    plot_magnetic_field(
        telemetry,
        figures_directory,
    )

    plot_orbit(
        telemetry,
        figures_directory,
    )

    metrics = compute_metrics(
        telemetry,
    )

    report = results_directory / "simulation_report.md"

    with open(report, "w") as f:

        f.write("# Earth Observation Satellite Simulation Report\n\n")

        f.write("## Summary\n\n")

        for key, value in metrics.items():

            f.write(f"- **{key.replace('_',' ').title()}** : {value}\n")

    print("✔ Figures saved")

    print("✔ Report saved")

    print(results_directory / "simulation_report.md")