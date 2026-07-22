"""
scenarios/nadir_pointing/plotting.py

Mission plotting utilities.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class MissionPlots:
    """
    Plot mission telemetry.
    """

    def __init__(self, telemetry):

        self.telemetry = telemetry

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def plot_all(self):

        self.plot_quaternion()

        self.plot_body_rates()

        self.plot_control_torque()

        self.plot_wheel_momentum()

        plt.show()

    # --------------------------------------------------------
    # Quaternion
    # --------------------------------------------------------

    def plot_quaternion(self):

        q = self.telemetry["quaternion"]

        t = self.telemetry["time"]

        plt.figure(figsize=(10,5))

        plt.plot(t, q[:,0], label="q0")
        plt.plot(t, q[:,1], label="q1")
        plt.plot(t, q[:,2], label="q2")
        plt.plot(t, q[:,3], label="q3")

        plt.title("Quaternion")

        plt.xlabel("Time [s]")

        plt.ylabel("Quaternion")

        plt.grid(True)

        plt.legend()

    # --------------------------------------------------------
    # Body Rates
    # --------------------------------------------------------

    def plot_body_rates(self):

        omega = self.telemetry["body_rates"]

        t = self.telemetry["time"]

        plt.figure(figsize=(10,5))

        plt.plot(t, omega[:,0], label="ωx")

        plt.plot(t, omega[:,1], label="ωy")

        plt.plot(t, omega[:,2], label="ωz")

        plt.grid(True)

        plt.legend()

        plt.title("Body Rates")

        plt.xlabel("Time [s]")

        plt.ylabel("rad/s")

    # --------------------------------------------------------
    # Control Torque
    # --------------------------------------------------------

    def plot_control_torque(self):

        torque = self.telemetry["control_torque"]

        t = self.telemetry["time"]

        plt.figure(figsize=(10,5))

        plt.plot(t, torque[:,0], label="Tx")

        plt.plot(t, torque[:,1], label="Ty")

        plt.plot(t, torque[:,2], label="Tz")

        plt.grid(True)

        plt.legend()

        plt.title("Control Torque")

        plt.xlabel("Time [s]")

        plt.ylabel("Nm")

    # --------------------------------------------------------
    # Wheel Momentum
    # --------------------------------------------------------

    def plot_wheel_momentum(self):

        H = self.telemetry["wheel_momentum"]

        t = self.telemetry["time"]

        plt.figure(figsize=(10,5))

        plt.plot(t, H)

        plt.grid(True)

        plt.title("Reaction Wheel Momentum")

        plt.xlabel("Time [s]")

        plt.ylabel("Nms")