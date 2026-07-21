"""
models/environment/orbit.py

Keplerian orbit model for 600 km SSO.
"""

import numpy as np
from config import ALTITUDE_KM, MU, R_EARTH, n  # mean motion from config


class Orbit:
    def __init__(self):
        self.altitude = ALTITUDE_KM * 1000.0
        self.a = R_EARTH + self.altitude
        self.n = n  # mean motion
        self.period = 2 * np.pi / self.n
        self.time = 0.0

        # Initial conditions for Sun-Synchronous Orbit (simplified)
        self.inclination = np.deg2rad(98.0)  # typical SSO

    def get_position_velocity(self, t: float):
        """Return position and velocity in ECI (simplified circular orbit)"""
        theta = self.n * t  # true anomaly
        r = self.a

        # Simplified ECI position (equatorial for basic propagation, can be improved)
        pos_eci = r * np.array([np.cos(theta), np.sin(theta), 0.0])
        vel_eci = r * self.n * np.array([-np.sin(theta), np.cos(theta), 0.0])

        self.time = t
        return pos_eci, vel_eci

    def reset(self):
        self.time = 0.0