"""
models/environment/orbit.py

Circular orbit propagator for a 600 km Sun-Synchronous Earth Observation
satellite.

This module is responsible only for orbital motion. It propagates the
spacecraft position and velocity in the Earth-Centered Inertial (ECI)
frame using a simple circular orbit assumption.

State
-----
position : ndarray (3,)
    Spacecraft position in ECI frame [m]

velocity : ndarray (3,)
    Spacecraft velocity in ECI frame [m/s]

Public Methods
--------------
update(t)
    Propagate the orbit to time t.

reset()
    Reset the orbit to the initial state.
"""

import numpy as np

from config import (
    ALTITUDE_KM,
    R_EARTH,
    n,
)


class Orbit:
    """
    Circular Sun-Synchronous orbit model.

    Notes
    -----
    Assumptions
    -----------
    - Circular orbit
    - Constant altitude
    - Constant mean motion
    - Two-body dynamics
    - ECI reference frame
    """

    def __init__(self):
        """Initialize orbital parameters."""

        # Orbital parameters
        self.altitude = ALTITUDE_KM * 1000.0          # [m]
        self.semi_major_axis = R_EARTH + self.altitude
        self.radius = self.semi_major_axis

        self.mean_motion = n                          # [rad/s]
        self.period = 2.0 * np.pi / self.mean_motion

        # Orbital elements (reserved for future expansion)
        self.eccentricity = 0.0
        self.inclination = np.deg2rad(98.0)
        self.raan = 0.0
        self.argument_of_perigee = 0.0
        self.true_anomaly = 0.0

        # Time
        self.time = 0.0

        # State vectors
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

        # Initialize orbit
        self.update(0.0)

    # ==========================================================
    # Orbit Propagation
    # ==========================================================

    def update(self, t: float) -> None:
        """
        Propagate the orbit to time t.

        Parameters
        ----------
        t : float
            Simulation time [s]
        """

        self.time = t

        # True anomaly (circular orbit)
        theta = self.mean_motion * t
        self.true_anomaly = theta

        r = self.semi_major_axis

        # Position in ECI frame
        self.position = r * np.array([
            np.cos(theta),
            np.sin(theta),
            0.0,
        ])

        # Velocity in ECI frame
        self.velocity = (
            r * self.mean_motion *
            np.array([
                -np.sin(theta),
                np.cos(theta),
                0.0,
            ])
        )

        self.radius = np.linalg.norm(self.position)

    # ==========================================================
    # Utility Functions
    # ==========================================================

    def reset(self) -> None:
        """Reset the orbit to the initial state."""

        self.update(0.0)