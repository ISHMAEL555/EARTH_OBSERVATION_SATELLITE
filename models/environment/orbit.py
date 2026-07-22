"""
models/environment/orbit.py

Two-body circular orbit propagator.

This module computes the spacecraft position and velocity in the
Earth-Centered Inertial (ECI) frame for a circular orbit.

The model is completely independent of:

- Simulation
- Spacecraft
- Controllers
- Sensors
- Actuators

Current Model
-------------
- Circular Orbit
- Two-body dynamics

Future Models
-------------
- Keplerian orbit propagation
- J2 perturbation
- SGP4
- Numerical propagator
"""

import numpy as np


class Orbit:
    """
    Circular orbit propagator.

    Parameters
    ----------
    mu : float
        Gravitational parameter [m³/s²].

    semi_major_axis : float
        Orbit semi-major axis [m].

    eccentricity : float
        Orbital eccentricity.

    inclination : float
        Inclination [rad].

    raan : float
        Right Ascension of Ascending Node [rad].

    argument_of_perigee : float
        Argument of perigee [rad].

    true_anomaly : float
        Initial true anomaly [rad].
    """

    def __init__(
        self,
        mu: float,
        semi_major_axis: float,
        eccentricity: float = 0.0,
        inclination: float = 0.0,
        raan: float = 0.0,
        argument_of_perigee: float = 0.0,
        true_anomaly: float = 0.0,
    ):

        if mu <= 0.0:
            raise ValueError("mu must be positive.")

        if semi_major_axis <= 0.0:
            raise ValueError(
                "semi_major_axis must be positive."
            )

        if not (0.0 <= eccentricity < 1.0):
            raise ValueError(
                "eccentricity must satisfy 0 ≤ e < 1."
            )

        self.mu = float(mu)

        self.semi_major_axis = float(
            semi_major_axis
        )

        self.eccentricity = float(
            eccentricity
        )

        self.inclination = float(
            inclination
        )

        self.raan = float(
            raan
        )

        self.argument_of_perigee = float(
            argument_of_perigee
        )

        self.initial_true_anomaly = float(
            true_anomaly
        )

        # Circular orbit mean motion
        self.mean_motion = np.sqrt(
            self.mu / self.semi_major_axis**3
        )

        self.period = (
            2.0 * np.pi / self.mean_motion
        )

    # ======================================================
    # Orbit Propagation
    # ======================================================

    def propagate(
        self,
        time: float,
    ):
        """
        Propagate the orbit.

        Parameters
        ----------
        time : float
            Simulation time [s].

        Returns
        -------
        position_eci : ndarray (3,)
            Position in ECI frame [m].

        velocity_eci : ndarray (3,)
            Velocity in ECI frame [m/s].
        """

        if not np.isscalar(time):
            raise TypeError(
                "time must be a scalar."
            )

        if time < 0.0:
            raise ValueError(
                "time must be non-negative."
            )

        theta = (
            self.initial_true_anomaly
            + self.mean_motion * time
        )

        r = self.semi_major_axis

        position_eci = r * np.array(
            [
                np.cos(theta),
                np.sin(theta),
                0.0,
            ]
        )

        velocity_eci = (
            r
            * self.mean_motion
            * np.array(
                [
                    -np.sin(theta),
                    np.cos(theta),
                    0.0,
                ]
            )
        )

        return (
            position_eci,
            velocity_eci,
        )