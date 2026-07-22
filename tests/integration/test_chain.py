"""
Integration Test 01

Simulation Chain

Orbit
↓

Controller
↓

Reaction Wheels
↓

Spacecraft
"""

import numpy as np

from models.environment.orbit import Orbit
from models.spacecraft import Spacecraft
from controllers.quaternion_pd import QuaternionPD
from models.actuators.rw_pyramid import RWPyramid


def test_simulation_chain():

    # ------------------------------------------
    # Orbit
    # ------------------------------------------

    orbit = Orbit(
        mu=3.986004418e14,
        semi_major_axis=6878e3,
    )

    position, velocity = orbit.propagate(0.0)

    assert position.shape == (3,)
    assert velocity.shape == (3,)

    # ------------------------------------------
    # Spacecraft
    # ------------------------------------------

    spacecraft = Spacecraft(
        inertia=np.diag([20.0, 30.0, 40.0]),
        mass=100.0,
    )

    # ------------------------------------------
    # Controller
    # ------------------------------------------

    controller = QuaternionPD(
        proportional_gain=np.eye(3),
        derivative_gain=np.eye(3),
    )

    torque = controller.compute(

        current_quaternion=spacecraft.q,

        desired_quaternion=np.array(
            [1.0,0.0,0.0,0.0]
        ),

        body_rates=spacecraft.omega,

    )

    assert torque.shape == (3,)

    # ------------------------------------------
    # Reaction Wheels
    # ------------------------------------------

    wheel_axes = np.array(

        [

            [1,0,0,0],

            [0,1,0,0],

            [0,0,1,1],

        ],

        dtype=float,

    )

    rw = RWPyramid(

        wheel_axes=wheel_axes,

        max_torque=0.1,

        max_momentum=10.0,

    )

    _, _, body_torque = rw.update(

        torque,

        0.1,

    )

    assert body_torque.shape == (3,)

    # ------------------------------------------
    # Spacecraft Dynamics
    # ------------------------------------------

    spacecraft.propagate(

        body_torque,

        0.1,

    )

    assert spacecraft.q.shape == (4,)

    assert spacecraft.omega.shape == (3,)