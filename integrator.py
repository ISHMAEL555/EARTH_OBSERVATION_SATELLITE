"""
simulation/integrator.py

Fixed-step Runge-Kutta 4 integrator for high accuracy.
"""

import numpy as np


def rk4_step(func, state, t, dt, *args):
    """
    Runge-Kutta 4th order integration step.
    
    Parameters
    ----------
    func : callable
        Derivative function: deriv = func(state, t, *args)
    state : np.ndarray
        Current state vector
    t : float
        Current time
    dt : float
        Time step
    
    Returns
    -------
    new_state : np.ndarray
    """
    k1 = func(state, t, *args)
    k2 = func(state + 0.5*dt*k1, t + 0.5*dt, *args)
    k3 = func(state + 0.5*dt*k2, t + 0.5*dt, *args)
    k4 = func(state + dt*k3, t + dt, *args)
    
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)