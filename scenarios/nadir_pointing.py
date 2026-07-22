"""
=========================================================
MISSION SCENARIO : NADIR POINTING
=========================================================

Mission Phase
-------------
Nominal Earth Observation

Description
-----------
Maintain continuous nadir pointing during imaging while
rejecting environmental disturbances.

Objectives
----------
- Maintain high pointing accuracy.
- Minimize attitude jitter.
- Maintain reaction wheel momentum within limits.
- Periodically unload wheel momentum.

Primary Actuator
----------------
- Four Reaction Wheels

Secondary Actuator
------------------
- Three Magnetorquers (Momentum Dumping)

Controller
----------
Quaternion PD / LQR

Disturbances
------------
- Gravity Gradient
- Atmospheric Drag
- Solar Radiation Pressure
- Residual Magnetic Torque

Performance Metrics
-------------------
- RMS Pointing Error
- 3σ Pointing Accuracy
- Wheel Momentum
- Control Effort
"""