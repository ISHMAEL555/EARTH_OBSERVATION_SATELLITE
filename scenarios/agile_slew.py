"""
=========================================================
MISSION SCENARIO : AGILE SLEW MANEUVER
=========================================================

Mission Phase
-------------
Target Acquisition

Description
-----------
Rotate the spacecraft from one imaging target to another
using a quaternion-based reference trajectory.

Objectives
----------
- Execute a 45° attitude maneuver.
- Complete the maneuver within the required time.
- Minimize overshoot and settling time.
- Respect reaction wheel torque and speed limits.

Primary Actuator
----------------
- Four Reaction Wheels (Pyramidal Configuration)

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
- Slew Time
- Settling Time
- Peak Torque
- Peak Wheel Speed
- Final Pointing Error
"""