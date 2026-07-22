### Hello world

# Hybrid ADCS for an Agile Earth Observation Satellite

Design and implementation of a **Hybrid Attitude Control System (ACS)** for an agile Low Earth Orbit (LEO) Earth Observation satellite using **Reaction Wheels (RW)**, **Variable-Speed Control Moment Gyroscopes (VSCMG)**, and **Magnetorquers**.

The project focuses on the **control subsystem** of the spacecraft. The spacecraft attitude is assumed to be known, allowing the study to concentrate on controller design, torque allocation, momentum management, actuator behavior, and closed-loop performance.

---

## Problem Statement

Develop a hybrid attitude control system capable of maintaining high pointing accuracy for an Earth Observation satellite while performing agile attitude maneuvers. The controller must coordinate multiple actuator types, manage actuator saturation, perform momentum dumping, and maintain stable operation under realistic disturbance torques and actuator failure scenarios.

---

## Objectives

- Maintain high-precision nadir pointing.
- Perform agile attitude slew maneuvers.
- Coordinate torque generation between Reaction Wheels and VSCMGs.
- Manage reaction wheel momentum using magnetorquers.
- Handle reaction wheel saturation.
- Evaluate controller performance during actuator failures.
- Compare different attitude control strategies.

---

## Features

- 3-DOF spacecraft rotational dynamics
- Quaternion-based attitude representation
- Pyramidal Reaction Wheel configuration
- Variable-Speed Control Moment Gyroscope (VSCMG) model
- Magnetorquer-based momentum dumping
- Environmental disturbance torques
- Modular attitude controller architecture
- Torque allocation algorithms
- Automatic visualization and performance metrics

---

## Project Structure

```text
adcs_hybrid_rw_vscmg/
├── README.md
├── requirements.txt
├── main.py
├── config.py
│
├── models/
│   ├── spacecraft.py
│   ├── actuators/
│   │   ├── rw_pyramid.py
│   │   ├── vscmg.py
│   │   └── magnetorquer.py
│   └── environment/
│       ├── orbit.py
│       ├── gravity_gradient.py
│       ├── magnetic_field.py
│       └── disturbances.py
│
├── controllers/
│   ├── attitude_controller.py
│   ├── torque_allocator.py
│   └── magnetic_dumping.py
│
├── simulation/
│   ├── integrator.py
│   ├── simulator.py
│   └── post_process.py
│
├── scenarios/
│   ├── scenario_a_slew.py
│   ├── scenario_b_singularity.py
│   ├── scenario_c_nadir_5orb.py
│   └── scenario_d_robustness.py
│
├── utils/
│   ├── quaternion.py
│   ├── kinematics.py
│   ├── visualization.py
│   └── metrics.py
│
├── data/results/
├── plots/
└── report/
```

---

## Simulation Scenarios

The project evaluates controller performance under several representative mission scenarios.

- Nadir Pointing
- Agile Slew Maneuver
- Reaction Wheel Saturation
- Momentum Dumping
- Reaction Wheel Failure
- VSCMG Singularity
- External Disturbance Rejection

---

## Controllers

The software architecture allows multiple control algorithms to be implemented and compared.

- PD Controller
- LQR Controller
- H∞ Controller *(planned)*
- Model Predictive Control *(planned)*

---

## Actuator Models

### Reaction Wheels

- Pyramidal 4-wheel configuration
- Torque limits
- Momentum limits
- Saturation

### Variable-Speed Control Moment Gyroscopes

- Gimbal dynamics
- Steering law
- Singularity analysis

### Magnetorquers

- Magnetic dipole generation
- Momentum dumping
- Wheel desaturation

---

## Disturbance Models

The spacecraft is subjected to representative environmental disturbances.

- Gravity Gradient Torque
- Residual Magnetic Torque
- Constant External Disturbance Torque
- Atmospheric Drag *(optional)*
- Solar Radiation Pressure *(optional)*

---

## Performance Metrics

Controller performance is evaluated using:

- Pointing Error
- RMS Attitude Error
- Maximum Attitude Error
- Settling Time
- Overshoot
- Control Torque
- Wheel Momentum
- Momentum Dumping Efficiency
- Actuator Utilization

---

## Future Improvements

- Multiplicative Extended Kalman Filter (MEKF)
- Unscented Kalman Filter (UKF)
- Star Tracker and IMU models
- Flexible spacecraft dynamics
- Fault Detection, Isolation and Recovery (FDIR)
- Monte Carlo analysis
- Hardware-in-the-Loop (HIL) testing

---

## References

1. Markley, F. L., & Crassidis, J. L. *Fundamentals of Spacecraft Attitude Determination and Control*
2. Wie, B. *Space Vehicle Dynamics and Control*
3. Wertz, J. R. *Spacecraft Attitude Determination and Control*
4. ECSS-E-ST-60-30C – Space Engineering: Attitude and Orbit Control Systems

---

## License

This project is intended for educational, research, and portfolio purposes to demonstrate the design and evaluation of modern spacecraft attitude control systems.

# Verification & Validation

The framework follows a modular verification approach with comprehensive unit testing for each subsystem.

## Verification Status

| Module | Status | Description |
|--------|:------:|-------------|
| Spacecraft Dynamics | ✅ | Rigid-body rotational dynamics, quaternion kinematics, state propagation |
| Orbit Model | ✅ | Circular orbit propagation in ECI frame |
| Earth's Magnetic Field | ✅ | Dipole magnetic field model |
| Gravity Gradient Disturbance | ✅ | Gravity gradient torque computation |
| Atmospheric Drag | ✅ | Aerodynamic disturbance torque model |
| Solar Radiation Pressure | ✅ | SRP disturbance torque model |
| Disturbance Manager | ✅ | Disturbance aggregation and management |

## Test Summary

- **101 automated unit tests**
- **101 tests passed**
- **0 failures**
- Validation of nominal, boundary, and error conditions
- Numerical stability verification
- Physics consistency checks
- Input validation and exception handling

```
================== test session starts ==================
collected 101 items

================== 101 passed in 0.36 s ==================
```

The project is developed using **pytest** with a strong emphasis on verification and maintainability. Every implemented subsystem is accompanied by dedicated unit tests to ensure correctness and support future development.