# ADCS Hybrid RW + VSCMG Controller Design

## Overview

This project implements a complete closed-loop ADCS for a small EO spacecraft equipped with:
- A **pyramidal reaction wheel (RW)** cluster (4 wheels, 45° skew)
- A **variable-speed, variable-axis Control Moment Gyro (VSCMG)** cluster
- **3-axis magnetorquers** for momentum management

The system is designed to:
- Fill the combined momentum envelope during agile maneuvers
- Maintain tight pointing performance during nominal and dumping operations
- Bound stored momentum indefinitely using magnetic desaturation
- Handle singularities in the VSCMG cluster

---

## Project Structure

```bash
adcs_hybrid_rw_vscmg/
├── README.md
├── requirements.txt
├── main.py                          # Single entry point
├── config.py                        # All parameters & gains
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
├── data/results/                    # Simulation outputs (gitignored)
├── plots/                           # Auto-generated figures
└── report/
    └── ADCS_Controller_Report.md    # Final report (source)
