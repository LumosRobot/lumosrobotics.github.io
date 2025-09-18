---
layout: default
title: Models
nav_enabled: true
nav_order: 2
---

# Robots
This chapter introduces the robot platforms currently supported in Lumos RL Workspace.Lumos RL Workspace currently supports two humanoid robots: Lus2 and Nix1.
Both robots are described using modular configuration files and URDF/XML models, including detailed kinematics, joint limits, actuator properties, and mass distributions to ensure smooth simulation and accurate sim-to-real transfer. These resources are stored under the robot_models/ directory.
- Currently available models
  - Lus2: Full-sized humanoid robot
  - NIX1: Small-sized humanoid robot
Both robots are modeled with high-fidelity physics, including joint limits, actuator properties, and accurate mass distribution. This ensures realistic training dynamics and smoother transfer to real hardware.

## Device details

### Lus2

- Height:  1.6 m
- Weight:  57 kg
- Degrees of Freedom (DoF): 28
- Actuators: Position–torque actuators with compliant control

<div align="center">
  <img src="../assets/figures/lus2.png" alt="Lus2" width="600"/>
  <p><b>Figure : Lus2 Joint Limits (in Radians, left side example)</b></p>
</div>

<div align="center">
  <img src="../assets/figures/lus2_urdf.png" alt="Lus2 urdf" width="600"/>
  <p><b>Figure : Lus2  skeleton tree</b></p>
</div>



### Nix1

- Height: 0.886m
- Weight: 18 kg
- Degrees of Freedom (DoF): 21
- Actuators: Hybrid position–torque actuators with higher torque limits

<div align="center">
  <img src="../assets/figures/nix_structure.png" alt="nix_structure" width="600"/>
  <p><b>Figure : Nix1 Overall Dimension Diagram</b></p>
</div>

<div align="center">
  <img src="../assets/figures/nix_urdf.png" alt="nix urdf" width="600"/>
  <p><b>Figure : Nix1 skeleton tree</b></p>
</div>