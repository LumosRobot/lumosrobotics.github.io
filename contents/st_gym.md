---
layout: default
title: RL Env Code
nav_enabled: true
nav_order: 5
---

# 1. Key Features:

- **Built on Isaac Sim & Isaac Lab** for high-fidelity robotics simulation.
- **Train & evaluate locomotion policies** for humanoid robots.
- **Supports sim-to-real transfer** via domain randomization.

# 2. Code Structure

The structure of st_gym:

```Python
st_gym
├── exts
│   └── legged_robots
│       ├── legged_robots
│       │   ├── assets/              # Robot models configuration (e.g., lumos.py)
│       │   ├── tasks/               # Task configurations (env, agents, AMP, etc.)
│       │   │   ├── config/          
│       │   │   │   └── lus2/        # lus2 task configuration
│       │   │   ├── mdp/             # RL MDP components (actions, commands, curriculums, events, observations, rewards, terminations)
│       │   │   └── utils/           # Utilities and wrappers (logger, wrappers)
│       │   └── tests/               # Unit tests
│       └── setup.py                 # Module installation entry
│
├── scripts
│   ├── inves_train.py               # inves_training entry script
│   ├── list_envs.py                 # Environment listing tool
│   ├── skrl/                        # skrl algorithm train and play interface
│   └── st_rl/                       # st_rl algorithm interface and utilities
│       ├── conf/                    # Training configuration files
│       ├── train.py                 # Main training script
│       ├── play.py                  # Policy evaluation and replay
│       ├── ros2interface.py         # ROS2 communication interface
│       ├── sim2mujoco.py            # Sim-to-Sim transfer tool (IsaacSim -> MuJoCo)
│       └── sim2sim.py               # General Sim-to-Sim transfer
│
├── third_party
│   └── refmotion_manager/           # Reference motion manager 
│
├── run.sh                           # Quick start script
└── setup.sh                         # Environment initialization script
```

# 3. Configuration Details

Configuration files for different environments and algorithms are located in:

st_gym/exts/legged_robots/tasks/config/

└── lus2/

​    ├── agents/

​    │   └── st_rl_ppo_cfg.py       # PPO training parameters

​    ├── __init__.py                     # register train and play environments

​    ├── flat_env_cfg.py              # Flat configuration

​    ├── rough_env_cfg.py          # Rough configuration

​    └── amp_mimic_cfg.py         # Motion data, AMP and mimic configuration

To modify motion sources or environmental properties, edit the corresponding files above.

Next, we will introduce the parameters in `rough_env_cfg.py` and `amp_mimic_cfg.py` in detail.

## 3.1 InteractiveScene(`rough_env_cfg.py`)

- Objects: terrain, light, sky_light
- Articulation: robot
- Sensors: height_scanner, contact_forces

## 3.2 Event(`rough_env_cfg.py`)

- reset_base: Resets robot base pose & velocity ranges
- reset_robot_joints: Resets joint positions and velocities
- Domain randomization
  - physics_material: Randomizes rigid body material properties (friction, restitution)
  - reset_robot_rigid_body_mass: Randomizes robot rigid body masses
  - reset_robot_base_com: Randomizes center of mass of torso and hip
  - randomize_actuator_gains: Randomizes actuator stiffness & damping gains
  - randomize_joint_parameters: Randomizes joint parameters (friction, armature, limits)
- External disturbance
  - push_robot: Applies random pushes to the robot at intervals
  - Interval: Applies random external forces/torques on the torso

## 3.3 Curriculum(`rough_env_cfg.py`)

| **Curriculum Term**  | **Function / Purpose**                                 | **Weight** | **Num Steps** |
| -------------------- | ------------------------------------------------------ | ---------- | ------------- |
| terrain_levels       | Adjusts terrain difficulty according to robot velocity | N/A        | N/A           |
| alive_rew            | Modifies weight of the “alive” reward term             | 1          | 500           |
| action_rate_l2       | Penalizes high action rate (L2 norm)                   | -0.1       | 500           |
| action_smooothness_2 | Penalizes jerkiness in actions                         | -0.1       | 500           |
| dof_torques_l2       | Penalizes large joint torques (L2 norm)                | -0.000001  | 1000          |
| dof_acc_l2           | Penalizes high joint accelerations (L2 norm)           | -5E-8      | 1000          |
| contact_forces       | Penalizes excessive contact forces                     | -0.0001    | 1000          |

## 3.4 Reward Keys and Weights

Defined in`rough_env_cfg.py`, and the weights that are often modified are defined in `amp_mimic_cfg.py`

```
Task rewards
```

| **Parameter**         | **Description**                                              | **Example** |
| --------------------- | ------------------------------------------------------------ | ----------- |
| termination_penalty   | Penalize termination                                         | -450.0      |
| alive                 | Survival reward                                              | 5.0         |
| track_lin_vel_xy_exp  | Reward for tracking target linear velocity in XY plane       | 2.0         |
| track_ang_vel_z_exp   | Reward for tracking target angular velocity around Z axis    | 2.0         |
| feet_air_time         | Reward for swing foot airtime                                | 0.8         |
| feet_slide            | Penalty for sliding feet                                     | -1.0        |
| joint_deviation_hip   | Penalize deviation from default of the joints that are not essential for locomotion | -0.1        |
| joint_deviation_feet  | -0.1                                                         |             |
| joint_deviation_arms  | -0.1                                                         |             |
| joint_deviation_wrist | -0.2                                                         |             |
| joint_deviation_torso | -0.1                                                         |             |
| energy_cost           | Penalize energy consumption                                  | -2.0e-7     |
| feet_parallel_v1      | Reward for keeping feet parallel                             | 1.0         |
| undesired_contacts    | Penalize undesired body-ground contacts                      | -20.0       |
| feet_stumble          | Penalize stumbling feet                                      | -500.0      |
| action_smooothness_2  | Penalize jerky actions (smoothness regularization)           | -5.0e-2     |
| action_rate_l2        | Penalize high action change rate                             | -5.0e-2     |
| dof_acc_l2            | Penalize large joint accelerations                           | -1e-8       |
| dof_torques_l2        | Penalize large joint torques                                 | -4.0e-6     |
| dof_pos_limits        | Penalize exceeding joint position limits                     | -10.0       |
| dof_vel_limits        | Penalize exceeding joint velocity limits                     | -5.0        |
| dof_torques_limits    | Penalize exceeding torque limits                             | -1.0        |
| contact_forces        | Penalize large contact forces                                | -2.0e-5     |

```
AMP style reward
```

| **Parameter**        | **Description**                        | **Example** |
| -------------------- | -------------------------------------- | ----------- |
| track_style_goal_exp | Reward for tracking style goals in AMP | 1.47        |

```
Mimic tracking rewards
```

| **Parameter**             | **Description**                              | **Example** |
| ------------------------- | -------------------------------------------- | ----------- |
| track_upper_joint_pos_exp | Track upper body joint positions             | 20.4        |
| track_upper_joint_vel_exp | Track upper body joint velocities            | 3.5         |
| track_lower_joint_pos_exp | Track lower body joint positions             | 15.0        |
| track_lower_joint_vel_exp | Track lower body joint velocities            | 2.2         |
| track_feet_joint_pos_exp  | Track feet joint positions                   | 2.0         |
| track_feet_joint_vel_exp  | Track feet joint velocities                  | 1.0         |
| track_link_pos_exp        | Track link positions                         | 0.0         |
| track_link_vel_exp        | Track link velocities                        | 0.0         |
| track_root_pos_exp        | Track root position                          | 2.0         |
| track_root_quat_exp       | Track root quaternion (absolute orientation) | 0.0         |
| track_root_rotation_exp   | Track root rotation                          | 2.0         |
| track_root_lin_vel_exp    | Track root linear velocity                   | 1.0         |
| track_root_ang_vel_exp    | Track root angular velocity                  | 1.0         |

## 3.5 Observation and Action(`rough_env_cfg.py`)

In our environments, the agent receives observations and outputs actions that include:

- Policy Observation Space:
  - Proprioception states
    - Base angle velocities
    - Project gravity
    - Joint positions
    - Joint velocities
    - Last actions
  - Goal states 
    - Velocity commands
    - optional
      - style_goal_commands (If use amp)
      - expressive_goal_commands (If use mimic)
- Action Space:
  - **Joint position control:** joint angle/desired position
- Critic Policy Observation Space:
  - Root states
    - Base position
    - Base orientation
    - Base linear velocity
    - Base angular velocity
    - Projected gravity
  - Joint & action states
    - Joint positions
    - Joint velocities
    - Last actions
  - **Body states** *(for Mimic)*
    - Expressive link positions (body pos)
    - Expressive link velocities (body lin vel)
  - Goal states from the next time frame (future)
    - Velocity commands
    - *Optional*
      - style_goal_commands (If use amp)
      - expressive_goal_commands (If use mimic)
    - Privileged information
      - Masses, contact forces, joint stiffness/damping, friction coeff

## 3.6 Commands(`rough_env_cfg.py`)

| **Command**              | **Type / Class**                   | **Resampling Time (s)** | **Velocity / Heading Ranges**                                | **Notes**                                        |
| ------------------------ | ---------------------------------- | ----------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| base_velocity            | BaseVelocityCommand                | (5.0, 10.0)             | Linear X: (-0.2,0.2), Linear Y: (-0.1,0.1), Angular Z: (-0.1,0.1), Heading: (-π, π) | Debug visualization enabled                      |
| style_goal_commands      | StyleCommand (if using AMP)        | (10.0, 10.0)            | Linear & angular velocities: (-1.0, 1.0), Heading: (-π, π)   | Number of commands = len(style_goal_fields)      |
| expressive_goal_commands | ExpressiveCommand (if using Mimic) | (0.0, 0.0)              | Linear & angular velocities: (-1.0, 1.0), Heading: (-π, π)   | Number of commands = len(expressive_goal_fields) |

## 3.7 Terminations(`rough_env_cfg.py`)

| **Termination Condition**          | **Function**                               | **Parameters / Notes**                                 |
| ---------------------------------- | ------------------------------------------ | ------------------------------------------------------ |
| time_out                           | mdp.time_out                               | Ends episode when max time reached                     |
| base_height                        | mdp.root_height_below_minimum              | Minimum pelvis height = 0.4 m                          |
| bad_orientation                    | mdp.bad_orientation                        | Limit angle = 1.0 rad                                  |
| tracking_lower_dof_error           | st_mdp.tracking_error_adaptive_termination | Monitors lower joint position error (min 0.3, max 1.5) |
| tracking_upper_dof_error           | st_mdp.tracking_error_adaptive_termination | Monitors upper joint position error (min 0.2, max 1.5) |
| (Optional) tracking_root_pos_error | st_mdp.tracking_error_adaptive_termination | Monitors root position error (min 0.4, max 2.0)        |

## 3.8 Environment Parameters

| **Parameter**                      | **Description**                                              | **Example**                                   |
| ---------------------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| `amp_mimic_cfg.py`                 |                                                              |                                               |
| num_envs                           | Number of parallel environments for training                 | 4096                                          |
| using_21_joint                     | Whether to use the 21-joint robot model (otherwise 27 joints) | True                                          |
| motion_files                       | Reference motion files used for imitation learning (AMP / mimic tasks) | dance2_subject4_1871_6771_fps25.pkl           |
| random_start                       | Randomize start the robot                                    | True                                          |
| amp_obs_frame_num                  | Number of consecutive frames for AMP observation (history length) | 2                                             |
| INIT_STATE_FIELDS                  | Initial state variables (root state + joint DOF pos/vel)     | root_pos_x, root_rot_w, …                     |
| style_fields                       | State features used for style tracking reward                | root_rot_w, joint_dof_pos, joint_dof_vel, …   |
| style_goal_fields                  | Target style features for goal tracking (optional, often `None`) | None                                          |
| style_reward_coef                  | Reward coefficient for style tracking                        | 10.0                                          |
| expressive_goal_fields             | Features used in expressive imitation (joint DOF states + link positions/vels) | joint_dof_pos, joint_dof_vel, link_pos_x_b, … |
| ref_motion_cfg.ref_length_s        | Duration of reference trajectory segment (seconds)           | 2.0                                           |
| ref_motion_cfg.time_between_frames | Time interval between motion frames                          | 0.02                                          |
| trajectory_num                     | Number of trajectories sampled per environment               | 4096                                          |
| specify_init_values                | Customized initial posture values (optional)                 | dict of joint positions (stand pose)          |
| episode_length_s                   | Episode duration in seconds                                  | 2                                             |
| `st_rl_ppo_cfg.py`                 |                                                              |                                               |
| save_interval                      | Checkpoint saving frequency                                  | 500                                           |
| max_iterations                     | Maximum training iterations                                  | 20000                                         |
| experiment_name                    | Name of the experiment                                       | lus2_flat                                     |
| algorithm_name                     | Algorithm used                                               | PPO                                           |
| policy_name                        | Policy architecture                                          | ActorCritic                                   |
| runner_name                        | Runner type                                                  | AmpPolicyRunner                               |

## 3.9 PPO Parameters(`st_rl_ppo_cfg.py`)

| **Parameter**       | **Description**                        | **Example** |
| ------------------- | -------------------------------------- | ----------- |
| clip_param          | Clipping range for PPO ratio           | 0.2         |
| entropy_coef        | Coefficient for entropy regularization | 0.01        |
| value_loss_coef     | Weight of value function loss          | 1.0         |
| num_learning_epochs | Number of learning epochs per update   | 5           |
| num_mini_batches    | Mini-batch splits per epoch            | 4           |
| learning_rate       | Policy optimization learning rate      | 1e-3        |
| gamma               | Discount factor                        | 0.99        |
| lam                 | GAE lambda                             | 0.95        |
| desired_kl          | Target KL divergence                   | 0.01        |
| max_grad_norm       | Maximum gradient clipping              | 1.0         |

## 3.10 Policy Parameters(`st_rl_ppo_cfg.py`)

| **Parameter**                      | **Description**                                  | **Example**     |
| ---------------------------------- | ------------------------------------------------ | --------------- |
| actor_hidden_dims                  | Hidden layers for actor network                  | [512, 256, 128] |
| critic_hidden_dims                 | Hidden layers for critic network                 | [512, 256, 128] |
| activation                         | Nonlinear activation function                    | ELU             |
| rnn_type                           | Recurrent module type (for temporal correlation) | LSTM            |
| init_noise_std(`amp_mimic_cfg.py`) | Initial exploration noise standard deviation     | 1.2             |
