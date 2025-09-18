---
layout: default
title: User Modification Guide
nav_enabled: true
nav_order: 7
---

# 1. Environment Registration Flow
st_gym's environment registration follows the following process：
<div align="center">
  <img src="../assets/figures/Environment Registration Flow.PNG" alt="Registration" width="800"/>
</div>

# 2. Train Workflow
<div align="center">
  <img src="../assets/figures/Train Workflow.PNG" alt="Train Workflow" width="900"/>
</div>

# 3. Robot-Specific Configuration

<div align="center">
  <img src="../assets/figures/ Robot-Specific_Configuration_table.png" alt="Robot-Specific_Configuration_table" width="900"/>
</div>


## 3.1 Robot Model
Load model from:
``` python
# /st_gym/exts/legged_robots/legged_robots/assets/lumos.py
usd_dir_path = os.path.join(BASE_DIR, "../../../../../robot_models/")

Lus2_Joint27_CFG.spawn.usd_path = f"{usd_dir_path}/lus2/usd/lus2_joint27.usd"

Lus2_Joint21_CFG.spawn.usd_path = f"{usd_dir_path}/lus2/usd/lus2_joint21.usd"
```

## 3.2 Reward Weights & Command Ranges
For detailed parameters, refer to RL Env Code Configuration and modify them according to different tasks.

## 3.3 Action Scale:
Default: scale=0.25 
``` python
# st_gym/exts/legged_robots/legged_robots/tasks/config/lus2/rough_env_cfg.py
class ActionsCfg:
 """Action specifications for the MDP."""
     joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)
```

## 3.4. Body Part Names
Modify joint name information according to the robot's xml or urdf：
``` python
# st_gym/exts/legged_robots/legged_robots/tasks/config/lus2/amp_mimic_cfg.py
# 27 joints
lus2_27joint_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
'left_wrist_roll_joint', 'right_wrist_roll_joint'
]

# 21 joints
lus2_21joint_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint'
]

all_joint_names = lus2_21joint_names if using_21_joint else lus2_27joint_names
``` 
## 3.5 motion_files
Select the motion file you want to train and replace the path below：

``` python
# st_gym/exts/legged_robots/legged_robots/tasks/config/lus2/amp_mimic_cfg.py
using_21_joint = True
# 27 joints
if not using_21_joint:
  motion_files=glob.glob(os.getenv("HOME")+"/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/CMU_CMU_07_07*_fps*.pkl")
else:
# 21 joints
  motion_files=glob.glob(os.getenv("HOME")+"/workspace/lumos_ws/humanoid_demo_retarget/sources/data/motions/lus2_joint21/pkl/CMU_CMU_07_07*.pkl")
```

# 4. Troubleshooting
## 4.1 Out of memory

``` bash
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB. GPU 0 has a total capacity of 7.63 GiB of which 104.88 MiB is free. Including non-PyTorch memory, this process has 6.67 GiB memory in use. Of the allocated memory 1.85 GiB is allocated by PyTorch, and 133.34 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```
Reduce the number of parallel environments：
``` python
# lumos_ws/st_gym/exts/legged_robots/legged_robots/tasks/config/lus2/agents/st_rl_ppo_cfg.py
num_envs = 1000#4096
```