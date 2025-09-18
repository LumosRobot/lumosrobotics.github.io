---
layout: default
title: Data retarget
nav_enabled: true
nav_order: 4
---



This guide will walk you through the process of retargeting human motion capture data  to a humanoid robot model.

# Chapter 1. Environment Setup

## 1.1 Create a Conda Environment

```Python
# Create environment
conda create -n retarget python=3.8

# Activate environment
conda activate retarget
```

## 1.2 Install dependencies

```Python
# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# Install GLFW
conda install -c conda-forge glfw

# Install Python requirements
pip install -r requirements.txt
```

# Chapter 2. Dataset and Configuration

All dataset preparation and configuration should be done inside the **`sources/`** directory .

## 2.1 Prepare SMPL Models

Download the[SMPL](https://smpl.is.tue.mpg.de/)  **v1.1.0** model parameters and place them in `data/smpl/`. Rename the original files as follows:

- `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` → `SMPL_NEUTRAL.pkl`
- `basicmodel_m_lbs_10_207_0_v1.1.0.pkl` → `SMPL_MALE.pkl`
- `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` → `SMPL_FEMALE.pkl`

Expected directory:

```Python
|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
```

## 2.2 Prepare the AMASS Dataset

Download the [AMASS](https://amass.is.tue.mpg.de/) **dataset**.

- Our pipeline uses `SMPL+H` **format** (hand motions are ignored).
- Common subsets: `AMASS_CMU, KIT, Eyes_Japan_Dataset, HUMAN4D, ACCAD, HumanEva, SFU, DanceDB`.
- You may include more subsets for broader coverage.

Decompress the datasets into `data/AMASS/`.

## 2.3 Directory structure

The `sources/data/` directory should look like this:

```Bash
sources/data/
    ├── AMASS/       # Motion datasets (AMASS/CMU/Taichi, etc.)
    ├── assets/      # Robot assets, meshes, URDF, and MJCF files
    ├── cfg/         # Configuration files (YAML)
    ├── motions/     # Pre-processed motion clips (npy/npz/pkl format)
    └── smpl/        # SMPL body models
           ├── SMPL_FEMALE.pkl
           ├── SMPL_NEUTRAL.pkl
           └── SMPL_MALE.pkl
```

**Explanation of folders:**

- **smpl/** → SMPL model parameters (`.pkl` files for male, female, neutral).
- **AMASS/** → Source motion datasets (e.g., CMU, ACCAD).
- **motions/** →Pre-processed and retargeted motion clips.
- **cfg/** → Robot-specific configuration files (`.yaml`).
- **assets/** → Visualization and robot resources (meshes, textures, URDF/MJCF).

# Chapter 3. Running Demos

This chapter explains how to run motion retargeting demos in **two ways**:

1. **Manual step-by-step execution** – call each Python script separately.
2. **Automated execution with** **`run.sh`** – run the entire pipeline in one command.

Before starting, navigate to the project root directory:

```Python
cd sources/
```

## 3.1 Manual Step-by-Step Execution

### Step One: Fit SMPL Shape to Robot Joints

```Bash
python fit_smpl_shape.py robot=lumos_lus2_joint27_fitting
```

Output:

- `beta` parameters (shape coefficients)
- `scale` factor
- **Saved at:** `data/motions/lus2_joint27/fit_shape/shape_optimized_v1.pkl`

Purpose: Align SMPL body shape with Lus2 joint configuration.

### Step Two: Motion Retargeting from AMASS

```Bash
python fit_smpl_motion.py robot=lumos_lus2_joint27_fitting +motion_name=CMU_CMU_13_13_21_poses
```

Input motion: `data/AMASS/CMU/13/13_21_poses.npz`

Output:

- Retargeted motion files (`.pkl`, `.npz`) in `data/motions/lus2_joint27/fit_motion/`

Purpose: Map human trajectories from AMASS onto Lus2 robot joints.

The motion file corresponds to:

```Bash
humanoid_demo_retarget/data/AMASS/CMU/
└── CMU
    └── 13
        └── 13_21_poses.npz
```

**Note**: Replace the motion name with your desired AMASS file.

### Step Three: Visualization

```Bash
python vis_data/vis_mj.py robot=lumos_lus2_joint27_fitting +motion_name=CMU_CMU_13_13_21_poses
```

This will generate two `.pkl` files:

- `fit_motion/...pkl`
- `pkl/...pkl` (used in downstream RL projects such as `st_gym`)

The `.pkl` file contains motion fields required by `st_gym`.

## 3.2 Automated Execution (run.sh)

Instead of running multiple Python scripts manually, Lumos RL Workspace provides a **unified automation script.**

### Common Workflows

- **Shape Fitting**

```Python
./run.sh -r lumos_lus2_joint27 -s
```

**Underlying script**: `fit_smpl_shape.py`

**Purpose**: Fits SMPL body shape parameters to the specified robot skeleton.

- **Motion Retargeting**

```Python
./run.sh -r lumos_lus2_joint21 -f -m CMU_CMU_13_13_21_poses
```

**Underlying script**: `fit_smpl_motion.py`

**Important**: The `-m` flag must specify a motion file name; otherwise, the script will not run.

- **Visualization**

```Python
./run.sh -r lumos_lus2_joint27 -v -m CMU_CMU_13_13_21_poses
```

**Underlying script**: `vis_data/vis_mj.py`

- **Full Pipeline (Recommended)**

```Python
./run.sh -r lumos_lus2_joint27 -s -f -v -m CMU_CMU_13_13_21_poses
```

Executes the full pipeline in sequence: **Shape Fitting → Motion Retargeting → Visualization**

This one-line command is recommended for running the entire process end-to-end.

### Parameters

- `-r <robot_id>` : Robot ID (e.g., `lumos_lus2_joint27`, `lumos_lus2_joint21`)
- `-s` : Run shape fitting
- `-f` : Run motion retargeting (**requires** **`-m`**)
- `-m <motion_name>` : Motion file name (e.g., `CMU_CMU_13_13_21_poses`)
- `-v` : Run visualization

Flags can be combined in any order:

```Python
./run.sh -r lumos_lus2_joint27 -sfvm CMU_CMU_13_13_21_poses
./run.sh -r lumos_lus2_joint27 -vsf -m CMU_CMU_13_13_21_poses
```

# Chapter 4. YAML Configuration Files

YAML files (`data/cfg/robot/`) define **retargeting setup**: how SMPL joints map to robot joints.

## 4.1 File Location

Example:

```Bash
data/cfg/robot/lumos_lus2_joint27_fitting.yaml
```

## 4.2 Key Fields

- **`humanoid_type`** – Robot ID (e.g., `lus2_joint27`)
- **`asset`** – Path to URDF/MJCF
- **`extend_config`** – Virtual joints (e.g., Defines additional virtual joints (e.g., a `head_link` added under `pelvis`) to provide more constraints for motion fitting.)
- **`base_link`** – Root of kinematic tree
- **`joint_matches`** – SMPL ↔ robot joint mappings (e.g., `"left_hip_pitch_link"` ↔ `"L_Hip"`).
- **`smpl_pose_modifier`** – Fixed offsets for alignment

## 4.3 Role in the Pipeline

1. **Shape Fitting** – Aligns SMPL and robot skeletons
2. **Motion Retargeting** – Applies mappings & modifiers
3. **Visualization** – Ensures rendered motions match robot

In short: the YAML file is the **contract** between SMPL and robot models.

# Chapter 5.Viewer Shortcuts

Keyboard shortcuts for **MuJoCo viewer** during playback:

| Key / Combo | Description                                  |
| ----------- | -------------------------------------------- |
| R           | Reset playback to the beginning              |
| Space       | Toggle pause/resume                          |
| T           | Switch to the next motion in the loaded list |
| Ctrl + A    | Reset camera view                            |
| F1          | Show help menu                               |
| F5          | Toggle fullscreen mode                       |
| F6          | Toggle world frame visualization             |
| C           | Visualize contact points                     |
| F           | Visualize contact force magnitude            |

# Acknowledgements

This project builds upon the **PHC (Perpetual Humanoid Control)** framework.

If you find this work useful for your research, please consider citing:

```Bash
@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}
```
