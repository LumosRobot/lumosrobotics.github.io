---
title: Quickly Start
nav_enabled: true
nav_order: 3

---

# Installation

## Prerequisites

Make sure your system meets the following requirements:

- **OS**: Ubuntu 20.04 / 22.04
- **Python**: 3.10
- **CUDA**: 12.8
- **Isaac Sim**: 4.5.0
- **Isaac Lab**: 2.1.0


## Step 1. Repository Setup

```Bash
mkdir -p ~/workspace/lumos_ws
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
```

Clone additional repositories:

```Bash
cd ~/workspace/lumos_ws
git clone https://github.com/LumosRobot/st_gym.git
git clone https://github.com/LumosRobot/st_rl.git
git clone https://github.com/LumosRobot/robot_models.git
git clone https://github.com/LumosRobot/humanoid_demo_retarget.git

cd st_gym/third_party
git clone https://github.com/sunzhon/refmotion_manager.git
```

**Project structure**

```Plain
workspace
├── IsaacLab
└── lumos_ws
    ├── humanoid_demo_retarget
    ├── installation
    ├── robot_models
    ├── st_gym
    │   ├── exts/legged_robots
    │   └── third_party/refmotion_manager
    └── st_rl
```

## Step 2. Environment Setup

- Create and activate conda environment:

```Bash
conda create -n lumos_env python=3.10
conda activate lumos_env
```

- Install PyTorch (CUDA build):

```Bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

- Update pip and install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/isaaclab_pip_installation.html):

```Bash
pip install --upgrade pip
pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com
```

- Verify Isaac Sim:

```Bash
isaacsim
```

- Install project dependencies:

```Bash
# legged_robots
cd ~/workspace/lumos_ws/st_gym/exts/legged_robots
pip install -e .

# st_rl
cd ~/workspace/lumos_ws/st_rl
pip install -e .

# refmotion_manager
cd ~/workspace/lumos_ws/st_gym/third_party/refmotion_manager
pip install -e .
```

- System tools:

```Bash
sudo apt-get update
sudo apt install cmake build-essential
```

- Install IsaacLab script:

```Bash
cd ~/workspace/IsaacLab
git checkout 2.1.0
./isaaclab.sh -i
```

- Install rsl_rl:

```Bash
pip install rsl-rl-lib
# or
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```

## Step 3. Robot Resources

- Convert URDF → USD:

```Bash
python ~/workspace/IsaacLab/scripts/tools/convert_urdf.py \
  ~/workspace/lumos_ws/robot_models/lus2/urdf/lus2_joint21.urdf \
  ~/workspace/lumos_ws/robot_models/lus2/usd/lus2_joint21.usd \
  --merge-joints --joint-stiffness 10000 --joint-damping 0.0 --rendering_mode quality
```

# Running

## Training

Flat terrain:

```Bash
cd ~/workspace/lumos_ws/st_gym
python scripts/st_rl/train.py --task Flat-Lus2 --headless
```

Rough terrain:

```Bash
cd ~/workspace/lumos_ws/st_gym
python scripts/st_rl/train.py --task Rough-Lus2 --headless
```

Or use run.sh:

```Bash
./run.sh -m train
```

Logs are saved to:

```Plain
st_gym/logs/st_rl/lus2_flat/yyyy-mm-dd_hh-mm-ss
```

**Run Script (****`run.sh`****) Options**

| Option | Description                                                  |
| ------ | ------------------------------------------------------------ |
| -n     | Specify the task name (default: `Flat-Lus2`)                 |
| -m     | Set run mode to **training** (`train`), **playback** (`play`) or **simulation only** (`sim2mujoco`) |
| -l     | Load a previous run (implies `--load_run xxx` `--resume=True`) |
| -h     | Nohup output file to extract from (default `nohup.out`)      |
| -c     | Load a specific checkpoint by index                          |
| -d     | Use specific device (example: `--device cuda:0`)             |
| -r     | Export the trained model to RKNN format (`--export_rknn`)    |
| -e     | Specify the experiment name (default: `flat-Lus2`)           |

## Play

Run trained policy in Isaac Lab:

```Bash
python scripts/st_rl/play.py --task Flat-Lus2 \
  --load_run 2025-06-05_15-16-48 --checkpoint model_400.pt
```

Or:

```Bash
./run.sh -n Flat-Lus2 -m play -l 2025-06-05_15-16-48 -c model_400.pt
```

Exported ONNX policy:

```Plain
st_gym/logs/st_rl/lus2_flat/yyyy-mm-dd_hh-mm-ss/exported/policy.onnx
```

## Sim2Sim (MuJoCo)

Replay with MuJoCo using python:

```Bash
python scripts/st_rl/sim2mujoco.py --task Flat-Lus2-Play \
--experiment_name lus2_flat --load_run 2025-07-27_15-13-37
```

or by bash script:

```Bash
./run.sh -n Flat-Lus2 -m sim -l 2025-07-27_15-13-37
```
