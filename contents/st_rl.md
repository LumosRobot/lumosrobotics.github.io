---
layout: default
title: St Rl
nav_enabled: true
nav_order: 6
has_children: true
---

# st_rl
You can follow along using the code available in our [GitHub repository](https://github.com/LumosRobot/st_rl).

---

# Overview of `st_rl`

`st_rl` is a modular reinforcement learning (RL) framework designed to facilitate research and development of advanced RL algorithms, particularly for robotics and simulation environments. The framework emphasizes flexibility, modularity, and scalability, providing well-structured components for training, evaluation, and storage of RL agents. Its design allows for easy integration of different RL algorithms, neural network architectures, and simulation runners.

The main components of `st_rl` are:

1. **Algorithms**
   This module contains implementations of state-of-the-art RL algorithms, such as PPO, TPPO, APPO, and others. It includes the core update rules, policy optimization routines, advantage estimation, and clipping strategies commonly used in modern RL.

2. **Modules**
   This section provides reusable model components, such as actor-critic networks, visual encoders, convolutional networks, and other neural network blocks. The modular design allows flexible assembly of different architectures tailored to specific observation spaces, action spaces, or simulation requirements.

3. **Runners**
   Runners handle the interaction between the agent and the environment. They manage data collection, trajectory storage, batching, and support for both on-policy and off-policy algorithms. Some runners also support staged learning, such as pretraining from demonstration data.

4. **Storage**
   The storage module defines how experience data is recorded, organized, and retrieved for training. It includes structures for rollouts, transitions, and utilities for compressing or managing large datasets efficiently.

Overall, `st_rl` provides a comprehensive framework for reinforcement learning research, combining flexible model components, efficient data handling, and support for modern RL algorithms. Its modular architecture allows researchers and engineers to rapidly prototype, test, and deploy RL agents across a variety of simulated environments.
