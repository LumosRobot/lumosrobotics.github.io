---
layout: default
title: RL Algorithm Code
nav_enabled: true
nav_order: 6
---

### Algorithms

#### ppo.py

#####  General Overview

This file implements **PPO (Proximal Policy Optimization)**, a widely used reinforcement learning algorithm. It defines the **PPO class** which manages training with:

- A policy/value network (actor_critic).
- Experience storage (`RolloutStorage`).
- The PPO update procedure (surrogate loss + value loss + entropy bonus).

This is the **foundation**: later, `APPO` extends this base PPO by adding adversarial imitation learning.



##### Class Breakdown

1. ###### Imports and class definition

```Python
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from st_rl.modules import ActorCritic
from st_rl.modules import EmpiricalNormalization
from st_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from st_rl.modules import ActorCritic
from st_rl.modules import EmpiricalNormalization
from st_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
```

- Imports PyTorch modules and custom classes.
- `ActorCritic`: a neural network that contains both the policy (actor) and value function (critic).
- `EmpiricalNormalization`: for normalizing observations.
- `RolloutStorage`: handles storing trajectories.
- Defines the `PPO` class, which will implement the PPO algorithm.

1. ###### Initialization (__init__)

```Python
def __init__(self,
             actor_critic,
             num_learning_epochs=1,
             num_mini_batches=1,
             clip_param=0.2,
             gamma=0.998,
             lam=0.95,
             value_loss_coef=1.0,
             entropy_coef=0.0,
             learning_rate=1e-3,
             max_grad_norm=1.0,
             use_clipped_value_loss=True,
             clip_min_std=1e-15,
             optimizer_class_name="Adam",
             schedule="fixed",
             desired_kl=0.01,
             device='cpu',
             empirical_normalization=False,
             *args,
             **kwargs):
```

- Initializes the PPO object with important hyperparameters:
  - `clip_param`: PPO clipping factor (controls update size).
  - `gamma`: discount factor for rewards.
  - `lam`: lambda for GAE (Generalized Advantage Estimation).
  - `value_loss_coef`: weight for value function loss.
  - `entropy_coef`: weight for entropy (encourages exploration).
  - `learning_rate`: optimizer step size.
  - `max_grad_norm`: gradient clipping to stabilize training.
  - `schedule`, `desired_kl`: optional adaptive learning rate control.
  - `empirical_normalization`: whether to normalize observations.
- Sets up optimizer and stores other config variables.

1. ###### Observation normalization

```Python
def init_obs_norm(self, num_obs, num_critic_obs):
    if self.empirical_normalization:
        self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
        self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
    else:
        self.obs_normalizer = torch.nn.Identity().to(self.device)
        self.critic_obs_normalizer = torch.nn.Identity().to(self.device)
        raise (f"[ppo] Do not use normalization for obs and critic_obs")
```

- Set up normalizers for both actor and critic observations.
- If `empirical_normalization=True`: normalize with running statistics.
- Otherwise: just identity (no normalization).

1. ###### Rollout storage

```Python
def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, **kwargs):
    self.transition = RolloutStorage.Transition()
    self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device, **kwargs)
```

- Initializes `RolloutStorage` to hold collected trajectories.
- Each transition contains: observations, actions, rewards, values, etc.

1. ###### Acting (act)

```Python
def act(self, obs, critic_obs):
    if self.actor_critic.is_recurrent:
        self.transition.hidden_states = self.actor_critic.get_hidden_states()
    # Compute the actions and values
    self.transition.actions = self.actor_critic.act(obs).detach()
    self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()
    # need to record obs and critic_obs before env.step()
    self.transition.observations = obs
    self.transition.critic_observations = critic_obs

    return self.transition.actions
```

- If the actor-critic is recurrent (e.g., RNN/LSTM), it saves hidden states.
- Computes the current action, value estimate, and log-probability of the action.
- Stores additional policy distribution info: mean and standard deviation.
- Saves both actor observations and critic observations for training later.
- Return the chosen action to be executed in the environment.

1. ###### Processing environment step (process_env_step)

```Python
def process_env_step(self, rewards, dones, infos):
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    # Bootstrapping on time outs
    if 'time_outs' in infos:
        self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

    # Record the transition
    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.actor_critic.reset(dones)
```

- Stores rewards and done flags into the transition.
- If an episode ends due to a timeout, it adds a bootstrap value to the reward (so the agent still learns from that state).
- Adds the current transition to rollout storage.
- Clears transition for the next step.
- Resets the actor-critic if episodes ended.

1. ###### Compute returns (compute_returns)

```Python
def compute_returns(self, last_critic_obs):
    last_values = self.actor_critic.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)
```

- At the end of a rollout, computes the discounted returns and advantages.
- Uses the critic’s value for the last observation as a bootstrap.
- Calls `compute_returns` in `RolloutStorage`, which usually implements GAE (Generalized Advantage Estimation).

1. ###### Update loop (update)

```Python
def update(self, current_learning_iteration):
    self.current_learning_iteration = current_learning_iteration
    mean_losses = defaultdict(lambda :0.)
    average_stats = defaultdict(lambda :0.)
    if self.actor_critic.is_recurrent:
        generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    else:
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for minibatch in generator:

        losses, _, stats = self.compute_losses(minibatch)

        loss = 0.
        for k, v in losses.items():
            loss += getattr(self, k + "_coef", 1.) * v
            mean_losses[k] = mean_losses[k] + v.detach()
        mean_losses["total_loss"] = mean_losses["total_loss"] + loss.detach()
        for k, v in stats.items():
            average_stats[k] = average_stats[k] + v.detach()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    for k in mean_losses.keys():
        mean_losses[k] = mean_losses[k] / num_updates
    for k in average_stats.keys():
        average_stats[k] = average_stats[k] / num_updates
    self.storage.clear()
    if hasattr(self.actor_critic, "clip_std"):
        self.actor_critic.clip_std(min= self.clip_min_std)

    return mean_losses, average_stats
```

- Main PPO update step:
  - Iterates through minibatches of stored rollouts.
  - Calls `compute_losses` to get loss terms and stats.
  - Accumulates and averages the results.
  - Performs gradient descent with gradient clipping.
  - Clears storage afterward.
- Returns average losses and stats for logging.

1. ###### Compute losses (compute_losses)

```Python
def compute_losses(self, minibatch):
    self.actor_critic.act(minibatch.obs, masks=minibatch.masks, 
                            hidden_states= minibatch.hidden_states.actor if minibatch.hidden_states is not None else None)
    actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
    value_batch = self.actor_critic.evaluate(minibatch.critic_obs, masks=minibatch.masks, 
                                             hidden_states=minibatch.hidden_states.critic if minibatch.hidden_states is not None else None)
    mu_batch = self.actor_critic.action_mean
    sigma_batch = self.actor_critic.action_std
    try:
        entropy_batch = self.actor_critic.entropy
    except:
        entropy_batch = None
```

- Recomputes actions/log-probabilities and values for the minibatch (needed for PPO loss).
- `mu_batch`, `sigma_batch`: mean and std of the policy distribution.
- `entropy_batch`: entropy of policy distribution, used as an exploration bonus.

```Python
    # KL
    if self.desired_kl != None and self.schedule == 'adaptive':
        with torch.inference_mode():
            kl = torch.sum(torch.log(sigma_batch / minibatch.old_sigma + 1.e-5) + 
                           (torch.square(minibatch.old_sigma) + torch.square(minibatch.old_mu - mu_batch)) / 
                           (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
            kl_mean = torch.mean(kl)

            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
```

- Computes KL divergence between old and new policy distributions.
- If `schedule="adaptive"`, it adjusts the learning rate dynamically:
  - KL too large → lower learning rate.
  - KL too small → increase learning rate.
- Helps keep policy updates stable.

```Python
    # KL
    if self.desired_kl != None and self.schedule == 'adaptive':
        with torch.inference_mode():
            kl = torch.sum(torch.log(sigma_batch / minibatch.old_sigma + 1.e-5) + 
                           (torch.square(minibatch.old_sigma) + torch.square(minibatch.old_mu - mu_batch)) / 
                           (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
            kl_mean = torch.mean(kl)

            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
```

- Computes KL divergence between old and new policy distributions.
- If `schedule="adaptive"`, it adjusts the learning rate dynamically:
  - KL too large → lower learning rate.
  - KL too small → increase learning rate.
- Helps keep policy updates stable.

```Python
    # Surrogate loss
    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(minibatch.old_actions_log_prob))
    surrogate = -torch.squeeze(minibatch.advantages) * ratio
    surrogate_clipped = -torch.squeeze(minibatch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

- PPO’s clipped surrogate objective:
  - ratio: probability ratio between new and old policy.
  - surrogate: advantage × ratio (policy gradient term).
  - surrogate_clipped: same, but ratio is clipped.
  - Final loss = maximum of unclipped vs clipped loss (conservative update).

```Python
    # Value function loss
    if self.use_clipped_value_loss:
        value_clipped = minibatch.values + (value_batch - minibatch.values).clamp(-self.clip_param, self.clip_param)
        value_losses = (value_batch - minibatch.returns).pow(2)
        value_losses_clipped = (value_clipped - minibatch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
    else:
        value_loss = (minibatch.returns - value_batch).pow(2).mean()
```

- Value loss: mean squared error between predicted values and returns.
- Optionally uses clipped value loss to prevent large shifts in value function predictions.

```Python
    return_ = dict(
        surrogate_loss= surrogate_loss,
        value_loss= value_loss,
    )
    if entropy_batch is not None:
        return_["entropy"] = - entropy_batch.mean()
    
    inter_vars = dict(
        ratio= ratio,
        surrogate= surrogate,
        surrogate_clipped= surrogate_clipped,
    )
    if self.desired_kl != None and self.schedule == 'adaptive':
        inter_vars["kl"] = kl
    if self.use_clipped_value_loss:
        inter_vars["value_clipped"] = value_clipped
    return return_, inter_vars, dict()
```

Returns:

- `surrogate_loss`, `value_loss`, and optionally entropy (as a negative term so lower = more entropy).
- Intermediate variables (ratios, clipped values, etc.) for analysis/debugging.

1. ###### Save and load state

```Python
def state_dict(self):
    state_dict = {
        "model_state_dict": self.actor_critic.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
    }
    if hasattr(self, "lr_scheduler"):
        state_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

    # -- Save observation normalizer if used
    if self.empirical_normalization:
        state_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        state_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
    
    return state_dict
```

- Saves model weights, optimizer state, scheduler, and normalizers (if used).
- Used for checkpointing.

```Python
def load_state_dict(self, state_dict):
    resumed_trianing = self.actor_critic.load_state_dict(state_dict["model_state_dict"])

    # -- Load RND model if used
    if self.rnd:
        self.alg.rnd.load_state_dict(state_dict["rnd_state_dict"])

    if "optimizer_state_dict" in state_dict:
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    if "obs_norm_state_dict" in state_dict:
        self.obs_normalizer.load_state_dict(state_dict["obs_norm_state_dict"])

    if "critic_obs_norm_state_dict" in state_dict:
        self.critic_obs_normalizer.load_state_dict(state_dict["critic_obs_norm_state_dict"])

    if hasattr(self, "lr_scheduler"):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])
    elif "lr_scheduler_state_dict" in state_dict:
        print("Warning: lr scheduler state dict loaded but no lr scheduler is initialized. Ignored.")
```

- Loads saved weights into the model, optimizer, normalizers, and scheduler.
- Ensures training can resume from saved checkpoints.
- Prints a warning if scheduler state exists but no scheduler is initialized.

#### 5.2.3.2 appo.py

#####  General Overview

**This file implements APPO (Adversarial Proximal Policy Optimization), an extension of PPO for adversarial imitation learning.** It defines the APPO-related classes which manage training with:

- A base PPO algorithm (for stable policy optimization).
- Additional rollout storage (`AmpRolloutStorage`) that includes expert reference motions.
- A discriminator network that distinguishes expert motions from policy-generated motions.
- Extra loss terms (mimic loss, discriminator loss) to encourage the policy to imitate expert data.
- Style rewards combined with task rewards to balance imitation and task completion.

**This extends PPO**: while PPO only optimizes the policy using environment rewards and clipped updates, APPO adds **adversarial motion priors** so the agent learns both to solve the task and to move like an expert.

**APPO Basics (Reminder)**

- **Goal**: Train a policy that not only maximizes task reward but also imitates expert motion style.
- **Challenge**: Pure PPO may solve the task but produce unnatural movements.
- **Solution (APPO)**: Introduce a discriminator + imitation losses, giving the agent additional learning signals from expert demonstrations.

#####  Utility Functions

```Python
def GET_PROB_FUNC(option, iteration_scale):
    PROB_options = {
        "linear": (lambda x: max(0, 1 - 1 / iteration_scale * x)),
        "exp": (lambda x: max(0, (1 - 1 / iteration_scale) ** x)),
        "tanh": (
            lambda x: max(
                0, 0.5 * (1 - torch.tanh(1 / iteration_scale * (x - iteration_scale)))
            )
        ),
    }
    return PROB_options[option]
```

- Provides a way to generate a probability schedule function.
- Options:
  - `"linear"` → decreases linearly.
  - `"exp"` → decreases exponentially.
  - `"tanh"` → decreases smoothly following a tanh curve.
- Typically used to **anneal imitation probability or weighting** during training.

##### Class Breakdown

1. ###### Imports and utility function

```Python
 import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import st_rl.modules as modules
from st_rl.utils import unpad_trajectories
from st_rl.storage.rollout_storage import ActionLabelRollout, AmpRolloutStorage
from st_rl.algorithms.ppo import PPO
from st_rl.algorithms.tppo import TPPO
from st_rl.datasets.motion_loader import AMPLoader
from st_rl.modules.normalizer import Normalizer
from st_rl.modules import EmpiricalNormalization
```

- Imports PyTorch modules and several project-specific modules.
- `PPO` and `TPPO`: base algorithms (this file extends them).
- `AmpRolloutStorage`: rollout storage tailored for AMP (Adversarial Motion Priors).
- `EmpiricalNormalization`: normalization tool.
- `AMPLoader`: loads expert motion data for adversarial imitation.

1. ###### `APPOAlgoMixin` (Mixin for adversarial imitation) 

```Python
class APPOAlgoMixin:
    def __init__(
        self,
        *args,
        discriminator_loss_coef=1.0,
        mimic_loss_coef=0.0,
        amp_obs_dim=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.discriminator_loss_coef = discriminator_loss_coef
        self.mimic_loss_coef = mimic_loss_coef
        self.ref_obs_normalizer = EmpiricalNormalization(shape=[amp_obs_dim], until=1.0e8).to(self.device)
        self.ref_obs_normalizer.train()
        self.ref_motion_normalizer = EmpiricalNormalization(shape=[amp_obs_dim], until=1.0e8).to(self.device)
        self.ref_motion_normalizer.train()
```

- `APPOAlgoMixin` is a **mixin**: it’s meant to be combined with PPO/TPPO.
- Adds adversarial imitation learning features:
  - `discriminator_loss_coef`: weight for discriminator loss.
  - `mimic_loss_coef`: weight for imitation loss.
  - `ref_obs_normalizer` / `ref_motion_normalizer`: normalize expert reference states and motions.

1. ###### Storage initialization

```Python
def init_storage(self, *args, **kwargs):
    self.transition = AmpRolloutStorage.Transition()
    self.storage = AmpRolloutStorage(*args, **kwargs)
```

- Initializes storage for AMP rollouts.
- `AmpRolloutStorage` is similar to normal rollout storage, but it also stores reference motions/observations for adversarial training.

1. ###### Acting (`act` override)

```Python
def act(self, obs, critic_obs):
    return_ = super().act(obs, critic_obs)  # return is transition.actions
    self.transition.action_labels = return_
    return return_
```

- Calls the parent act function (from PPO/TPPO).
- Saves the chosen actions as action_labels (used later for supervised losses like imitation)
- Return the actions to execute.

1. ###### Processing environment step

```Python
def process_env_step(self, rewards, dones, infos):
    assert "observations" in infos and "ref_obs" in infos["observations"], "Missing ref_obs in infos"
    ref_obs = infos["observations"]["ref_obs"]
    B = ref_obs.shape[0]
    ref_obs = ref_obs.reshape(B, -1).to(self.device)
    ref_obs = self.ref_obs_normalizer(ref_obs)

    ref_motion = infos["ref_motion"]
    ref_motion = self.ref_motion_normalizer(ref_motion)

    self.transition.ref_obs = ref_obs
    self.transition.ref_motion = ref_motion

    style_reward, task_reward, rewards = self.actor_critic.discriminator.predict_style_reward(ref_obs, rewards)
    infos["log"]["Episode_Reward/style_reward"] = style_reward.mean().item()
    infos["log"]["Episode_Reward/task_reward"] = task_reward.mean().item()

    return_ = super().process_env_step(rewards, dones, infos)
    return return_
```

- Extracts reference observations (ref_obs) and motions (ref_motion) from the environment info.
- Normalizes them and stores in the transition.
- Uses the discriminator to compute:
  - style_reward: how much the agent matches expert style.
  - task_reward: normal task reward (environment reward).
- Logs both rewards separately.
- Calls  parent process_env_step to finish storing transition.

1. ###### Compute losses (compute_losses)

```Python
def compute_losses(self, minibatch):
    losses, inter_vars, stats = super().compute_losses(minibatch)
```

- First, calls the parent class (`PPO` or `TPPO`) `compute_losses`.
- This gives the standard PPO losses: surrogate loss, value loss, entropy.

```Python
if hasattr(self, "dagger_loss_coef"):
    if self.dagger_loss_coef != 0:
        dagger_loss = torch.norm(
            self.actor_critic.action_mean - minibatch.action_labels, dim=-1
        ).mean()
        losses["dagger_loss"] = dagger_loss
```

- If DAgger (Dataset Aggregation) is enabled:
  - Computes L2 loss between policy's mean action and the stored action_labels.
  - Add this as dagger_loss.
- Used for imitation learning with expert supervision.

```Python
if hasattr(self, "mimic_loss_coef"):
    if self.mimic_loss_coef != 0:
        policy_state = minibatch.ref_obs
        expert_state = minibatch.ref_motion
        mimic_loss = torch.norm(
            policy_state - expert_state, dim=-1
        ).mean()
        losses["mimic_loss"] = mimic_loss
```

- Computes **mimic loss**: L2 distance between policy’s reference obs and expert reference motion.
- Encourages the agent to mimic the expert states more closely.

```Python
if hasattr(self.actor_critic, "discriminator"):
    if self.discriminator_loss_coef != 0:
        policy_state = minibatch.ref_obs
        expert_state = minibatch.ref_motion
        policy_d = self.actor_critic.discriminator(policy_state)
        expert_d = self.actor_critic.discriminator(expert_state)
        expert_loss = torch.nn.MSELoss()(
            expert_d, torch.ones(expert_d.size(), device=self.device)
        )
        policy_loss = torch.nn.MSELoss()(
            policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
        )
        grad_pen_loss = self.actor_critic.discriminator.compute_grad_pen(expert_state)
        losses["discriminator_expert_loss"] = 0.5*expert_loss
        losses["discriminator_policy_loss"] = 0.5*policy_loss
        losses["discriminator_grad_len_loss"] = grad_pen_loss
```

- If the policy has a discriminator:
  - `policy_state = minibatch.ref_obs` (policy’s imitation).
  - `expert_state = minibatch.ref_motion` (true expert motion).
  - The discriminator is trained to output **1 for expert**, **-1 for policy**.
  - Uses MSE loss between predictions and targets.
  - `grad_pen_loss`: gradient penalty to stabilize training.
  - Adds three losses: expert loss, policy loss, gradient penalty.

1. ###### Return values

```Python
return losses, inter_vars, stats
```

- Returns the extended loss dictionary (PPO losses + imitation/adversarial losses), intermediate variables, and stats.

1. APPO class

```Python
class APPO(APPOAlgoMixin, PPO):
    """
    APPO (Adversarial Proximal Policy Optimization)
    Inherits:
        APPOAlgoMixin: adversarial training features
        PPO: base PPO algorithm
    """
    pass
```

- `APPO` combines `APPOAlgoMixin` and `PPO`.
- This is the **standard adversarial PPO**:
  - PPO for stable updates.
  - Mixin for adversarial imitation learning.

1. ######  APPO class

```Python
class APPO(APPOAlgoMixin, PPO):
    """
    APPO (Adversarial Proximal Policy Optimization)
    Inherits:
        APPOAlgoMixin: adversarial training features
        PPO: base PPO algorithm
    """
    pass
```

- `APPO` combines `APPOAlgoMixin` and `PPO`.
- This is the **standard adversarial PPO**:
  - PPO for stable updates.
  - Mixin for adversarial imitation learning.

1. ###### ATPPO class

```Python
class ATPPO(APPOAlgoMixin, TPPO):
    """
    ATPPO (Adversarial Temporal PPO)
    Inherits:
        APPOAlgoMixin: adversarial training features
        TPPO: temporal PPO algorithm
    """
    pass
```

- `ATPPO` is similar, but instead of PPO it uses `TPPO` (temporal PPO).
- Suitable when temporal consistency is important (e.g., motion sequences).

#### 5.2.3.3  tppo.py

#####  General Overview

**This file implements TPPO (Teacher-guided Proximal Policy Optimization), which extends PPO by introducing a teacher network for distillation.** It defines the `TPPO` class, which manages training with:

- A base PPO algorithm (policy optimization with clipped surrogate loss).
- A **teacher policy network** that provides supervision signals (expert actions, latent embeddings).
- Distillation losses that force the student policy to match the teacher policy (actions and latent features).
- A mechanism to probabilistically decide when to use teacher actions vs. student actions during rollouts.
- Optional learning rate scheduler and hidden-state resampling for recurrent networks.

**TPPO vs PPO**

- PPO: learns only from environment rewards.
- TPPO: learns both from environment rewards **and** imitation/distillation signals from a teacher.

#####  Utility Functions

###### Utility function: `GET_PROB_FUNC`

```Python
def GET_PROB_FUNC(option, iteration_scale):
    PROB_options = {
        "linear": (lambda x: max(0, 1 - 1 / iteration_scale * x)),
        "exp": (lambda x: max(0, (1 - 1 / iteration_scale) ** x)),
        "tanh": (lambda x: max(0, 0.5 * (1 - torch.tanh(1 / iteration_scale * (x - iteration_scale))))),
    }
    return PROB_options[option]
```

- Returns a scheduling function for probabilities.
- Used to decide how often teacher actions are used during training.
- Options:
  - Linear decay
  - Exponential decay
  - Smooth decay (tanh)

#####  Class Breakdown

1. ###### Initialization

```Python
class TPPO(PPO):
    def __init__(self,
            *args,
            ppo=None,
            teacher_policy_class_name= "ActorCritic",
            teacher_policy= dict(),
            label_action_with_critic_obs= True,
            teacher_act_prob= "exp",
            update_times_scale= 100,
            using_ppo= True,
            distillation_loss_coef= 1.,
            distill_target= "real",
            distill_latent_coef= 1.,
            distill_latent_target= "real",
            distill_latent_obs_component_mapping = None,
            buffer_dilation_ratio= 1.,
            lr_scheduler_class_name= None,
            lr_scheduler= dict(),
            hidden_state_resample_prob= 0.0,
            action_labels_from_sample= False,
            **kwargs,
        ):
```

- Extends `PPO` with additional teacher-related configs.
- Key parameters:
  - `teacher_policy`: defines teacher network class and path to pretrained weights.
  - `teacher_act_prob`: probability schedule of using teacher actions.
  - `using_ppo`: if `False`, skip PPO loss and only use imitation (DAGGER mode).
  - `distillation_loss_coef`: weight for action distillation loss.
  - `distill_latent_*`: configs for latent embedding distillation.
  - `hidden_state_resample_prob`: random resampling for recurrent hidden states.
  - `action_labels_from_sample`: decides whether teacher actions come from sampling or deterministic inference.

1. ###### Teacher policy setup

```Python
teacher_actor_critic = getattr(modules, teacher_policy["class_name"])(**teacher_policy)
if teacher_policy["teacher_ac_path"] is not None:
    state_dict = torch.load(teacher_policy["teacher_ac_path"], map_location= "cpu")
    teacher_actor_critic_state_dict = state_dict["model_state_dict"]
    teacher_actor_critic.load_state_dict(teacher_actor_critic_state_dict)
else:
    print("TPPO Warning: No snapshot loaded for teacher policy. Make sure you have a pretrained teacher network")
teacher_actor_critic.to(self.device)
self.teacher_actor_critic = teacher_actor_critic
self.teacher_actor_critic.eval()
```

- Builds teacher policy from `modules`.
- Loads pretrained weights if available.
- Moves to device (GPU/CPU) and sets to evaluation mode.

1. ######  Storage

```Python
def init_storage(self, *args, **kwargs):
    self.transition = ActionLabelRollout.Transition()
    self.storage = ActionLabelRollout(
        *args,
        **kwargs,
        buffer_dilation_ratio= self.buffer_dilation_ratio,
        device= self.device,
    )
```

- Uses `ActionLabelRollout`, which stores both student actions and teacher action labels.

1. ###### Acting

```Python
def act(self, obs, critic_obs):
    return_ = super().act(obs, critic_obs) 
    ...
    self.transition.action_labels = self.teacher_actor_critic.act_inference(obs).detach()
    ...
    return return_
```

- Calls PPO’s `act` to get student actions.
- Gets teacher actions (`act` or `act_inference`) depending on configs.
- Stores them as `action_labels`.
- With probability, replaces student actions with teacher actions.

1. ###### Environment step

```Python
def process_env_step(self, rewards, dones, infos):
    return_ = super().process_env_step(rewards, dones, infos)
    self.teacher_actor_critic.reset(dones)
    index = dones.nonzero(as_tuple=False).squeeze(-1)
    self.use_teacher_act_mask[index] = (torch.rand(index.shape[0], device=self.device) < self.teacher_act_prob(self.current_learning_iteration))
    return return_
```

- Processes step as in PPO.
- Resets teacher hidden states when episodes end.
- Resamples which environments should use teacher actions.

1. ###### Distillation loss (core)

```Python
def compute_losses(self, minibatch):
    if self.using_ppo:
        losses, inter_vars, stats = super().compute_losses(minibatch)
    else:
        losses, inter_vars, stats = dict(), dict(), dict()
    ...
    # distillation loss of actions
    if self.distill_target == "real":
        dist_loss = torch.norm(self.actor_critic.action_mean - minibatch.action_labels, dim=-1)
    elif self.distill_target == "mse_sum":
        dist_loss = F.mse_loss(...).sum(-1)
    elif self.distill_target == "l1":
        dist_loss = torch.norm(..., p=1)
    elif self.distill_target == "tanh":
        dist_loss = F.binary_cross_entropy(...)
```

- Computes **distillation loss** between student actions and teacher action labels.
- Supports multiple formulations: L2, L1, MSE, BCE with tanh.
- Adds to total loss with `distillation_loss_coef`.

1. ######  Latent distillation

```Python
if self.distill_latent_obs_component_mapping is not None:
    for k, v in self.distill_latent_obs_component_mapping.items():
        latent = self.actor_critic.get_encoder_latent(minibatch.obs, k)
        target_latent = self.teacher_actor_critic.get_encoder_latent(minibatch.critic_obs, v)
        ...
        losses[f"distill_latent_{k}"] = dist_loss.mean()
```

- If configured, also distills **latent embeddings** from teacher → student.
- Matches internal representations, not just actions.
- Useful when student and teacher share encoder structures.

#### 5.2.3.4 estimator.py

##### General Overview

**This file implements an Estimator extension for PPO/TPPO.**

- Adds a supervised learning head inside the policy model: an **estimator network**.
- The estimator predicts some target components of the state (from observations).
- Training includes an **estimation loss** in addition to PPO losses.
- This allows the agent not only to act and optimize rewards, but also to learn **predictive representations** (helpful in environments where privileged info or auxiliary prediction improves generalization).

#####   Utility Functions

```Python
from st_rl.utils.utils import unpad_trajectories, get_subobs_by_components
from st_rl.storage.rollout_storage import SarsaRolloutStorage
```

- `unpad_trajectories`: removes padding from recurrent rollouts.
- `get_subobs_by_components`: extracts specific observation components.
- `SarsaRolloutStorage`: rollout buffer for SARSA-style updates (not directly used here, but relevant for supervised tasks).

#####  Class Breakdown

1. ###### EstimatorAlgoMixin

```Python
class EstimatorAlgoMixin:
    """ A supervised algorithm implementation that trains a state predictor in the policy model """
```

- A **mixin** class that adds supervised estimation to PPO/TPPO.
- Not a standalone algorithm, but combined with PPO or TPPO to form `EstimatorPPO` / `EstimatorTPPO`.

1. ###### Initialization

```Python
def __init__(self,
        *args,
        estimator_loss_func= "mse_loss",
        estimator_loss_kwargs= dict(),
        **kwargs,
    ):
    super().__init__(*args, **kwargs)
    self.estimator_obs_components = self.actor_critic.estimator_obs_components
    self.estimator_target_obs_components = self.actor_critic.estimator_target_components
    self.estimator_loss_func = estimator_loss_func
    self.estimator_loss_kwargs = estimator_loss_kwargs
```

Initializes estimator configs:

- `estimator_loss_func`: loss type (default: MSE).
- `estimator_obs_components`: input features for estimator.
- `estimator_target_obs_components`: target state components to predict.
- `estimator_loss_kwargs`: extra kwargs for loss function.

1. ###### compute_losses

```Python
def compute_losses(self, minibatch):
    losses, inter_vars, stats = super().compute_losses(minibatch)
```

- Calls parent PPO/TPPO’s `compute_losses` first.
- Adds estimation loss on top.

```Python
estimation_target = get_subobs_by_components(
    minibatch.critic_obs,
    component_names= self.estimator_target_obs_components,
    obs_segments= self.actor_critic.privileged_obs_segments,
)
if self.actor_critic.is_recurrent:
    estimation_target = unpad_trajectories(estimation_target, minibatch.masks)
```

- Extracts the target state from critic observations.
- For recurrent models, unpads trajectories.

```Python
estimation = unpad_trajectories(self.actor_critic.get_estimated_state(), minibatch.masks)
```

- Gets the predicted state from the policy’s estimator head.
- Also unpads if recurrent.

```Python
estimator_loss = getattr(F, self.estimator_loss_func)(
    estimation,
    estimation_target,
    **self.estimator_loss_kwargs,
    reduction= "none",
).sum(dim= -1)
```

- Computes estimator loss using the chosen function (e.g., `F.mse_loss`).
- `reduction="none"` → loss per sample.
- Sum over feature dimension.

```Python
losses["estimator_loss"] = estimator_loss.mean()
return losses, inter_vars, stats
```

- Adds mean estimator loss to the total losses dict.
- Returns extended losses.

1. ###### EstimatorPPO

```Python
class EstimatorPPO(EstimatorAlgoMixin, PPO):
    pass
```

- Combines `EstimatorAlgoMixin` + PPO.
- Runs PPO with estimation loss.

1. ###### EstimatorTPPO

```Python
class EstimatorTPPO(EstimatorAlgoMixin, TPPO):
    pass
```

- Combines `EstimatorAlgoMixin` + TPPO.
- Runs TPPO with estimation loss.

### 5.2.4 modules

#### ⚙️ actor_critic.py

##### 🧩 General Overview 

This module defines the *Actor-Critic architecture* used in reinforcement learning algorithms such as PPO and APPO. It provides unified implementations of the policy network (Actor) and value network (Critic), as well as a combined wrapper class (ActorCritic) that manages their interactions. In the overall algorithm pipeline, this file serves as the **core model definition**, enabling the agent to output actions and estimate values for policy optimization.

##### **📝** Key Classes & Functions

1. ###### ActorCritic

```Python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)

    def forward(self, obs):
        action_dist = self.actor(obs)
        value = self.critic(obs)
        return action_dist, value

    def act(self, obs):
        dist = self.actor(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        dist = self.actor(obs)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        values = self.critic(obs)
        return log_probs, entropy, values
```

- Wraps the Actor and Critic into one interface.
- `forward()` → returns both action distribution and value estimate.
- `act()` → samples actions + log-probs (used in rollouts).
- `evaluate_actions()` → evaluates log-probs, entropy, and values (used in PPO loss).

1. ###### Actor

```Python
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = self.fc(obs)
        mean = self.mean(x)
        std = self.log_std.exp()
        return Normal(mean, std)
```

- Policy network: maps observations → action distribution.
- Uses a 2-layer MLP with `tanh` activations.
- Outputs mean & std for Gaussian distribution (continuous actions).

1. ###### Critic

```Python
class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        return self.fc(obs)
```

- Value network: maps observations → scalar state value V(s)V(s)V(s).
- Used for advantage estimation and baseline in PPO/APPO.

##### ⚡ Usage Notes

- `ActorCritic` is the main entry point for PPO/APPO training loops.
- PPO uses `evaluate_actions()` during optimization (advantage & entropy terms).
- APPO may extend ActorCritic for distributed or parallel training.
- Ensure observation & action dimensions match environment spaces.

#### ⚙️actor_critic_field_mutex.py

##### 🧩 General Overview 

This module extends the **Actor-Critic architecture with sub-policy switching mechanisms**.

- `ActorCriticFieldMutex`: Handles environments (e.g., legged robots with obstacles/blocks) where multiple sub-policies exist, and a **field-based selection** determines which sub-policy is active.
- `ActorCriticClimbMutex`: A specialized variant for climbing/jumping tasks, adding **jump-up and jump-down policies** with custom velocity commands.

In the PPO/APPO training loop, these classes are used as **policy managers** that select, override, and reset sub-policies during inference.

##### **📝**  Key Classes & Functions

1. ###### `ActorCriticFieldMutex`

```Python
class ActorCriticFieldMutex(ActorCriticMutex):
    def __init__(self,
            *args,
            cmd_vel_mapping = dict(),
            reset_non_selected = "all",
            action_smoothing_buffer_len = 1,
            **kwargs,
        ):
        ...
```

- Inherits from `ActorCriticMutex`.
- Adds **velocity command overrides**, **policy selection smoothing**, and **reset logic**.
- Loads `cmd_scales` from sub-policy configs for normalization.

```Python
def resample_cmd_vel_current(self, dones=None):
    ...
```

- Resamples velocity commands for each sub-policy.
- Supports fixed values or random values (if tuple given).
- If `dones` is provided, applies batchwise updates.

```Python
def recover_last_action(self, observations, policy_selection):
    ...
```

- Recovers action scaling when sub-policies use different action scales.
- Ensures consistency of proprioception inputs across sub-policies.

```Python
def get_policy_selection(self, observations):
    ...
```

- Extracts `engaging_block` observation and returns a **one-hot policy ID**.
- If no obstacle is detected, defaults to the first policy.

```Python
def override_cmd_vel(self, observations, policy_selection):
    ...
```

- Overrides velocity commands in the proprioception observation.
- Uses `cmd_scales` for proper normalization.

```Python
@torch.no_grad()
def act_inference(self, observations):
    ...
```

- Runs all sub-policies in parallel for a batch.
- Maintains a buffer (action_smoothing_buffer) to smooth policy selection.
- Combines outputs with scaling factors.
- Resets non-selected sub-policies according to config (`all` / `when_skill`).

```Python
@torch.no_grad()
def reset(self, dones=None):
    ...
```

- Resamples velocity commands on reset.
- Calls parent reset.

1. ###### ActorCriticClimbMutex

```Python
class ActorCriticClimbMutex(ActorCriticFieldMutex):
    """ A variant to handle jump-up and jump-down with seperate policies """
    JUMP_OBSTACLE_ID = 3
    ...
```

- Extends `ActorCriticFieldMutex` for climbing/jumping tasks.
- Adds a **jump-down policy** (last submodule).
- Handles velocity override logic for jump-down.

```Python
def resample_cmd_vel_current(self, dones=None):
    ...
```

- Overrides parent method.
- Ensures the last policy (jump-down) has correct velocity.
- Supports fixed / random velocity ranges.

```Python
def get_policy_selection(self, observations):
    ...
```

- Extends parent policy selection.
- Uses `engaging_block` to check if it’s **jump-up or jump-down**.
- Adds a new one-hot entry for jump-down policy.

##### ⚡ Usage Notes

- These classes assume environments provide segmented observations (obs_segments).
- `cmd_vel_mapping` allows per-subpolicy velocity override; can be fixed values or ranges.
- `action_smoothing_buffer` is crucial when transitions between policies are noisy.
- `ActorCriticClimbMutex` is specifically for tasks with **jump-up / jump-down differentiation**.

#### ⚙️actor_critic_mutex.py

##### 🧩 General Overview 

This module defines the **ActorCriticMutex** class, which extends the base `ActorCritic` to support **multiple sub-policies (submodules)**. It handles **loading pre-trained sub-policy snapshots**, managing per-subpolicy action scales, and orchestrating multiple sub-policies in a single actor-critic wrapper.

##### **📝**  Key Classes & Functions

1. ###### ActorCriticMutex.__init__

```Python
def __init__(self,
            num_actor_obs,
            num_critic_obs,
            num_actions,
            sub_policy_class_name,
            sub_policy_paths,
            obs_segments= None,
            critic_obs_segments= None,
            env_action_scale = 0.5,
            **kwargs,
        ):
    ...
```

- Initializes multiple sub-policy instances from given `sub_policy_paths`.
- Loads each sub-policy config (config.json) and pre-trained weights.
- Registers `subpolicy_action_scale` for normalizing actions.
- Checks for recurrent sub-policies and sets `self.is_recurrent`.

```Python
self.submodules = nn.ModuleList()
for subpolicy_idx, sub_path in enumerate(sub_policy_paths):
    with open(osp.join(sub_path, "config.json"), "r") as f:
        run_kwargs = json.load(f, object_pairs_hook= OrderedDict)
        policy_kwargs = run_kwargs["policy"]
    self.submodules.append(getattr(modules, sub_policy_class_name)(
        num_actor_obs,
        num_critic_obs,
        num_actions,
        obs_segments= obs_segments,
        critic_obs_segments= critic_obs_segments,
        **policy_kwargs,
    ))
    if self.submodules[-1].is_recurrent: self.is_recurrent = True
```

- Iterates through sub-policy paths and loads each policy instance.
- Updates `is_recurrent` if any sub-policy uses a recurrent architecture.

1. ###### `reset()`

```Python
def reset(self, dones=None):
    for module in self.submodules:
        module.reset(dones)
```

- Resets all sub-policy modules.
- Propagates `dones` to each sub-policy.

1. ######  `act()` 与 `act_inference()`

```Python
def act(self, observations, **kwargs):
    raise NotImplementedError("Please make figure out how to load the hidden_state from exterior maintainer.")

def act_inference(self, observations):
    raise NotImplementedError("Please make figure out how to load the hidden_state from exterior maintainer.")
```

- Placeholder methods for action selection.
- These need to be implemented in derived classes (like `ActorCriticFieldMutex`) for actual inference.

1. ###### `subpolicy_action_scale registration`

```Python
self.register_buffer(
    "subpolicy_action_scale_{:d}".format(subpolicy_idx),
    torch.tensor(run_kwargs["control"]["action_scale"]) \
    if isinstance(run_kwargs["control"]["action_scale"], (tuple, list)) \
    else torch.tensor([run_kwargs["control"]["action_scale"]])[0]
)
```

- Registers the action scale for each sub-policy as a persistent buffer.
- Used to normalize outputs from each sub-policy.

##### ⚡ Usage Notes

- `ActorCriticMutex` itself does **not implement action inference**.
- Must be used as a base class for more specialized mutex policies (e.g., `ActorCriticFieldMutex`).
- Handles **loading and managing multiple pre-trained sub-policies**.
- Supports recurrent sub-policies, action scaling, and batched resets.

#### ⚙️actor_critic_recurrent.py

##### 🧩 General Overview 

This module extends the **Actor-Critic framework** with **recurrent memory** using GRU or LSTM. It enables policies to **maintain hidden states across time**, allowing learning from sequential and partially observable environments.

##### **📝**  Key Classes & Functions

1. ###### ActorCriticRecurrent

```Python
class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True
```

-  A recurrent version of `ActorCritic`, with RNN layers (GRU/LSTM) inserted before the standard actor and critic MLP networks.

```Python
def __init__(self, num_actor_obs, num_critic_obs, num_actions,
             actor_hidden_dims=[256, 256, 256],
             critic_hidden_dims=[256, 256, 256],
             activation='elu',
             rnn_type='lstm',
             rnn_hidden_size=256,
             rnn_num_layers=1,
             init_noise_std=1.0,
             **kwargs):
```

- Wraps input observations with `Memory` before passing to actor/critic.
- Configurable RNN type: `lstm` (default) or `gru`.
- Uses separate memory modules for actor (`memory_a`) and critic (`memory_c`).
- Sets RNN hidden size and layers.

```Python
def reset(self, dones=None):
    self.memory_a.reset(dones)
    self.memory_c.reset(dones)
```

- Resets the hidden states for both actor and critic memories.
- Accepts `dones` mask to selectively reset only terminated environments.

```Python
def act(self, observations, masks=None, hidden_states=None):
    input_a = self.memory_a(observations, masks, hidden_states)
    return super().act(input_a)
```

- Passes observations through the actor RNN (`memory_a`).
- Calls the parent `ActorCritic.act()` with processed inputs.

```Python
def act_inference(self, observations):
    input_a = self.memory_a(observations)
    return super().act_inference(input_a)
```

- Inference-only version of `act()`.
- For rollout without exploration noise.

```Python
def evaluate(self, critic_observations, masks=None, hidden_states=None):
    input_c = self.memory_c(critic_observations, masks, hidden_states)
    return super().evaluate(input_c)
```

- Processes critic observations with RNN (`memory_c`).
- Evaluates critic value function with sequential input.

```Python
def get_hidden_states(self):
    return ActorCriticHiddenState(self.memory_a.hidden_states, self.memory_c.hidden_states)
```

- Return the current hidden states of both actor and critic.
- Useful for saving/restoring recurrent policy states.

1. ###### Memory

```Python
class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        ...
```

- Encapsulates an RNN (`LSTM` or `GRU`) with hidden state management.

```Python
rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
self.hidden_states = None
```

- Selects RNN type (`GRU` or `LSTM`).
- Initializes the RNN with configurable hidden size and layers.
- Stores hidden states for sequential updates.

1. ######  `ActorCriticHiddenState` & `LstmHiddenState`

```Python
ActorCriticHiddenState = namedarraytuple('ActorCriticHiddenState', ['actor', 'critic'])
LstmHiddenState = namedarraytuple('LstmHiddenState', ['hidden', 'cell'])
```

- `ActorCriticHiddenState`: holds actor & critic hidden states.
- `LstmHiddenState`: holds LSTM’s `(hidden, cell)` states.

##### ⚡ Usage Notes

- `ActorCriticRecurrent` is suitable for **partially observable environments**.
- Requires careful handling of hidden states during rollout/episode transitions.
- `reset(dones)` must be called whenever environments terminate.
- Action & value networks are conditioned on **RNN-encoded inputs**, not raw observations.

#### ⚙️all_mixer.py

##### 🧩 General Overview 

This module defines **composite Actor-Critic classes** by combining mixins:

- `EstimatorMixin` → adds state estimation.
- `EncoderActorCriticMixin` → adds encoder functionality.
- `ActorCritic` / `ActorCriticRecurrent` → base policy architecture.

It provides modular, reusable policy classes with extended functionality (encoding + estimation + recurrent memory).

##### **📝**  Key Classes & Functions

1. ###### EncoderStateAc

```Python
class EncoderStateAc(EstimatorMixin, EncoderActorCriticMixin, ActorCritic):
    pass
```

- Combines **state estimation**, **encoder**, and **standard Actor-Critic**.
- Used when the policy requires both latent encoding (e.g., from raw inputs) and state estimation.

1. ###### EncoderStateAcRecurrent

```Python
class EncoderStateAcRecurrent(EstimatorMixin, EncoderActorCriticMixin, ActorCriticRecurrent):
    
    def load_misaligned_state_dict(self, module, obs_segments, critic_obs_segments=None):
        pass
```

- Extends `EncoderStateAc` with **recurrent memory** (via `ActorCriticRecurrent`).
- Suitable for **partially observable tasks** with encoder + estimator + RNN memory.
- Defines placeholder method `load_misaligned_state_dict` for handling **parameter misalignment** when loading pretrained models.

##### ⚡ Usage Notes

- These composite classes **do not add new methods** (except the placeholder in `EncoderStateAcRecurrent`).
- Their role is to **combine behaviors from multiple mixins** into a single policy class.
- `EncoderStateAc` → non-recurrent version.
- `EncoderStateAcRecurrent` → recurrent version, must manage hidden states across rollouts.
- The `load_misaligned_state_dict` method needs proper implementation before model loading works safely.

#### ⚙️amp_discriminator.py

##### 🧩 General Overview 

This module integrates **Adversarial Motion Priors (AMP)** into the Actor-Critic framework.

- Defines a **discriminator network** (`AMPDiscriminator`) that distinguishes between expert and policy-generated states.
- Provides a **mixin class** (`AmpMixin`) to add discriminator functionality into Actor-Critic policies.
- Defines two AMP-enabled policy classes:
  - `AmpActorCritic` (standard)
  - `AmpActorCriticRecurrent` (with RNN memory).

##### **📝**  Key Classes & Functions

1. ###### AMPDiscriminator

```Python
class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim, style_reward_coef, hidden_dims, task_reward_lerp=0.0, discriminator_grad_pen=5, **kwargs)
```

- A neural network discriminator used in AMP.
- Architecture: MLP (`nn.Linear` + `ReLU`) → final linear layer.
- Outputs a scalar prediction for whether input is expert-like.

**Key Methods:**

- `forward(x)` → standard forward pass.
- `compute_grad_pen(amp_obs)` → gradient penalty regularization.
- `predict_style_reward(state, task_reward)` → computes style reward from discriminator + mixes with task reward.

1. ######  `AmpMixin`

```Python
class AmpMixin:
    def __init__(..., **kwargs):
        super().__init__(...)
        cfg = kwargs.get('amp_discriminator', {})
        self.discriminator = AMPDiscriminator(**cfg)
```

- A mixin that **injects AMPDiscriminator into Actor-Critic**.
- Initializes the discriminator from config (`amp_discriminator` kwargs).

1. ######  `AmpActorCritic` / `AmpActorCriticRecurrent`

```Python
class AmpActorCritic(AmpMixin, ActorCritic):
    pass

class AmpActorCriticRecurrent(AmpMixin, ActorCriticRecurrent):
    pass
```

- **`AmpActorCritic`** → standard actor-critic with discriminator.
- **`AmpActorCriticRecurrent`** → recurrent version with memory (suitable for partially observable tasks).

##### ⚡ Usage Notes

- AMP introduces a **style reward** from the discriminator that complements task rewards.
- `task_reward_lerp` controls interpolation between style and task rewards.
- `discriminator_grad_pen` helps stabilize training via gradient penalty.
- AMP policies must handle both **policy optimization** (PPO/APPO) and **adversarial training** of the discriminator.
- `AmpActorCriticRecurrent` requires managing hidden states properly across rollouts.

#### ⚙️conv2d.py

##### 🧩 General Overview 

This file defines convolutional model components used in `st_rl` to process visual observations (images). It provides a generic **Conv2dModel** for stacked convolutional layers and a higher-level **Conv2dHeadModel** that combines convolutional feature extraction with a fully connected MLP head. These modules are typically used in actor-critic architectures when the policy or value network needs to handle image inputs.

##### **📝**  Key Classes & Functions

1. ###### Conv2dModel

```Python
class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """
```

- A stack of 2D convolutional layers (`torch.nn.Conv2d`).
- Supports **optional normalization layers** and **nonlinear activations**.
- Can use either **strides** or **max-pooling** for downsampling.
- Provides utility functions:
  - `conv_out_size(h, w)`: Computes the flattened output size for a given input resolution.
  - `conv_out_resolution(h, w)`: Computes the height and width after convolutions.

1. ######  `Conv2dHeadModel`

```Python
class Conv2dHeadModel(torch.nn.Module):
    """Model component composed of a ``Conv2dModel`` component followed by 
    a fully-connected ``MlpModel`` head.  Requires full input image shape to
    instantiate the MLP head.
    """
```

- A higher-level model that **first applies convolution** (`Conv2dModel`) and then **adds a fully-connected head** (`MlpModel`).
- Requires the **full image shape (C, H, W)** to build the MLP head.
- Output size can be specified explicitly via `output_size`, otherwise it defaults to the last hidden size.

##### ⚡ Usage Notes

- `Conv2dModel` is useful for building **feature extractors** for images in reinforcement learning environments.
- `Conv2dHeadModel` is especially handy when you want both **convolutional features** and a **flattened MLP head** (e.g., for actor-critic input).
- If `use_maxpool=True`, convolutions will have stride=1 and downsampling will happen via `MaxPool2d`.
- `conv_out_size` is very useful when you need to compute the number of features before flattening into MLP.

#### ⚙️deterministic_policy.py

##### 🧩 General Overview 

This file defines a simple **mixin class** `DeterministicPolicyMixin` that modifies the behavior of the `act()` method in policy networks. Instead of sampling actions (like in stochastic policies), it enforces **deterministic actions** by always returning the mean action (`self.action_mean`).

This is useful in contexts such as **evaluation/inference**, where deterministic behavior is preferred over exploration.

##### **📝**  Key Classes & Functions

1. ###### DeterministicPolicyMixin

```Python
class DeterministicPolicyMixin:
    def act(self, *args, **kwargs):
        return_ = super().act(*args, **kwargs)
        return self.action_mean
```

- **Purpose**: Overrides the `act()` method of a policy.
- Calls the **parent’s** **`act()`** **method** (`super().act(...)`) to preserve preprocessing logic.
- Instead of returning the sampled action, it returns `self.action_mean`, i.e., the **mean of the action distribution**.

##### ⚡ Usage Notes

- This mixin is not standalone; it must be combined with a base policy class (e.g., `ActorCritic`) that defines `self.action_mean`.
- Often used for **evaluation** (deterministic rollout) while training may still rely on stochastic policies.
- If `self.action_mean` is not defined in the parent class, this mixin will fail.

#### ⚙️encoder_actor_critic.py

##### 🧩 General Overview 

This file introduces the **EncoderActorCriticMixin**, which extends Actor-Critic architectures by embedding observations (or privileged observations) through dedicated **encoders** (MLPs or CNNs). It allows modular handling of complex observations (like proprioceptive + vision input), replacing raw segments of the observation vector with **latent features** before feeding them into the actor/critic networks.

It also provides concrete combined classes (`EncoderActorCritic`, `EncoderActorCriticRecurrent`, `EncoderAmpActorCriticRecurrent`) by mixing the encoder logic with base Actor-Critic variants.

##### **📝**  Key Classes & Functions

1. ###### EncoderActorCriticMixin

```Python
class EncoderActorCriticMixin:
    """ A general implementation where a seperate encoder is used to embed the obs/privileged_obs """
```

- **Purpose**: Provides encoder integration for observations.
- **Initialization arguments**:
  - `num_actor_obs`, `num_critic_obs`, `num_actions`: base dimensions.
  - `encoder_component_names`: names of obs segments to encode.
  - `encoder_class_name`: `"MlpModel"` / `"Conv2dHeadModel"` / list.
  - `encoder_kwargs`: encoder hyperparameters (hidden sizes, etc.).
  - `critic_encoder_component_names`: separate encoder(s) for critic input, or `"shared"` to reuse actor encoders.

```Python
def prepare_obs_slices(self):
    self.encoder_obs_slices = [get_obs_slice(self.obs_segments, name) for name in self.encoder_component_names]
    ...
```

- Computes observation slices for each encoder input. Ensures that latent embeddings are inserted in the correct order when reconstructing the observation vector.

```Python
def build_encoders(self, component_names, class_name, obs_slices, kwargs, encoder_output_size):
    ...
```

- Builds encoder modules (MLP or Conv2D) for each specified observation segment.

```Python
def embed_encoders_latent(self, observations, obs_slices, encoders, latents_order):
    ...
```

- Applies encoders to the respective observation slices, replaces them with latent vectors, and concatenates back into a full observation vector.

```Python
def get_encoder_latent(self, observations, obs_component, critic=False):
    ...
```

- Retrieves the latent representation for a **specific observation component** (useful for debugging or specialized processing).

```Python
def act(self, observations, **kwargs): ...
def act_inference(self, observations): ...
def evaluate(self, critic_observations, ...): ...
```

- Override methods from parent Actor-Critic classes:
  - `act`: encodes obs, then calls parent `act`.
  - `act_inference`: deterministic inference with encoders.
  - `evaluate`: encodes critic obs if needed, then calls parent evaluation.

1. ###### Combined Classes

```Python
class EncoderActorCritic(EncoderActorCriticMixin, ActorCritic): pass
class EncoderActorCriticRecurrent(EncoderActorCriticMixin, ActorCriticRecurrent): pass
class EncoderAmpActorCriticRecurrent(EncoderActorCriticMixin, AmpActorCriticRecurrent): pass
```

These combine the **EncoderActorCriticMixin** with different Actor-Critic variants:

- `EncoderActorCritic`: standard.
- `EncoderActorCriticRecurrent`: with recurrent policy.
- `EncoderAmpActorCriticRecurrent`: for AMP (Adversarial Motion Prior) training.

##### ⚡ Usage Notes

- Encoders are modular: add more by listing names in `encoder_component_names`.
- `critic_encoder_component_names="shared"` → critic reuses actor encoders.
- Must carefully configure `obs_segments` so slices match actual obs layout.
- Useful when obs is multi-modal (e.g., proprioception + images).

#### ⚙️mlp.py

##### 🧩 General Overview 

This module defines a **Multilayer Perceptron (MLP)** model as a reusable PyTorch component.

- Supports **flexible hidden layer configuration** (including none, making it linear).
- Allows **custom nonlinearities** (by class, not functional).
- Last layer can be **linear or nonlinear**, depending on whether `output_size` is provided.
- Provides a clean interface for retrieving the effective output dimensionality.

##### **📝**  Key Classes & Functions

1. ###### `MlpModel`

```Python
class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear."""
```

**Purpose:** General MLP block with configurable hidden layers, activation functions, and optional final linear output.

**Constructor Arguments:**

- `input_size (int)` → Input feature dimension.
- `hidden_sizes (list or int or None)` → Defines the hidden layer widths.
  - `None` or empty list → no hidden layers (pure linear).
  - Single `int` → converted to `[int]`.
- `output_size (int or None)` → Output feature dimension.
  - If `None`, the last hidden layer size is used, and **nonlinearity** is applied.
  - If specified, appends a linear output layer.
- `nonlinearity (torch.nn.Module class or str)` → Nonlinear activation.
  - Default: `torch.nn.ReLU`.
  - If passed as string, resolved dynamically from `torch.nn`.

**Implementation Details:**

- Builds `hidden_layers` using `torch.nn.Linear` + `nonlinearity`.
- Assembles the model in `torch.nn.Sequential`.
- Keeps track of `_output_size` depending on whether `output_size` is given.

**Methods:**

- `forward(input)` → Runs data through the model.
  - Input shape: `[B, input_size]`.
- `output_size (property)` → Returns effective model output dimensionality.

##### ⚡ Usage Notes

- Flexible: can represent **linear, shallow, or deep MLPs** depending on `hidden_sizes`.
- When `output_size=None`, the model ends in a **nonlinear hidden state**, often used as a feature encoder.
- When `output_size` is set, the model outputs a **linear projection**, suitable for regression or policy/value outputs.
- Useful as a **building block** in RL architectures (e.g., policy networks, critics, encoders).

#### ⚙️normalizer.py

##### 🧩 General Overview 

This module provides different strategies for **normalizing data** in reinforcement learning and machine learning pipelines.

- Implements **empirical normalization** (online updates during training).
- Provides a simple **unit vector normalization** wrapper.
- Includes **running mean and variance statistics** for streaming data.
- Extends these with a **normalizer utility** that clips observations and supports both NumPy and PyTorch. These tools are essential for stabilizing learning, avoiding exploding values, and ensuring consistent input scaling.

##### **📝**  Key Classes & Functions

1. ###### `EmpiricalNormalization`

```Python
class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""
```

- Purpose: Normalizes input data using online-updated mean and variance.
- Constructor Arguments:
  - shape (int or tuple) → Expected input shape (excluding batch).
  - eps (float) → Small constant to avoid division by zero.
  - until (int or None) → If set, stops updating after processing this many samples.

​    Key Methods & Properties:

​        forward(x) → Returns normalized values (x - mean) / (std + eps).

​        update(x) → Updates running mean/variance (only during training).

​        inverse(y) → Reverts normalization.

​        mean, std (properties) → Return current statistics.

##### ⚡ Usage Notes

### 📂runners

### 📂storage
