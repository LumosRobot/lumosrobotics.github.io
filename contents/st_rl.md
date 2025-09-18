---
layout: default
title: St Rl
nav_enabled: true
nav_order: 6
---

# st_rl
You can follow along using the code available in our [GitHub repository](https://github.com/LumosRobot/st_rl).


## algorithms

### 1.ppo.py

#### General Overview

This file implements **PPO** **(****Proximal Policy Optimization****)**, a widely used reinforcement learning algorithm. It defines the **PPO class** which manages training with:

- A policy/value network (actor_critic).
- Experience storage (`RolloutStorage`).
- The PPO update procedure (surrogate loss + value loss + entropy bonus).

This is the **foundation**: later, `APPO` extends this base PPO by adding adversarial imitation learning.

#### Class Breakdown

###### 1.Imports and class definition

``` python
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

###### 2.Initialization (__init__)

``` python
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

###### 3.Observation normalization

``` python
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

###### 4.Rollout storage

``` python
def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, **kwargs):
    self.transition = RolloutStorage.Transition()
    self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device, **kwargs)
```

- Initializes `RolloutStorage` to hold collected trajectories.
- Each transition contains: observations, actions, rewards, values, etc.

###### 5.Acting (act)

``` python
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

###### 6.Processing environment step (process_env_step)

``` python
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

###### 7.Compute returns (compute_returns)

``` python
def compute_returns(self, last_critic_obs):
    last_values = self.actor_critic.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)
```

- At the end of a rollout, computes the discounted returns and advantages.
- Uses the critic’s value for the last observation as a bootstrap.
- Calls `compute_returns` in `RolloutStorage`, which usually implements GAE (Generalized Advantage Estimation).

###### 8.Update loop (update)

``` python
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

###### 9.Compute losses (compute_losses)

``` python
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

``` python
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

``` python
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

``` python
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

``` python
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

``` python
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

###### 10.Save and load state

``` python
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

``` python
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

### 2.appo.py

#####  General Overview

**This file implements APPO (Adversarial** **Proximal Policy Optimization****), an extension of** **PPO** **for adversarial imitation learning.** It defines the APPO-related classes which manage training with:

- A base PPO algorithm (for stable policy optimization).
- Additional rollout storage (`AmpRolloutStorage`) that includes expert reference motions.
- A discriminator network that distinguishes expert motions from policy-generated motions.
- Extra loss terms (mimic loss, discriminator loss) to encourage the policy to imitate expert data.
- Style rewards combined with task rewards to balance imitation and task completion.

**This extends** **PPO**: while PPO only optimizes the policy using environment rewards and clipped updates, APPO adds **adversarial motion priors** so the agent learns both to solve the task and to move like an expert.

**APPO Basics (Reminder)**

- **Goal**: Train a policy that not only maximizes task reward but also imitates expert motion style.
- **Challenge**: Pure PPO may solve the task but produce unnatural movements.
- **Solution (APPO)**: Introduce a discriminator + imitation losses, giving the agent additional learning signals from expert demonstrations.

#####  Utility Functions

``` python
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

###### 1.Imports and utility function

``` python
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

###### 2.`APPOAlgoMixin` (Mixin for adversarial imitation) 

``` python
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

###### 3.Storage initialization

``` python
def init_storage(self, *args, **kwargs):
    self.transition = AmpRolloutStorage.Transition()
    self.storage = AmpRolloutStorage(*args, **kwargs)
```

- Initializes storage for AMP rollouts.
- `AmpRolloutStorage` is similar to normal rollout storage, but it also stores reference motions/observations for adversarial training.

###### 4.Acting (`act` override)

``` python
def act(self, obs, critic_obs):
    return_ = super().act(obs, critic_obs)  # return is transition.actions
    self.transition.action_labels = return_
    return return_
```

- Calls the parent act function (from PPO/TPPO).
- Saves the chosen actions as action_labels (used later for supervised losses like imitation)
- Return the actions to execute.

###### 5.Processing environment step

``` python
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

###### 6.Compute losses (compute_losses)

``` python
def compute_losses(self, minibatch):
    losses, inter_vars, stats = super().compute_losses(minibatch)
```

- First, calls the parent class (`PPO` or `TPPO`) `compute_losses`.
- This gives the standard PPO losses: surrogate loss, value loss, entropy.

``` python
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

``` python
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

``` python
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

###### 7.Return values

``` python
return losses, inter_vars, stats
```

- Returns the extended loss dictionary (PPO losses + imitation/adversarial losses), intermediate variables, and stats.

###### 8.APPO class

``` python
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

###### 9.ATPPO class

``` python
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

### 3.tppo.py

#####  General Overview

**This file implements TPPO (Teacher-guided** **Proximal Policy Optimization****), which extends** **PPO** **by introducing a teacher network for distillation.** It defines the `TPPO` class, which manages training with:

- A base PPO algorithm (policy optimization with clipped surrogate loss).
- A **teacher policy network** that provides supervision signals (expert actions, latent embeddings).
- Distillation losses that force the student policy to match the teacher policy (actions and latent features).
- A mechanism to probabilistically decide when to use teacher actions vs. student actions during rollouts.
- Optional learning rate scheduler and hidden-state resampling for recurrent networks.

**TPPO vs** **PPO**

- PPO: learns only from environment rewards.
- TPPO: learns both from environment rewards **and** imitation/distillation signals from a teacher.

#####  Utility Functions

Utility function: `GET_PROB_FUNC`

``` python
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

###### 1.Initialization

``` python
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

###### 2.Teacher policy setup

``` python
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

###### 3.Storage

``` python
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

###### 4.Acting

``` python
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

###### 5.Environment step

``` python
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

###### 6.Distillation loss (core)

``` python
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

###### 7. Latent distillation

``` python
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

### 4.estimator.py

##### General Overview

**This file implements an Estimator extension for** **PPO****/TPPO.**

- Adds a supervised learning head inside the policy model: an **estimator network**.
- The estimator predicts some target components of the state (from observations).
- Training includes an **estimation loss** in addition to PPO losses.
- This allows the agent not only to act and optimize rewards, but also to learn **predictive representations** (helpful in environments where privileged info or auxiliary prediction improves generalization).

#####   Utility Functions

``` python
from st_rl.utils.utils import unpad_trajectories, get_subobs_by_components
from st_rl.storage.rollout_storage import SarsaRolloutStorage
```

- `unpad_trajectories`: removes padding from recurrent rollouts.
- `get_subobs_by_components`: extracts specific observation components.
- `SarsaRolloutStorage`: rollout buffer for SARSA-style updates (not directly used here, but relevant for supervised tasks).

#####  Class Breakdown

###### 1.EstimatorAlgoMixin

``` python
class EstimatorAlgoMixin:
    """ A supervised algorithm implementation that trains a state predictor in the policy model """
```

- A **mixin** class that adds supervised estimation to PPO/TPPO.
- Not a standalone algorithm, but combined with PPO or TPPO to form `EstimatorPPO` / `EstimatorTPPO`.

###### 2.Initialization

``` python
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

###### 3.compute_losses

``` python
def compute_losses(self, minibatch):
    losses, inter_vars, stats = super().compute_losses(minibatch)
```

- Calls parent PPO/TPPO’s `compute_losses` first.
- Adds estimation loss on top.

``` python
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

``` python
estimation = unpad_trajectories(self.actor_critic.get_estimated_state(), minibatch.masks)
```

- Gets the predicted state from the policy’s estimator head.
- Also unpads if recurrent.

``` python
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

``` python
losses["estimator_loss"] = estimator_loss.mean()
return losses, inter_vars, stats
```

- Adds mean estimator loss to the total losses dict.
- Returns extended losses.

###### 4.EstimatorPPO

``` python
class EstimatorPPO(EstimatorAlgoMixin, PPO):
    pass
```

- Combines `EstimatorAlgoMixin` + PPO.
- Runs PPO with estimation loss.

###### 5.EstimatorTPPO

``` python
class EstimatorTPPO(EstimatorAlgoMixin, TPPO):
    pass
```

- Combines `EstimatorAlgoMixin` + TPPO.
- Runs TPPO with estimation loss.

## modules

### 1.actor_critic.py

##### General Overview 

This module defines the *Actor-Critic architecture* used in reinforcement learning algorithms such as PPO and APPO. It provides unified implementations of the policy network (Actor) and value network (Critic), as well as a combined wrapper class (ActorCritic) that manages their interactions. In the overall algorithm pipeline, this file serves as the **core model definition**, enabling the agent to output actions and estimate values for policy optimization.

##### Key Classes & Functions

###### 1.`ActorCritic`

``` python
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

###### 2.`Actor`

``` python
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

###### 3.`Critic`

``` python
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

##### Usage Notes

- `ActorCritic` is the main entry point for PPO/APPO training loops.
- PPO uses `evaluate_actions()` during optimization (advantage & entropy terms).
- APPO may extend ActorCritic for distributed or parallel training.
- Ensure observation & action dimensions match environment spaces.

### 2.actor_critic_field_mutex.py

##### General Overview 

This module extends the **Actor-Critic architecture with sub-policy switching mechanisms**.

- `ActorCriticFieldMutex`: Handles environments (e.g., legged robots with obstacles/blocks) where multiple sub-policies exist, and a **field-based selection** determines which sub-policy is active.
- `ActorCriticClimbMutex`: A specialized variant for climbing/jumping tasks, adding **jump-up and jump-down policies** with custom velocity commands.

In the PPO/APPO training loop, these classes are used as **policy managers** that select, override, and reset sub-policies during inference.

##### Key Classes & Functions

###### 1.`ActorCriticFieldMutex`

``` python
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

``` python
def resample_cmd_vel_current(self, dones=None):
    ...
```

- Resamples velocity commands for each sub-policy.
- Supports fixed values or random values (if tuple given).
- If `dones` is provided, applies batchwise updates.

``` python
def recover_last_action(self, observations, policy_selection):
    ...
```

- Recovers action scaling when sub-policies use different action scales.
- Ensures consistency of proprioception inputs across sub-policies.

``` python
def get_policy_selection(self, observations):
    ...
```

- Extracts `engaging_block` observation and returns a **one-hot policy ID**.
- If no obstacle is detected, defaults to the first policy.

``` python
def override_cmd_vel(self, observations, policy_selection):
    ...
```

- Overrides velocity commands in the proprioception observation.
- Uses `cmd_scales` for proper normalization.

``` python
@torch.no_grad()
def act_inference(self, observations):
    ...
```

- Runs all sub-policies in parallel for a batch.
- Maintains a buffer (action_smoothing_buffer) to smooth policy selection.
- Combines outputs with scaling factors.
- Resets non-selected sub-policies according to config (`all` / `when_skill`).

``` python
@torch.no_grad()
def reset(self, dones=None):
    ...
```

- Resamples velocity commands on reset.
- Calls parent reset

###### 2.`ActorCriticClimbMutex`

``` python
class ActorCriticClimbMutex(ActorCriticFieldMutex):
    """ A variant to handle jump-up and jump-down with seperate policies """
    JUMP_OBSTACLE_ID = 3
    ...
```

- Extends `ActorCriticFieldMutex` for climbing/jumping tasks.
- Adds a **jump-down policy** (last submodule).
- Handles velocity override logic for jump-down.

``` python
def resample_cmd_vel_current(self, dones=None):
    ...
```

- Overrides parent method.
- Ensures the last policy (jump-down) has correct velocity.
- Supports fixed / random velocity ranges.

``` python
def get_policy_selection(self, observations):
    ...
```

- Extends parent policy selection.
- Uses `engaging_block` to check if it’s **jump-up or jump-down**.
- Adds a new one-hot entry for jump-down policy.

##### Usage Notes

- These classes assume environments provide segmented observations (obs_segments).
- `cmd_vel_mapping` allows per-subpolicy velocity override; can be fixed values or ranges.
- `action_smoothing_buffer` is crucial when transitions between policies are noisy.
- `ActorCriticClimbMutex` is specifically for tasks with **jump-up / jump-down differentiation**.

### 3.actor_critic_mutex.py

##### General Overview 

This module defines the **ActorCriticMutex** class, which extends the base `ActorCritic` to support **multiple sub-policies (submodules)**. It handles **loading pre-trained sub-policy snapshots**, managing per-subpolicy action scales, and orchestrating multiple sub-policies in a single actor-critic wrapper.

#####  Key Classes & Functions

###### 1.`ActorCriticMutex.__init__`

``` python
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

``` python
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

###### 2.`reset()`

``` python
def reset(self, dones=None):
    for module in self.submodules:
        module.reset(dones)
```

- Resets all sub-policy modules.
- Propagates `dones` to each sub-policy.

###### 3.`act()` 与 `act_inference()`

``` python
def act(self, observations, **kwargs):
    raise NotImplementedError("Please make figure out how to load the hidden_state from exterior maintainer.")

def act_inference(self, observations):
    raise NotImplementedError("Please make figure out how to load the hidden_state from exterior maintainer.")
```

- Placeholder methods for action selection.
- These need to be implemented in derived classes (like `ActorCriticFieldMutex`) for actual inference.

###### 4.`subpolicy_action_scale registration`

``` python
self.register_buffer(
    "subpolicy_action_scale_{:d}".format(subpolicy_idx),
    torch.tensor(run_kwargs["control"]["action_scale"]) \
    if isinstance(run_kwargs["control"]["action_scale"], (tuple, list)) \
    else torch.tensor([run_kwargs["control"]["action_scale"]])[0]
)
```

- Registers the action scale for each sub-policy as a persistent buffer.
- Used to normalize outputs from each sub-policy.

##### Usage Notes

- `ActorCriticMutex` itself does **not implement action inference**.
- Must be used as a base class for more specialized mutex policies (e.g., `ActorCriticFieldMutex`).
- Handles **loading and managing multiple pre-trained sub-policies**.
- Supports recurrent sub-policies, action scaling, and batched resets.

### 4.actor_critic_recurrent.py

##### General Overview 

This module extends the **Actor-Critic framework** with **recurrent memory** using GRU or LSTM. It enables policies to **maintain hidden states across time**, allowing learning from sequential and partially observable environments.

##### Key Classes & Functions

###### 1.`ActorCriticRecurrent`

``` python
class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True
```

-  A recurrent version of `ActorCritic`, with RNN layers (GRU/LSTM) inserted before the standard actor and critic MLP networks.

``` python
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

``` python
def reset(self, dones=None):
    self.memory_a.reset(dones)
    self.memory_c.reset(dones)
```

- Resets the hidden states for both actor and critic memories.
- Accepts `dones` mask to selectively reset only terminated environments.

``` python
def act(self, observations, masks=None, hidden_states=None):
    input_a = self.memory_a(observations, masks, hidden_states)
    return super().act(input_a)
```

- Passes observations through the actor RNN (`memory_a`).
- Calls the parent `ActorCritic.act()` with processed inputs.

``` python
def act_inference(self, observations):
    input_a = self.memory_a(observations)
    return super().act_inference(input_a)
```

- Inference-only version of `act()`.
- For rollout without exploration noise.

``` python
def evaluate(self, critic_observations, masks=None, hidden_states=None):
    input_c = self.memory_c(critic_observations, masks, hidden_states)
    return super().evaluate(input_c)
```

- Processes critic observations with RNN (`memory_c`).
- Evaluates critic value function with sequential input.

``` python
def get_hidden_states(self):
    return ActorCriticHiddenState(self.memory_a.hidden_states, self.memory_c.hidden_states)
```

- Return the current hidden states of both actor and critic.
- Useful for saving/restoring recurrent policy states.

###### 2.`Memory`

``` python
class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        ...
```

- Encapsulates an RNN (`LSTM` or `GRU`) with hidden state management.

``` python
rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
self.hidden_states = None
```

- Selects RNN type (`GRU` or `LSTM`).
- Initializes the RNN with configurable hidden size and layers.
- Stores hidden states for sequential updates.

###### 3.`ActorCriticHiddenState` & `LstmHiddenState`

``` python
ActorCriticHiddenState = namedarraytuple('ActorCriticHiddenState', ['actor', 'critic'])
LstmHiddenState = namedarraytuple('LstmHiddenState', ['hidden', 'cell'])
```

- `ActorCriticHiddenState`: holds actor & critic hidden states.
- `LstmHiddenState`: holds LSTM’s `(hidden, cell)` states.

##### Usage Notes

- `ActorCriticRecurrent` is suitable for **partially observable environments**.
- Requires careful handling of hidden states during rollout/episode transitions.
- `reset(dones)` must be called whenever environments terminate.
- Action & value networks are conditioned on **RNN-encoded inputs**, not raw observations.

### 5.all_mixer.py

#####  General Overview 

This module defines **composite Actor-Critic classes** by combining mixins:

- `EstimatorMixin` → adds state estimation.
- `EncoderActorCriticMixin` → adds encoder functionality.
- `ActorCritic` / `ActorCriticRecurrent` → base policy architecture.

It provides modular, reusable policy classes with extended functionality (encoding + estimation + recurrent memory).

##### Key Classes & Functions

###### 1.`EncoderStateAc`

``` python
class EncoderStateAc(EstimatorMixin, EncoderActorCriticMixin, ActorCritic):
    pass
```

- Combines **state estimation**, **encoder**, and **standard Actor-Critic**.
- Used when the policy requires both latent encoding (e.g., from raw inputs) and state estimation.

###### 2.`EncoderStateAcRecurrent`

``` python
class EncoderStateAcRecurrent(EstimatorMixin, EncoderActorCriticMixin, ActorCriticRecurrent):
    
    def load_misaligned_state_dict(self, module, obs_segments, critic_obs_segments=None):
        pass
```

- Extends `EncoderStateAc` with **recurrent memory** (via `ActorCriticRecurrent`).
- Suitable for **partially observable tasks** with encoder + estimator + RNN memory.
- Defines placeholder method `load_misaligned_state_dict` for handling **parameter misalignment** when loading pretrained models.

#####  Usage Notes

- These composite classes **do not add new methods** (except the placeholder in `EncoderStateAcRecurrent`).
- Their role is to **combine behaviors from multiple mixins** into a single policy class.
- `EncoderStateAc` → non-recurrent version.
- `EncoderStateAcRecurrent` → recurrent version, must manage hidden states across rollouts.
- The `load_misaligned_state_dict` method needs proper implementation before model loading works safely.

### 6.amp_discriminator.py

#####  General Overview 

This module integrates **Adversarial Motion Priors (AMP)** into the Actor-Critic framework.

- Defines a **discriminator network** (`AMPDiscriminator`) that distinguishes between expert and policy-generated states.
- Provides a **mixin class** (`AmpMixin`) to add discriminator functionality into Actor-Critic policies.
- Defines two AMP-enabled policy classes:
  - `AmpActorCritic` (standard)
  - `AmpActorCriticRecurrent` (with RNN memory).

#####  Key Classes & Functions

###### 1.`AMPDiscriminator`

``` python
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

###### 2.`AmpMixin`

``` python
class AmpMixin:
    def __init__(..., **kwargs):
        super().__init__(...)
        cfg = kwargs.get('amp_discriminator', {})
        self.discriminator = AMPDiscriminator(**cfg)
```

- A mixin that **injects AMPDiscriminator into Actor-Critic**.
- Initializes the discriminator from config (`amp_discriminator` kwargs).

###### 3. `AmpActorCritic` / `AmpActorCriticRecurrent`

``` python
class AmpActorCritic(AmpMixin, ActorCritic):
    pass

class AmpActorCriticRecurrent(AmpMixin, ActorCriticRecurrent):
    pass
```

- **`AmpActorCritic`** → standard actor-critic with discriminator.
- **`AmpActorCriticRecurrent`** → recurrent version with memory (suitable for partially observable tasks).

##### Usage Notes

- AMP introduces a **style reward** from the discriminator that complements task rewards.
- `task_reward_lerp` controls interpolation between style and task rewards.
- `discriminator_grad_pen` helps stabilize training via gradient penalty.
- AMP policies must handle both **policy** **optimization** (PPO/APPO) and **adversarial training** of the discriminator.
- `AmpActorCriticRecurrent` requires managing hidden states properly across rollouts.

### 7.conv2d.py

#####  General Overview 

This file defines convolutional model components used in `st_rl` to process visual observations (images). It provides a generic **Conv2dModel** for stacked convolutional layers and a higher-level **Conv2dHeadModel** that combines convolutional feature extraction with a fully connected MLP head. These modules are typically used in actor-critic architectures when the policy or value network needs to handle image inputs.

##### Key Classes & Functions

###### 1.`Conv2dModel`

``` python
class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """
```

- A stack of 2D convolutional layers (`torch.nn.Conv2d`).
- Supports **optional** **normalization** **layers** and **nonlinear activations**.
- Can use either **strides** or **max-pooling** for downsampling.
- Provides utility functions:
  - `conv_out_size(h, w)`: Computes the flattened output size for a given input resolution.
  - `conv_out_resolution(h, w)`: Computes the height and width after convolutions.

###### 2.`Conv2dHeadModel`

``` python
class Conv2dHeadModel(torch.nn.Module):
    """Model component composed of a ``Conv2dModel`` component followed by 
    a fully-connected ``MlpModel`` head.  Requires full input image shape to
    instantiate the MLP head.
    """
```

- A higher-level model that **first applies** **convolution** (`Conv2dModel`) and then **adds a fully-connected head** (`MlpModel`).
- Requires the **full image shape (C, H, W)** to build the MLP head.
- Output size can be specified explicitly via `output_size`, otherwise it defaults to the last hidden size.

##### Usage Notes

- `Conv2dModel` is useful for building **feature extractors** for images in reinforcement learning environments.
- `Conv2dHeadModel` is especially handy when you want both **convolutional features** and a **flattened** **MLP** **head** (e.g., for actor-critic input).
- If `use_maxpool=True`, convolutions will have stride=1 and downsampling will happen via `MaxPool2d`.
- `conv_out_size` is very useful when you need to compute the number of features before flattening into MLP.

### 8.deterministic_policy.py

##### General Overview 

This file defines a simple **mixin class** `DeterministicPolicyMixin` that modifies the behavior of the `act()` method in policy networks. Instead of sampling actions (like in stochastic policies), it enforces **deterministic actions** by always returning the mean action (`self.action_mean`).

This is useful in contexts such as **evaluation/inference**, where deterministic behavior is preferred over exploration.

##### Key Classes & Functions

###### 1.`DeterministicPolicyMixin`

``` python
class DeterministicPolicyMixin:
    def act(self, *args, **kwargs):
        return_ = super().act(*args, **kwargs)
        return self.action_mean
```

- **Purpose**: Overrides the `act()` method of a policy.
- Calls the **parent’s** **`act()`** **method** (`super().act(...)`) to preserve preprocessing logic.
- Instead of returning the sampled action, it returns `self.action_mean`, i.e., the **mean of the action distribution**.

##### Usage Notes

- This mixin is not standalone; it must be combined with a base policy class (e.g., `ActorCritic`) that defines `self.action_mean`.
- Often used for **evaluation** (deterministic rollout) while training may still rely on stochastic policies.
- If `self.action_mean` is not defined in the parent class, this mixin will fail.

### 9.encoder_actor_critic.py

##### General Overview 

This file introduces the **EncoderActorCriticMixin**, which extends Actor-Critic architectures by embedding observations (or privileged observations) through dedicated **encoders** (MLPs or CNNs). It allows modular handling of complex observations (like proprioceptive + vision input), replacing raw segments of the observation vector with **latent features** before feeding them into the actor/critic networks.

It also provides concrete combined classes (`EncoderActorCritic`, `EncoderActorCriticRecurrent`, `EncoderAmpActorCriticRecurrent`) by mixing the encoder logic with base Actor-Critic variants.

#####  Key Classes & Functions

###### 1.EncoderActorCriticMixin

``` python
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

``` python
def prepare_obs_slices(self):
    self.encoder_obs_slices = [get_obs_slice(self.obs_segments, name) for name in self.encoder_component_names]
    ...
```

- Computes observation slices for each encoder input. Ensures that latent embeddings are inserted in the correct order when reconstructing the observation vector.

``` python
def build_encoders(self, component_names, class_name, obs_slices, kwargs, encoder_output_size):
    ...
```

- Builds encoder modules (MLP or Conv2D) for each specified observation segment.

``` python
def embed_encoders_latent(self, observations, obs_slices, encoders, latents_order):
    ...
```

- Applies encoders to the respective observation slices, replaces them with latent vectors, and concatenates back into a full observation vector.

``` python
def get_encoder_latent(self, observations, obs_component, critic=False):
    ...
```

- Retrieves the latent representation for a **specific observation component** (useful for debugging or specialized processing).

``` python
def act(self, observations, **kwargs): ...
def act_inference(self, observations): ...
def evaluate(self, critic_observations, ...): ...
```

- Override methods from parent Actor-Critic classes:
  - `act`: encodes obs, then calls parent `act`.
  - `act_inference`: deterministic inference with encoders.
  - `evaluate`: encodes critic obs if needed, then calls parent evaluation.

###### 2.Combined Classes

``` python
class EncoderActorCritic(EncoderActorCriticMixin, ActorCritic): pass
class EncoderActorCriticRecurrent(EncoderActorCriticMixin, ActorCriticRecurrent): pass
class EncoderAmpActorCriticRecurrent(EncoderActorCriticMixin, AmpActorCriticRecurrent): pass
```

These combine the **EncoderActorCriticMixin** with different Actor-Critic variants:

- `EncoderActorCritic`: standard.
- `EncoderActorCriticRecurrent`: with recurrent policy.
- `EncoderAmpActorCriticRecurrent`: for AMP (Adversarial Motion Prior) training.

#####  Usage Notes

- Encoders are modular: add more by listing names in `encoder_component_names`.
- `critic_encoder_component_names="shared"` → critic reuses actor encoders.
- Must carefully configure `obs_segments` so slices match actual obs layout.
- Useful when obs is multi-modal (e.g., proprioception + images).

### 10.mlp.py

##### General Overview 

This module defines a **Multilayer Perceptron** **(****MLP****)** model as a reusable PyTorch component.

- Supports **flexible** **hidden layer** **configuration** (including none, making it linear).
- Allows **custom nonlinearities** (by class, not functional).
- Last layer can be **linear or nonlinear**, depending on whether `output_size` is provided.
- Provides a clean interface for retrieving the effective output dimensionality.

##### Key Classes & Functions

###### `MlpModel`

``` python
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

#####  Usage Notes

- Flexible: can represent **linear, shallow, or deep MLPs** depending on `hidden_sizes`.
- When `output_size=None`, the model ends in a **nonlinear hidden state**, often used as a feature encoder.
- When `output_size` is set, the model outputs a **linear projection**, suitable for regression or policy/value outputs.
- Useful as a **building block** in RL architectures (e.g., policy networks, critics, encoders).

### 11.normalizer.py

##### General Overview 

This module provides different strategies for **normalizing data** in reinforcement learning and machine learning pipelines.

- Implements **empirical** **normalization** (online updates during training).
- Provides a simple **unit vector** **normalization** wrapper.
- Includes **running mean and** **variance** **statistics** for streaming data.
- Extends these with a **normalizer utility** that clips observations and supports both NumPy and PyTorch. These tools are essential for stabilizing learning, avoiding exploding values, and ensuring consistent input scaling.

##### Key Classes & Functions

###### 1.`EmpiricalNormalization`

``` python
class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""
```

- Purpose: Normalizes input data using online-updated mean and variance.
- Constructor Arguments:
  - shape (int or tuple) → Expected input shape (excluding batch).
  - eps (float) → Small constant to avoid division by zero.
  - until (int or None) → If set, stops updating after processing this many samples.
- Key Methods & Properties:
  - forward(x) → Returns normalized values (x - mean) / (std + eps).
  - update(x) → Updates running mean/variance (only during training).
  - inverse(y) → Reverts normalization.
  - mean, std (properties) → Return current statistics.

###### 2.`Normalize`

``` python
class Normalize(torch.nn.Module):
    """Wrapper around torch.nn.functional.normalize (L2 norm)."""
```

- Purpose: Applies L2 normalization along the last dimension.
- Methods:
  - forward(x) → Normalizes vectors to unit length (dim=-1).
- Use Case: Feature embedding normalization (e.g., in contrastive learning).

###### 3.`RunningMeanStd`

``` python
class RunningMeanStd(object):
    """Tracks running mean and variance of a data stream."""
```

**Purpose:** Online algorithm for mean/variance (parallel variance algorithm).

**Constructor Arguments:**

- `epsilon (float)` → Small constant to initialize counts.
- `shape (tuple)` → Shape of tracked statistics.

**Key Methods:**

- `update(arr)` → Updates stats from a batch of samples.
- `update_from_moments(batch_mean, batch_var, batch_count)` → Incremental moment updates.

**Use Case:** Tracking normalization stats for streaming data in RL.

###### 4.`Normalizer`

``` python
class Normalizer(RunningMeanStd):
    """Extends RunningMeanStd with clipping and PyTorch support."""
```

**Purpose:** Normalizes and optionally clips observations for stability.

**Constructor Arguments:**

- `input_dim (tuple)` → Shape of input.
- `epsilon (float)` → Stability constant.
- `clip_obs (float)` → Max absolute value after normalization.

**Key Methods:**

- `normalize(input)` → Normalizes NumPy array, clips to `[-clip_obs, clip_obs]`.
- `normalize_torch(input, device)` → Updates stats, normalizes PyTorch tensor, clips.
- `update_normalizer(rollouts, expert_loader)` → Updates stats from both policy and expert batches (AMP training).

**Use Case:** Widely used in RL pipelines to preprocess states/observations.

##### Usage Notes

- `EmpiricalNormalization` → Best when normalization should update **in-model during training**.
- `Normalize` → Lightweight L2 normalization, mainly for **embedding scaling**.
- `RunningMeanStd` → Provides a **base algorithm** for streaming stats.
- `Normalizer` → Practical RL tool: combines streaming stats, clipping, and PyTorch integration.
- `update_normalizer` is especially relevant in **adversarial imitation learning (AMP/GAIL)**, where both **expert** and **policy** data streams are combined for consistent normalization.

### 12.state_adaptor.py

##### General Overview 

This module introduces an **actor-critic extension with privileged state estimation**.

- Adds a **state adaptor network** that predicts certain hidden/privileged states from available observations.
- Supports both **feedforward** and **recurrent** (memory-based) adaptors.
- Allows probabilistic replacement of raw observations with estimated states, improving robustness and enabling partial observability handling.
- Defines recurrent and non-recurrent composite Actor-Critic classes via mixins.

##### Key Classes & Functions

###### 1.`AdaptorActorHiddenState`

``` python
AdaptorActorHiddenState = namedarraytuple('AdaptorActorHiddenState', ['adaptor', 'actor'])
```

- **Purpose:** Defines a structured container for hidden states when both **adaptor** and **actor** have their own recurrent states.
- Used in recurrent setups to keep track of adaptor’s memory separately from actor’s RNN state.

###### 2.`PrivilegeEstimatorMixin`

``` python
class PrivilegeEstimatorMixin:
    """Adds a state adaptor module for estimating privileged state features."""
```

- **Purpose:** Core mixin that augments an Actor-Critic with a **learned state estimator (adaptor)**.
- **Constructor Arguments:**
  - `adaptor_obs_components` → Components of observation used as adaptor input.
  - `adaptor_target_components` → Components of observation to be estimated by adaptor.
  - `adaptor_kwargs (dict)` → Extra config for the MLP adaptor.
  - `privilege_replace_state_prob (float)` → Probability of replacing raw observation with estimated state.
- **Key Methods:**
  - `build_adaptor()`
    - If recurrent → uses `Memory` + `MlpModel` head.
    - If feedforward → uses `MlpModel` directly.
  - `reset(dones=None)` → Resets memory (if recurrent).
  - `act(observations, masks=None, hidden_states=None, inference=False)`
    - Runs adaptor to estimate target states.
    - With probability `privilege_replace_state_prob`, substitutes estimated state into actor input.
    - Delegates action selection to parent class (feedforward or recurrent).
  - `act_inference(observations)` → Inference-only variant (no hidden states).
  - `get_estimated_state()` → Returns last computed estimated state.
  - `get_hidden_states()` → Combines adaptor + actor hidden states into a structured tuple.

**Notable Implementation Detail:**

- `substitute_estimated_state` is used to splice estimated components into the observation tensor.
- In recurrent mode, the `Memory` module is responsible for handling sequence padding/unpadding.

###### 3.`PrivilegeStateAcRecurrent`

``` python
class PrivilegeStateAcRecurrent(PrivilegeEstimatorMixin, EstimatorMixin, ActorCriticRecurrent):
    pass
```

**Purpose:** Defines a **recurrent actor-critic policy** that integrates:

- State estimator mixin (`PrivilegeEstimatorMixin`).
- General state estimator (`EstimatorMixin`).
- Recurrent actor-critic (`ActorCriticRecurrent`).

**Use Case:** Policies for **partially observable RL tasks** that benefit from:

- Estimating hidden states,
- Recurrent memory,
- Privileged replacement of inputs.

##### Usage Notes

**Feedforward vs Recurrent:**

- If `is_recurrent=False` → adaptor is just an MLP.
- If `is_recurrent=True` → adaptor includes an RNN (`Memory`) + MLP head.

**State Replacement:**

- Controlled by `privilege_replace_state_prob`.
- Encourages robustness by blending estimated states with raw inputs.

**Hidden State Management:**

- Must track adaptor + actor RNN states separately (via `AdaptorActorHiddenState`).

**Integration:**

- Plug-in mixin style means this can be easily combined with existing Actor-Critic classes.

### 13.state_estimator.py

##### General Overview 

This module introduces an **actor-critic extension with an internal state estimator**.

- Adds a **state estimator network** that predicts certain target states (latent or privileged) from a subset of observation components.
- Supports both **feedforward** and **recurrent** estimator variants.
- Can probabilistically **replace raw observation components** with estimated values, improving robustness under partial observability.
- Provides both standard and recurrent Actor-Critic classes with built-in state estimation.

##### Key Classes & Functions

###### 1.`EstimatorActorHiddenState`

``` python
EstimatorActorHiddenState = namedarraytuple('EstimatorActorHiddenState', ['estimator', 'actor'])
```

- **Purpose:** Container for **estimator** and **actor** hidden states.
- Used in recurrent policies to keep RNN memory of both estimator and actor.

###### 2.`EstimatorMixin`

``` python
class EstimatorMixin:
    """Adds a learned state estimator to Actor-Critic."""
```

- **Purpose:** Core mixin that equips an Actor-Critic policy with a **state estimator model**.
- **Constructor** **Args****:**
  - `estimator_obs_components` → observation components used as estimator input.
  - `estimator_target_components` → observation components to be predicted.
  - `estimator_kwargs (dict)` → configuration for the estimator MLP.
  - `use_actor_rnn (bool)` → if true, use actor RNN outputs as estimator input.
  - `replace_state_prob (float)` → probability of replacing raw state with estimated state.
- **Key Methods:**
  - `build_estimator()`
    - Feedforward: `MlpModel(input_size, output_size)`.
    - Recurrent + `use_actor_rnn=True`: estimator directly consumes actor RNN hidden state.
    - Recurrent + `use_actor_rnn=False`: separate estimator memory (`Memory`) + MLP head.
  - `reset(dones=None)` → resets estimator memory if needed.
  - `act(observations, masks=None, hidden_states=None, inference=False)`
    - Runs estimator to compute target state.
    - With probability `replace_state_prob`, substitutes estimated values into observations.
    - Delegates action selection to the actor (feedforward or recurrent).
  - `act_inference(observations)` → inference-only shortcut.
  - `get_estimated_state()` → returns the most recent estimated state.
  - `get_hidden_states()` → merges estimator + actor hidden states into a structured tuple.

**Notable Detail:**

- The assertion ensures you cannot set both `replace_state_prob > 0` and `use_actor_rnn=True`, since replacement after actor’s RNN already processed the input would be inconsistent.

###### 3.`StateAc`

``` python
class StateAc(EstimatorMixin, ActorCritic):
    pass
```

**Purpose:** **Non-recurrent actor-critic** with state estimation.

Suitable when observations are fully available per timestep (no need for recurrent memory).

4.`StateAcRecurrent`

``` bash
class StateAcRecurrent(EstimatorMixin, ActorCriticRecurrent):
    pass
```

- **Purpose:** **Recurrent actor-critic** with state estimation.
- Combines estimator + actor RNN memory to handle partially observable tasks.

##### Usage Notes

**Choice of** **Input****:**

- If `use_actor_rnn=True` → estimator consumes actor’s recurrent hidden state.
- Else → estimator has its own memory (`memory_s`).

**State Replacement:**

- Controlled by `replace_state_prob`.
- Encourages robustness and consistency in observation usage.

**Integration:**

- `StateAc` → feedforward version.
- `StateAcRecurrent` → recurrent version.

**Workflow****:**

- Call `.act()` → estimator predicts hidden states → optional replacement → actor selects action.
- Call `.get_estimated_state()` to access estimator outputs after each action step.

### 14.utils.py

##### General Overview 

This module provides utility functions to support model construction:

- `get_activation_Cls` → Maps string names to PyTorch activation classes.
- `conv2d_output_shape` → Computes output height & width of a Conv2D / Pooling operation given kernel, stride, padding, and dilation.

##### Key Classes & Functions

###### 1.`get_activation_Cls(activation_name)`

- Maps a string (e.g., `"relu"`, `"tanh"`) to the corresponding `torch.nn` activation class.
- Supports both built-in PyTorch names and custom aliases (`"lrelu" → LeakyReLU`, `"crelu" → ReLU`).
- Returns the **class**, not an instance.
- If the activation name is invalid, prints a warning and returns `None`.

###### 2.`activation_utils.py`

- Computes the output `(height, width)` after applying a Conv2D or pooling layer.
- Uses the standard convolution formula accounting for kernel size, stride, padding, and dilation.
- Accepts both integers and `(h, w)` tuples for parameters.

##### Usage Notes

- `get_activation_Cls` is useful when models are defined from config files with activation specified as strings.

``` bash
act_cls = get_activation_Cls("relu")
activation = act_cls()   # instantiate
```

- `conv2d_output_shape` helps design CNNs without hardcoding dimensions.

``` bash
h, w = conv2d_output_shape(64, 64, kernel_size=3, stride=2, padding=1)
print(h, w)  # → (32, 32)
```

- These utilities simplify building flexible and modular neural network architectures.

### 15.visual_actor_critic.py

##### General Overview 

This module extends the Actor-Critic framework to handle **visual observations**.

- Encodes image-like inputs (e.g., depth maps, RGB frames) into a compact latent vector via a convolutional encoder.
- Replaces the raw visual input in the observation with its latent representation before passing it to the actor or critic.
- Supports both **feedforward** and **recurrent** variants, with optional deterministic policy behavior.
- Designed for tasks where agents rely on pixel-level or spatial observations alongside state vectors.

##### Key Classes & Functions

###### 1.`VisualActorCriticMixin`

- A mixin that augments Actor-Critic with a **visual encoder**.
- **Key Parameters:**
  - `visual_component_name`: name of the observation component treated as visual input.
  - `visual_kwargs`: CNN config (channels, kernel_sizes, strides, hidden_sizes).
  - `visual_latent_size`: size of the latent embedding vector.
- **Core Methods:**
  - `embed_visual_latent(observations)`: encodes visual slice → latent vector, reinserts into observation.
  - `act(observations, **kwargs)`: runs actor with latent-embedded obs.
  - `act_inference(observations)`: inference-only version.
  - `act_with_embedded_latent(observations, **kwargs)`: assumes latent already embedded, skips re-encoding.

###### 2.`VisualDeterministicRecurrent`

- Combines **DeterministicPolicyMixin + VisualActorCriticMixin + ActorCriticRecurrent**.
- A recurrent actor-critic with deterministic actions and visual encoder.

###### 3.`VisualDeterministicAC`

- Combines **DeterministicPolicyMixin + VisualActorCriticMixin + ActorCritic**.
- A feedforward actor-critic with deterministic actions and visual encoder.

##### Usage Notes

- To use with visual input, ensure `obs_segments` includes a component (e.g., `"forward_depth"`) matching `visual_component_name`.
- Example:

``` bash
model = VisualDeterministicAC(
    num_actor_obs=512,
    num_critic_obs=512,
    num_actions=12,
    obs_segments=obs_segments,
    visual_component_name="forward_depth",
    visual_kwargs=dict(channels=[32, 64], kernel_sizes=[5, 3], strides=[2, 2]),
    visual_latent_size=128,
)

actions = model.act(observations)
```

- `act_with_embedded_latent` can be used for efficiency if multiple modules share the same visual encoder.
- Useful in **vision-based locomotion, navigation, and manipulation tasks**.

## runners

### 1.amp_policy_runner.py

#### Class: `AmpPolicyRunner`

##### Definition

``` python
class AmpPolicyRunner(OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
```

- **Inheritance**: `AmpPolicyRunner` extends `OnPolicyRunner`.
- **Purpose**: It implements a full **training loop runner** for reinforcement learning.
- **Specialty**: The prefix **AMP** (Adversarial Motion Priors) indicates that this runner is designed for imitation + RL hybrid training.

####  Core Method: `learn`

``` python
def learn(self, num_learning_iterations, init_at_random_ep_len=False, trial=None):
```

##### Parameters

- `num_learning_iterations`: number of training iterations.
- `init_at_random_ep_len`: whether to initialize episodes at random lengths (data augmentation).
- `trial`: optional handle for hyperparameter tuning frameworks (Optuna, Ray Tune, etc.), used for reporting metrics.

####  Execution Flow

1.Writer & Observation Initialization

``` python
self.init_writter(init_at_random_ep_len)
obs, extras = self.env.get_observations()
critic_obs = extras["observations"].get("critic", obs)
```

- Initializes log writer (TensorBoard, wandb, etc.).
- Gets initial observations:
  - `obs`: actor observations.
  - `critic_obs`: critic privileged observations (may include more state info).

2.Switch to Training Mode

``` python
self.train_mode()
```

- Ensures models run in training mode (dropout, batch norm, etc.).

3.Buffers Setup

``` python
ep_infos = []
rframebuffer = deque(maxlen=2000)
rewbuffer = deque(maxlen=100)
lenbuffer = deque(maxlen=100)
cur_reward_sum = torch.zeros(self.env.num_envs, ...)
cur_episode_length = torch.zeros(self.env.num_envs, ...)
```

- `ep_infos`: stores episode statistics.
- `rframebuffer`: reward frame buffer.
- `rewbuffer`: episode rewards.
- `lenbuffer`: episode lengths.
- `cur_reward_sum`: cumulative reward per environment.
- `cur_episode_length`: cumulative episode length.

4.Main Training Loop

``` python
for it in range(start_iter, tot_iter):
```

(1) Rollout Phase

``` python
with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
    for i in range(self.num_steps_per_env):
        obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
```

- Executes steps in the environment.
- Collects `(obs, critic_obs, rewards, dones, infos)`.
- Updates cumulative rewards, lengths, and episode statistics.

(2) Learning Phase

``` python
self.alg.compute_returns(critic_obs)
losses, stats = self.alg.update(self.current_learning_iteration)
```

- `compute_returns`: calculates returns/GAE.
- `update`: updates policy and critic networks, returns losses and stats.

(3) Evaluation

``` python
self.evaluation()
if trial is not None:
    trial.report(self.metrics_velrmsd, self.current_learning_iteration)
```

- Runs evaluation episodes.
- Reports metrics (`metrics_velrmsd`, `metrics_CoT`) to hyperparameter search trial if provided.

(4) Logging & Saving

``` python
if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
    self.log(locals())
if self.current_learning_iteration % self.save_interval == 0:
    self.save(...)
```

- Periodically logs training data.
- Periodically saves checkpoints.

(5) Code State Archiving

``` python
if it == start_iter:
    git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
```

- Saves current git diff for reproducibility.
- Can upload code snapshot to wandb/neptune.

5.Final Save

``` python
self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
```

- Saves the final model after training ends.

#### Example Usage

``` python
from st_rl.runners.amp_policy_runner import AmpPolicyRunner
from st_rl.env import VecEnv
import st_rl.algorithms as algorithms

# 1. Create environment
env = VecEnv(cfg)

# 2. Create algorithm
algo = algorithms.APPO(cfg, env)

# 3. Create runner
runner = AmpPolicyRunner(env, algo, device="cuda:0", log_dir="./logs")

# 4. Train
runner.learn(num_learning_iterations=10000)

# 5. Get metrics
print("Final metrics:", runner.metrics_velrmsd, runner.metrics_CoT)
```

### 2.dagger_saver.py

#### Class: `DaggerSaver`

##### Definition

``` python
class DaggerSaver(DemonstrationSaver):
    """This demonstration saver will rollout the trajectory by running the student policy
    (with a probability) and label the trajectory by running the teacher policy."""
```

- **Inheritance**: Extends `DemonstrationSaver`.
- **Purpose**: Implements the **DAgger (Dataset Aggregation)** algorithm:
  - Rollouts may use the **student policy** (current trained policy).
  - Labels are always generated by the **teacher policy** (expert / ground truth).
- **Key Idea**: Correct student actions with expert labels to improve imitation.

#### Constructor (`init`)

``` python
def __init__(..., 
             training_policy_logdir, 
             teacher_act_prob="exp", 
             update_times_scale=5000, 
             action_sample_std=0.0, 
             log_to_tensorboard=False, 
             **kwargs):
```

##### Parameters

- `training_policy_logdir`: directory where the student policy is saved/loaded.
- `teacher_act_prob`: probability of using the teacher’s action instead of student’s (can be function or string like `"exp"`).
- `update_times_scale`: scale factor for probability schedule.
- `action_sample_std`: adds Gaussian noise to student actions (exploration).
- `log_to_tensorboard`: whether to log rollout statistics.

##### Special Initialization

- If `log_to_tensorboard=True`, creates a TensorBoard writer.
- Wraps `teacher_act_prob` into a function using `GET_PROB_FUNC` if it’s a string.

#### Core Methods

##### 1.`init_traj_handlers`

``` python
def init_traj_handlers(self):
    self.metadata["training_policy_logdir"] = ...
    self.build_training_policy()
```

- Extends base trajectory handler.
- Stores metadata and builds initial **training (student) policy**.

##### 2. `init_storage_buffer`

``` python
def init_storage_buffer(self):
    self.rollout_episode_infos = []
```

- Extends base buffer.
- Adds storage for rollout episode info.

##### 3. `build_training_policy`

``` python
def build_training_policy(self):
    with open(.../config.json) as f:
        config = json.load(f)
    training_policy = build_actor_critic(...)
    self.training_policy = training_policy
    self.training_policy_iteration = 0
```

- Loads model config.
- Builds a fresh **student policy network**.

##### 4. `load_latest_training_policy`

``` python
def load_latest_training_policy(self):
    models = [file for file in os.listdir(self.training_policy_logdir) if 'model' in file]
    ...
    self.training_policy.load_state_dict(loaded_dict["model_state_dict"])
```

- Finds the newest saved model checkpoint.
- Loads weights into `self.training_policy`.
- Updates internal iteration counter.
- Randomly samples mask `use_teacher_act_mask` → determines **which envs use teacher vs student actions**.

##### 5. `get_transition`

``` python
def get_transition(self):
    teacher_actions = self.get_policy_actions()
    actions = self.training_policy.act(self.obs)
    actions[self.use_teacher_act_mask] = teacher_actions[self.use_teacher_act_mask]
    n_obs, n_critic_obs, rewards, dones, infos = self.env.step(actions)
    return teacher_actions, rewards, dones, infos, n_obs, n_critic_obs
```

- Mixes actions:
  - Student actions by default.
  - Replaced with teacher actions based on mask.
- **Teacher actions always label the trajectory.**

##### 6. `add_transition`

``` python
def add_transition(self, step_i, infos):
    if "episode" in infos:
        self.rollout_episode_infos.append(infos["episode"])
```

- Saves episode info during rollouts.

##### 7. `policy_reset`

``` python
def policy_reset(self, dones):
    if dones.any():
        self.training_policy.reset(dones)
```

- Resets both teacher and student policies when episodes end.

##### 8. `check_stop`

``` python
def check_stop(self):
    self.load_latest_training_policy()
    return super().check_stop()
```

- Extends base stopping condition.
- Ensures the latest student model is always loaded.

##### 9. `print_log`

``` python
def print_log(self):
    for key in self.rollout_episode_infos[0].keys():
        ...
        self.tb_writer.add_scalar('Episode/' + key, value, self.training_policy_iteration)
```

- Collects statistics from all rollout episodes.
- Computes mean/max/min values.
- Optionally logs them to TensorBoard.
- Prints results in a table.
- Increments student training iteration counter.

####  Example Usage

``` python
from st_rl.datasets.dagger_saver import DaggerSaver

# Initialize saver
saver = DaggerSaver(
    env=my_env,
    save_dir="./dagger_demos",
    training_policy_logdir="./student_policy",
    teacher_act_prob="exp",
    action_sample_std=0.1,
    log_to_tensorboard=True
)

# Collect demonstrations
saver.init_traj_handlers()
for i in range(1000):
    saver.load_latest_training_policy()
    teacher_actions, rewards, dones, infos, obs, critic_obs = saver.get_transition()
    saver.add_transition(i, infos)
    if dones.any():
        saver.policy_reset(dones)
    if saver.check_stop():
        break

# Print episode statistics
saver.print_log()
```

### 3.demonstration.py

#### General Overview

The `DemonstrationSaver` is a utility class designed to **collect demonstration data from an environment using a given policy** and save it into structured trajectory files. It handles trajectory segmentation, storage management, metadata tracking, and optional compression of observation segments.

This tool is particularly useful in **imitation learning** and **offline** **reinforcement learning**, where high-quality datasets of agent–environment interactions are required.

#### Key Components

##### **Initialization**

- `env` → The simulation environment (must provide `step`, `reset`, `get_observations`, etc.).
- `policy` → Any policy object supporting:
  - `act(obs, critic_obs)`
  - `act_inference(obs, critic_obs)`
  - `reset(dones)`
  - (optional) `get_hidden_states()` if recurrent.
- `save_dir` → Directory to store demonstration data.
- `rollout_storage_length` → Number of steps per rollout buffer.
- `min_timesteps` & `min_episodes` → Stopping conditions.
- `success_traj_only` → If `True`, only saves non-timeout terminated trajectories.
- `use_critic_obs` → Determines whether to use privileged critic observations for action selection.
- `obs_disassemble_mapping` → Mapping for compressing observation segments.
- `demo_by_sample` → If `True`, sample actions stochastically; otherwise, use deterministic inference.

##### **Trajectory Management**

- **`init_traj_handlers()`** Handles checkpointing of previously collected data. Ensures new runs can continue from existing trajectories.
- **`update_traj_handler(env_i, step_slice)`** Updates trajectory index after an episode ends. Removes failed/timeout trajectories if required.
- **`dump_to_file(env_i, step_slice)`** Saves a slice of trajectory into a `.pickle` file.
- **`dump_metadata()`** Saves global metadata (`timesteps`, `trajectories`, configs).

##### **Storage & Transition Handling**

- **`init_storage_buffer()`** Initializes a rollout buffer for storing transitions (`RolloutStorage`).
- **`collect_step(step_i)`** Runs one step: queries policy for actions → steps environment → builds + adds transition.
- **`save_steps()`** Dumps rollouts into files whenever the buffer fills. Splits by `done` signals.
- **`wrap_up_trajectory(env_i, step_slice)`** Prepares trajectory dictionary for saving, including compression of observation components if specified.

##### **Policy Interaction**

- **`get_policy_actions()`** Chooses actions based on:
  - critic vs actor obs
  - sampling vs deterministic inference
- **`policy_reset(dones)`** Resets hidden states when environments finish.

##### **Control Flow**

- **`check_stop()`** → Determines whether collection should stop based on thresholds.
- **`collect_and_save(config=None)`** → Main loop to collect rollouts, save them, and log progress.
- **`print_log()`** → Prints progress (timesteps, throughput).
- **`close()`** & **`del()`** → Cleanup, removing empty directories and finalizing metadata.

####  Usage Notes

- Use `obs_disassemble_mapping` if your observations contain large image-like tensors (e.g., `{"forward_rgb": "normalized_image"}`).
- Set `success_traj_only=True` when generating expert datasets for imitation learning to avoid failed attempts.
- Compatible with vectorized environments (`env.num_envs > 1`). Each environment gets its own trajectory folder.
- Stores data in **pickle format** for easy reloading.
- Metadata is always updated in `metadata.json`.

1. ### on_policy_runner.py

#### General Overview

`OnPolicyRunner` is a training manager for **on-policy** **reinforcement learning** **algorithms** (e.g., PPO, APPO, TPPO). It handles:

- Environment interaction (rollouts, resets).
- Algorithm initialization and storage setup.
- Training loop execution (rollout → update → evaluation → logging).
- Logging with multiple backends (TensorBoard, WandB, Neptune).
- Model saving/loading and checkpoint management.

The class provides a full pipeline for training policies in simulated environments, from initialization to evaluation.

#### Class Breakdown

##### 🔹 `class OnPolicyRunner`

Main class that orchestrates **on-policy RL training**.

**Constructor**

``` python
def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu")
```

- **env** (`VecEnv`) → Vectorized environment wrapper.
- **train_cfg** (dict) → Training configuration, containing `algorithm`, `policy`, and general hyperparameters.
- **log_dir** (str, optional) → Path for logs and checkpoints.
- **device** (str) → `"cpu"` or `"cuda"`.

Responsibilities:

- Initialize environment, algorithm, actor-critic model.
- Configure rollout storage and observation normalization.
- Prepare logging directories and git state tracking.

##### 🔹 `init_writer`

``` python
def init_writter(self, init_at_random_ep_len: bool)
```

- Initializes the logging backend (`tensorboard`, `wandb`, or `neptune`).
- Optionally randomizes starting episode lengths for training diversity.

##### 🔹 `learn`

``` python
def learn(self, num_learning_iterations, init_at_random_ep_len=False, trial=None)
```

- Core training loop.
- Steps:
  - Collect rollouts (`rollout_step`).
  - Compute returns.
  - Update policy (`self.alg.update`).
  - Evaluate and log metrics.
  - Save checkpoints periodically.

Returns:

- Evaluation metrics (`velrmsd`, `CoT`).

##### 🔹 `evaluation`

``` python
def evaluation(self)
```

- Computes evaluation metrics from the environment:
  - `tracking_error` → RMSD of velocity.
  - `CoT` → Cost of Transport.

##### 🔹 `rollout_step`

``` python
def rollout_step(self, obs, critic_obs, **kwargs)
```

- Executes one environment step:
  - Selects actions from policy.
  - Advances environment.
  - Normalizes observations.
  - Passes step results to algorithm storage.

##### 🔹 `log`

``` python
def log(self, locs, width=80, pad=35)
```

- Logs training statistics to console and configured writer.
- Includes:
  - Loss values, reward statistics, episode length.
  - FPS, memory usage, learning rate.
  - Evaluation metrics (`velrmsd`, `CoT`).

##### 🔹 `save`

``` python
def save(self, path: str, infos=None)
```

- Saves model checkpoint:
  - Algorithm weights.
  - RND (Random Network Distillation) state if applicable.
  - Training iteration number.

##### 🔹 `load`

``` python
def load(self, path, load_optimizer=True, map_location=None)
```

- Loads model checkpoint.
- Supports checkpoint manipulation (via `ckpt_manipulator`).
- Restores iteration count and optional optimizer states.

##### 🔹 `get_inference_policy`

``` python
def get_inference_policy(self, device=None)
```

- Returns a policy function for inference.
- Ensures normalization and device placement.

##### 🔹 Training/Eval Mode Helpers

- `train_mode` → Sets networks to training mode.
- `eval_mode` → Sets networks to evaluation mode.

##### 🔹 Git Tracking

``` python
def add_git_repo_to_log(self, repo_file_path)
```

- Adds external repositories to the logging snapshot for reproducibility.

### 5.two_stage_runner.py

#### General Overview

`TwoStageRunner` extends `OnPolicyRunner` to add a **two-stage training process**:

1. **Pretraining Stage** — the agent collects transitions from a fixed demonstration dataset (`RolloutDataset`).
2. **RL** **Stage** — the agent switches back to the normal on-policy rollout with environment interaction.

This is typically used in **imitation learning +** **reinforcement learning** **hybrid training**, where demonstration data helps bootstrap learning before RL fine-tuning.

#### Class & Methods

##### `init(...)`

- Calls the parent (`OnPolicyRunner`) constructor.
- Loads configs related to pretraining:
  - `pretrain_iterations` → how many iterations should use demonstration data before switching to RL.
  - `log_interval` → logging frequency (default 50).
  - Requires `pretrain_dataset` in the config (`assert` check).
- Initializes a **`RolloutDataset`** with:
  - Dataset config (`**self.cfg["pretrain_dataset"]`)
  - Number of environments (`self.env.num_envs`)
  - Device (`self.alg.device`)

**Key role:** sets up dataset-driven pretraining.

##### `rollout_step(obs, critic_obs)`

Overrides the `OnPolicyRunner.rollout_step`.

Behavior:

- **If still in pretraining (****`current_learning_iteration < pretrain_iterations`****):**

  - Fetch a batch from the demonstration dataset:

  - ``` python
    transition, infos = self.rollout_dataset.get_transition_batch()
    ```

  - Log performance stats from `infos` (except `time_outs`) under `Perf/dataset_*`.

  - If a `transition` exists:

    1. Feed it into the algorithm:

    2. ``` python
       self.alg.collect_transition_from_dataset(transition, infos)
       ```

    3. Return the transition fields: next obs, next privileged obs, reward, done, infos.

  - If no transition is available:

    1. Refresh observations from the environment (`env.get_observations()` / `env.get_privileged_observations()`).

- **If outside pretraining:** Falls back to standard RL rollout:

``` python
return super().rollout_step(obs, critic_obs)
```

**Key role:** switches between dataset-driven rollouts (pretrain) and environment rollouts (RL).

####  Key Points

- `TwoStageRunner` is **compatible with** **`OnPolicyRunner`**, only overriding rollout behavior.
- Useful for **demonstration-guided** **RL** (AMP, DAgger, behavior cloning + RL).
- `RolloutDataset` supplies pre-collected transitions instead of calling `env.step()`.
- Logging ensures dataset metrics are tracked alongside RL metrics.

## Storage

### 1.reply_buffer.py

####  **Class: ReplayBuffer**

##### Purpose

- Stores experience data (here: **states** and **next states**) for reinforcement learning (RL).
- Implements a **circular buffer (****ring buffer****)**: once full, new data overwrites the oldest data.
- Provides a sampling function `feed_forward_generator` to draw mini-batches for training.

#### **Constructor:** **`init`**

``` python
def __init__(self, obs_dim, buffer_size, device):
    self.states = torch.zeros(buffer_size, obs_dim).to(device)
    self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
    self.buffer_size = buffer_size
    self.device = device

    self.step = 0
    self.num_samples = 0
```

##### Arguments

- **obs_dim**: Dimension of each state vector.
- **buffer_size**: Maximum number of entries the buffer can hold.
- **device**: Device to store the tensors on (CPU or GPU).

##### Internal variables

- `self.states`: Stores current states (`[buffer_size, obs_dim]`).
- `self.next_states`: Stores next states.
- `self.step`: The current write index in the buffer.
- `self.num_samples`: Number of valid samples currently in the buffer (≤ buffer_size).

####  **Method:** **`insert`**

``` python
def insert(self, states, next_states):
    num_states = states.shape[0]
    start_idx = self.step
    end_idx = self.step + num_states
    ...
```

##### Purpose

Insert a batch of states and their corresponding next states into the buffer.

##### Logic

1. **Determine the** **write** **range**: from `self.step` to `end_idx`.
2. **Check if it exceeds buffer size**:
   1. **Not exceeded** → write directly.
   2. **Exceeded** → split writing into two parts:
      - Fill from `self.step : buffer_size`.
      - Wrap around and fill from the start `[0 : (end_idx - buffer_size)]`.
3. **Update tracking**:
   1. `self.num_samples`: updated to the number of valid samples (max capped at `buffer_size`).
   2. `self.step`: advanced to the new write position (wrapped with modulo `% buffer_size`).

#### **Method:** **`feed_forward_generator`**

``` python
def feed_forward_generator(self, num_mini_batch, mini_batch_size):
    for _ in range(num_mini_batch):
        sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
        yield (self.states[sample_idxs].to(self.device),
               self.next_states[sample_idxs].to(self.device))
```

##### Purpose

Randomly sample mini-batches of data for training.

##### Arguments

- **num_mini_batch**: Number of mini-batches to generate.
- **mini_batch_size**: Number of samples per mini-batch.

##### Logic

1. Randomly pick `mini_batch_size` indices from the valid samples (`self.num_samples`).
2. Collect the corresponding states and next states.
3. Yield them one mini-batch at a time.

### 2.rollout_storage.py

#### **1.** **`RolloutStorage`**

This is the **base class** for storing rollouts (trajectories) in reinforcement learning. It keeps all the data you need for PPO (or other policy gradient methods):

- **Core storage**: observations, critic observations (privileged info), actions, rewards, dones.
- **PPO-specific**: action log-probs, value predictions, returns, advantages, policy distribution parameters (μ, σ).
- **Optional**: hidden states (for RNN policies).

**Key features**:

- `add_transitions()` → add one step of data.
- `compute_returns()` → calculate GAE (generalized advantage estimation).
- `mini_batch_generator()` → shuffle and sample minibatches.
- `reccurent_mini_batch_generator()` → specialized batching for RNNs.
- `get_statistics()` → average episode length + average reward.

#### **2.** **`QueueRolloutStorage`** **(extends** **`RolloutStorage`****)**

Adds support for a **rolling buffer** (like a queue), useful when the rollout length isn’t fixed.

- Can **expand** the buffer size dynamically.
- Can **loop** the buffer (new data overwrites old).
- `untie_buffer_loop()` → reorders buffer so the latest data is continuous.
- Designed for training with **buffered rollouts** instead of strict episode cuts.

#### **3.** **`ActionLabelRollout`** **(extends** **`QueueRolloutStorage`****)**

A variant that also stores **action labels** (e.g., for imitation learning).

- Adds an extra `action_labels` tensor.
- MiniBatch now includes `action_labels`.
- Everything else works the same as `QueueRolloutStorage`.

#### **4.** **`SarsaRolloutStorage`** **(extends** **`RolloutStorage`****)**

Specialized for algorithms like **SARSA**, where you need both the **current state** and the **next state**.

- Stores `next_observations` and `next_critic_observations`.
- Uses an extended buffer (`all_observations`) so you can easily shift data by 1 timestep.
- Ensures that each transition has `(s, a, r, s')` aligned.

### 3.rollout_files

#### 1.base.py

##### **Class:** **`RolloutFileBase`**

This is an **abstract base class** for datasets that load and serve **rollouts (trajectories / sequences)** from files. It inherits from `torch.utils.data.IterableDataset`, so you can iterate over it like a PyTorch dataset.

It’s designed as a **template** — real implementations (subclasses) must implement the abstract methods (`reset_all`, `refresh_handlers`, `get_buffer`, `fill_transition`).

##### **Key Attributes**

- `data_dir` → where the rollout data is stored (file directory).
- `num_envs` → how many environments (parallel envs / agents) to manage.
- `device` → usually `"cuda"` or `"cpu"`.
- `__initialized` → ensures lazy initialization (reset happens on first use).
- `all_env_ids` → tensor with IDs `[0, 1, ..., num_envs-1]` representing environments.

##### **Main Methods**

###### **`reset(env_ids=None)`**

- Resets rollout handlers.
- If no env_ids given → reset **all environments**.
- If env_ids provided → only refresh handlers for those envs.
- (Useful when some envs terminate early but others keep running.)

###### **`get_batch(num_transitions_per_env=None)`**

- Fills a **buffer** with rollout data.
- If `num_transitions_per_env=None` → returns a **single transition per** **env**.
- Else → returns a sequence of transitions of length `num_transitions_per_env`.
- Calls `fill_transition()` internally to populate the buffer.
- First time it’s called, it will automatically call `reset()`.

###### **`get_transition_batch()`**

- Convenience method to simulate **environment stepping**.
- Returns `(s, a, r, d, info)` transitions like a gym env.
- If `"timeout"` field exists in buffer, wraps it in `{"time_outs": buffer.timeout}` for compatibility.

###### **Dataset Interface (****`iter`****,** **`next`****)**

- Allows iteration in PyTorch’s `DataLoader`.
- `iter()` → resets the dataset.
- `next()` → returns the next batch (via `get_batch()`).

##### **Abstract Methods (must be implemented in subclasses)**

1. **`reset_all()`**
   1. Rebuild all handlers (e.g., file readers, trajectory pointers).
   2. Reset envs to initial states.
   3. Example: start reading from the first trajectory in each env.
2. **`refresh_handlers(env_ids)`**
   1. Reset only specific envs (e.g., when they hit end of trajectory).
   2. Useful for multi-env training where envs finish episodes at different times.
3. **`get_buffer(num_transitions_per_env=None)`**
   1. Allocate an empty buffer (PyTorch tensor/dict) for transitions.
   2. Shape depends on whether `num_transitions_per_env` is set.
4. **`fill_transition(buffer, env_ids=None)`**
   1. Actually **load transitions** from file into the buffer.
   2. Advance the trajectory cursor (like stepping forward in a video).
   3. Must include both current and **next observation**.
   4. Data format per step should be `(s, a, r, d, ...)`.

##### **High-level role**

- Provides a **unified interface** for trajectory loading.
- Can be used with:
  - **offline RL** (load dataset of rollouts from disk).
  - **imitation learning** (playback expert demonstrations).
  - **hybrid methods** (mix real env + replay buffer + offline data).

#### 2.rollout_dataset.py

##### **Class Overview:** **`RolloutDataset`**

- Inherits from **`RolloutFileBase`** (abstract base class for trajectory loaders).
- Purpose: Load, manage, and feed **rollout (trajectory) data** from files into training (e.g., imitation learning, RL).
- Handles multiple environments (`num_envs`), dataset looping, shuffling, and on-demand loading.
- Maintains transitions in a **named** **tuple** (`Transition`) containing:
  - `observation`, `privileged_observation`
  - `action`, `reward`
  - `done`, `timeout`
  - `next_observation`, `next_privileged_observation`

#####  **Constructor**

``` python
def __init__(self, data_dir, num_envs, dataset_loops=1, random_shuffle_traj_order=False, keep_latest_n_trajs=0, starting_frame_range=[0, 1], device="cuda"):
```

- **Args****:**
  - `data_dir`: directory containing trajectories.
  - `num_envs`: number of parallel environments.
  - `dataset_loops`: how many times to loop dataset before stopping.
  - `random_shuffle_traj_order`: whether to randomize trajectory order.
  - `keep_latest_n_trajs`: only keep the most recent N trajectories.
  - `starting_frame_range`: where to start inside a trajectory (random within range).
  - `device`: `"cuda"` or `"cpu"`.
- Initializes counters (`num_dataset_looped`) and configs.

#####  **Data Reading & Preparation**

``` python
get_frame_range(filename)
```

- Extracts frame index range `(start, end)` from filename (e.g., `"traj_100_200.pkl"` → `(100, 200)`).

``` python
read_dataset_directory()
```

- Scans `data_dir` for trajectories (`trajectory_*` folders).
- Loads and sorts trajectories by **modification time**.
- Loads metadata (`metadata.json`) if present.
- Keeps track of unused trajectories and supports random shuffling.
- Returns `True` if enough data exists, otherwise waits.

``` python
assemble_obs_components(traj_data)
```

- Reconstructs observations from compressed components using metadata.
- Concatenates different observation parts into a full observation tensor.

##### **Handler Management**

``` python
reset_all()
```

- Clears all handlers.
- Ensures dataset directory is valid and trajectories exist.
- Initializes tracking structures for each environment: identifiers, file names, lengths, cursors, etc.
- Calls `refresh_handlers()` to assign initial trajectories.

``` python
_refresh_traj_data(env_idx)
```

- Loads a specific **trajectory file** for a given environment.
- Converts numpy arrays → PyTorch tensors (on `device`).
- Optionally reconstructs observations from compressed components.

``` python
_refresh_traj_handler(env_idx)
```

- Assigns a trajectory to an environment.
- Randomizes starting frame (within `starting_frame_range`).
- Ensures the cursor is within a valid file.
- Marks the first frame as `done=True`.

``` python
refresh_handlers(env_ids)
```

- Refreshes trajectory handlers for selected envs.
- Assigns unused trajectory IDs to them.

``` python
_maintain_handler(env_idx)
```

- Maintains trajectory progress.
- If one trajectory finishes → loads next one.
- Handles looping if dataset ends and `dataset_loops > 1`.

##### **Buffer & Transition Filling**

``` python
get_buffer(num_transitions_per_env=None)
```

- Builds an **output transition buffer** (`Transition` tuple) with required shape.
- Pre-allocates tensors for efficiency (observations, actions, rewards, dones, etc.).
- Supports both single-step and multi-step (time-major) format.

``` python
_fill_transition_per_env(buffer, env_idx)
```

- Writes a **single environment’s transition** into buffer.
- Handles:
  - Copying observation, privileged observation, action, reward, done, timeout.
  - Advancing trajectory cursor.
  - Loading next trajectory when current is exhausted.
- Ensures **next_observation** is also filled.

``` python
fill_transition(buffer, env_ids=None)
```

- Iterates over environments and fills each env’s transition into the buffer.
- If `env_ids` is `None`, processes all environments.

##### **How It Works in Training**

1. On reset: scans directories, loads available trajectories, sets handlers.
2. On `get_batch`: requests a batch of transitions.
3. On `fill_transition`: loads actual `(s, a, r, d, next_s)` from trajectories.
4. Iteratively feeds these batches into RL training.