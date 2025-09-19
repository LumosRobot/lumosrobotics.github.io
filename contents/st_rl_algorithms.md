---
title: algorithms
parent: St Rl
nav_enabled: true
nav_order: 1
---

# algorithms

## 1.ppo.py

### General Overview

This file implements **PPO** **(****Proximal Policy Optimization****)**, a widely used reinforcement learning algorithm. It defines the **PPO class** which manages training with:

- A policy/value network (actor_critic).
- Experience storage (`RolloutStorage`).
- The PPO update procedure (surrogate loss + value loss + entropy bonus).

This is the **foundation**: later, `APPO` extends this base PPO by adding adversarial imitation learning.

### Class Breakdown

#### 1.1 Imports and class definition

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

#### 1.2 Initialization (__init__)

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

#### 1.3 Observation normalization

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

#### 1.4 Rollout storage

``` python
def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, **kwargs):
    self.transition = RolloutStorage.Transition()
    self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device, **kwargs)
```

- Initializes `RolloutStorage` to hold collected trajectories.
- Each transition contains: observations, actions, rewards, values, etc.

#### 1.5 Acting (act)

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

#### 1.6 Processing environment step (process_env_step)

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

#### 1.7 Compute returns (compute_returns)

``` python
def compute_returns(self, last_critic_obs):
    last_values = self.actor_critic.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)
```

- At the end of a rollout, computes the discounted returns and advantages.
- Uses the critic’s value for the last observation as a bootstrap.
- Calls `compute_returns` in `RolloutStorage`, which usually implements GAE (Generalized Advantage Estimation).

#### 1.8 Update loop (update)

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

#### 1.9 Compute losses (compute_losses)

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

#### 1.10 Save and load state

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

## 2.appo.py

###  General Overview

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

###  Utility Functions

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

### Class Breakdown

#### 2.1 Imports and utility function

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

#### 2.2`APPOAlgoMixin` (Mixin for adversarial imitation) 

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

#### 2.3 Storage initialization

``` python
def init_storage(self, *args, **kwargs):
    self.transition = AmpRolloutStorage.Transition()
    self.storage = AmpRolloutStorage(*args, **kwargs)
```

- Initializes storage for AMP rollouts.
- `AmpRolloutStorage` is similar to normal rollout storage, but it also stores reference motions/observations for adversarial training.

#### 2.4 Acting (`act` override)

``` python
def act(self, obs, critic_obs):
    return_ = super().act(obs, critic_obs)  # return is transition.actions
    self.transition.action_labels = return_
    return return_
```

- Calls the parent act function (from PPO/TPPO).
- Saves the chosen actions as action_labels (used later for supervised losses like imitation)
- Return the actions to execute.

#### 2.5 Processing environment step

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

#### 2.6 Compute losses (compute_losses)

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

#### 2.7 Return values

``` python
return losses, inter_vars, stats
```

- Returns the extended loss dictionary (PPO losses + imitation/adversarial losses), intermediate variables, and stats.

#### 8.APPO class

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

#### 2.9 ATPPO class

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

## 3.tppo.py

###  General Overview

**This file implements TPPO (Teacher-guided** **Proximal Policy Optimization****), which extends** **PPO** **by introducing a teacher network for distillation.** It defines the `TPPO` class, which manages training with:

- A base PPO algorithm (policy optimization with clipped surrogate loss).
- A **teacher policy network** that provides supervision signals (expert actions, latent embeddings).
- Distillation losses that force the student policy to match the teacher policy (actions and latent features).
- A mechanism to probabilistically decide when to use teacher actions vs. student actions during rollouts.
- Optional learning rate scheduler and hidden-state resampling for recurrent networks.

**TPPO vs** **PPO**

- PPO: learns only from environment rewards.
- TPPO: learns both from environment rewards **and** imitation/distillation signals from a teacher.

###  Utility Functions

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

###  Class Breakdown

#### 3.1 Initialization

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

#### 3.2 Teacher policy setup

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

#### 3.3 Storage

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

#### 3.4 Acting

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

#### 3.5 Environment step

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

#### 3.6 Distillation loss (core)

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

#### 3.7 Latent distillation

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

## 4.estimator.py

### General Overview

**This file implements an Estimator extension for** **PPO****/TPPO.**

- Adds a supervised learning head inside the policy model: an **estimator network**.
- The estimator predicts some target components of the state (from observations).
- Training includes an **estimation loss** in addition to PPO losses.
- This allows the agent not only to act and optimize rewards, but also to learn **predictive representations** (helpful in environments where privileged info or auxiliary prediction improves generalization).

###   Utility Functions

``` python
from st_rl.utils.utils import unpad_trajectories, get_subobs_by_components
from st_rl.storage.rollout_storage import SarsaRolloutStorage
```

- `unpad_trajectories`: removes padding from recurrent rollouts.
- `get_subobs_by_components`: extracts specific observation components.
- `SarsaRolloutStorage`: rollout buffer for SARSA-style updates (not directly used here, but relevant for supervised tasks).

###  Class Breakdown

#### 4.1 EstimatorAlgoMixin

``` python
class EstimatorAlgoMixin:
    """ A supervised algorithm implementation that trains a state predictor in the policy model """
```

- A **mixin** class that adds supervised estimation to PPO/TPPO.
- Not a standalone algorithm, but combined with PPO or TPPO to form `EstimatorPPO` / `EstimatorTPPO`.

#### 4.2 Initialization

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

#### 4.3 compute_losses

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

#### 4.4 EstimatorPPO

``` python
class EstimatorPPO(EstimatorAlgoMixin, PPO):
    pass
```

- Combines `EstimatorAlgoMixin` + PPO.
- Runs PPO with estimation loss.

#### 4.5 EstimatorTPPO

``` python
class EstimatorTPPO(EstimatorAlgoMixin, TPPO):
    pass
```

- Combines `EstimatorAlgoMixin` + TPPO.
- Runs TPPO with estimation loss.