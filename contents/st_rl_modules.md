---
title: modules
parent: St Rl
nav_enabled: true
nav_order: 2
---

# modules

## 1.actor_critic.py

### General Overview 

This module defines the *Actor-Critic architecture* used in reinforcement learning algorithms such as PPO and APPO. It provides unified implementations of the policy network (Actor) and value network (Critic), as well as a combined wrapper class (ActorCritic) that manages their interactions. In the overall algorithm pipeline, this file serves as the **core model definition**, enabling the agent to output actions and estimate values for policy optimization.

### Key Classes & Functions

#### 1.1 `ActorCritic`

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

#### 1.2 `Actor`

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

#### 1.3 `Critic`

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

### Usage Notes

- `ActorCritic` is the main entry point for PPO/APPO training loops.
- PPO uses `evaluate_actions()` during optimization (advantage & entropy terms).
- APPO may extend ActorCritic for distributed or parallel training.
- Ensure observation & action dimensions match environment spaces.

## 2.actor_critic_field_mutex.py

### General Overview 

This module extends the **Actor-Critic architecture with sub-policy switching mechanisms**.

- `ActorCriticFieldMutex`: Handles environments (e.g., legged robots with obstacles/blocks) where multiple sub-policies exist, and a **field-based selection** determines which sub-policy is active.
- `ActorCriticClimbMutex`: A specialized variant for climbing/jumping tasks, adding **jump-up and jump-down policies** with custom velocity commands.

In the PPO/APPO training loop, these classes are used as **policy managers** that select, override, and reset sub-policies during inference.

### Key Classes & Functions

#### 2.1 `ActorCriticFieldMutex`

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

#### 2.2 `ActorCriticClimbMutex`

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

### Usage Notes

- These classes assume environments provide segmented observations (obs_segments).
- `cmd_vel_mapping` allows per-subpolicy velocity override; can be fixed values or ranges.
- `action_smoothing_buffer` is crucial when transitions between policies are noisy.
- `ActorCriticClimbMutex` is specifically for tasks with **jump-up / jump-down differentiation**.

## 3.actor_critic_mutex.py

### General Overview 

This module defines the **ActorCriticMutex** class, which extends the base `ActorCritic` to support **multiple sub-policies (submodules)**. It handles **loading pre-trained sub-policy snapshots**, managing per-subpolicy action scales, and orchestrating multiple sub-policies in a single actor-critic wrapper.

###  Key Classes & Functions

#### 3.1 `ActorCriticMutex.__init__`

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

#### 3.2 `reset()`

``` python
def reset(self, dones=None):
    for module in self.submodules:
        module.reset(dones)
```

- Resets all sub-policy modules.
- Propagates `dones` to each sub-policy.

#### 3.3 `act()` 与 `act_inference()`

``` python
def act(self, observations, **kwargs):
    raise NotImplementedError("Please make figure out how to load the hidden_state from exterior maintainer.")

def act_inference(self, observations):
    raise NotImplementedError("Please make figure out how to load the hidden_state from exterior maintainer.")
```

- Placeholder methods for action selection.
- These need to be implemented in derived classes (like `ActorCriticFieldMutex`) for actual inference.

#### 3.4 `subpolicy_action_scale registration`

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

### Usage Notes

- `ActorCriticMutex` itself does **not implement action inference**.
- Must be used as a base class for more specialized mutex policies (e.g., `ActorCriticFieldMutex`).
- Handles **loading and managing multiple pre-trained sub-policies**.
- Supports recurrent sub-policies, action scaling, and batched resets.

## 4.actor_critic_recurrent.py

### General Overview 

This module extends the **Actor-Critic framework** with **recurrent memory** using GRU or LSTM. It enables policies to **maintain hidden states across time**, allowing learning from sequential and partially observable environments.

### Key Classes & Functions

#### 4.1 `ActorCriticRecurrent`

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

#### 4.2 `Memory`

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

#### 4.3 `ActorCriticHiddenState` & `LstmHiddenState`

``` python
ActorCriticHiddenState = namedarraytuple('ActorCriticHiddenState', ['actor', 'critic'])
LstmHiddenState = namedarraytuple('LstmHiddenState', ['hidden', 'cell'])
```

- `ActorCriticHiddenState`: holds actor & critic hidden states.
- `LstmHiddenState`: holds LSTM’s `(hidden, cell)` states.

### Usage Notes

- `ActorCriticRecurrent` is suitable for **partially observable environments**.
- Requires careful handling of hidden states during rollout/episode transitions.
- `reset(dones)` must be called whenever environments terminate.
- Action & value networks are conditioned on **RNN-encoded inputs**, not raw observations.

## 5.all_mixer.py

###  General Overview 

This module defines **composite Actor-Critic classes** by combining mixins:

- `EstimatorMixin` → adds state estimation.
- `EncoderActorCriticMixin` → adds encoder functionality.
- `ActorCritic` / `ActorCriticRecurrent` → base policy architecture.

It provides modular, reusable policy classes with extended functionality (encoding + estimation + recurrent memory).

### Key Classes & Functions

#### 5.1 `EncoderStateAc`

``` python
class EncoderStateAc(EstimatorMixin, EncoderActorCriticMixin, ActorCritic):
    pass
```

- Combines **state estimation**, **encoder**, and **standard Actor-Critic**.
- Used when the policy requires both latent encoding (e.g., from raw inputs) and state estimation.

#### 5.2 `EncoderStateAcRecurrent`

``` python
class EncoderStateAcRecurrent(EstimatorMixin, EncoderActorCriticMixin, ActorCriticRecurrent):
    
    def load_misaligned_state_dict(self, module, obs_segments, critic_obs_segments=None):
        pass
```

- Extends `EncoderStateAc` with **recurrent memory** (via `ActorCriticRecurrent`).
- Suitable for **partially observable tasks** with encoder + estimator + RNN memory.
- Defines placeholder method `load_misaligned_state_dict` for handling **parameter misalignment** when loading pretrained models.

###  Usage Notes

- These composite classes **do not add new methods** (except the placeholder in `EncoderStateAcRecurrent`).
- Their role is to **combine behaviors from multiple mixins** into a single policy class.
- `EncoderStateAc` → non-recurrent version.
- `EncoderStateAcRecurrent` → recurrent version, must manage hidden states across rollouts.
- The `load_misaligned_state_dict` method needs proper implementation before model loading works safely.

## 6.amp_discriminator.py

###  General Overview 

This module integrates **Adversarial Motion Priors (AMP)** into the Actor-Critic framework.

- Defines a **discriminator network** (`AMPDiscriminator`) that distinguishes between expert and policy-generated states.
- Provides a **mixin class** (`AmpMixin`) to add discriminator functionality into Actor-Critic policies.
- Defines two AMP-enabled policy classes:
  - `AmpActorCritic` (standard)
  - `AmpActorCriticRecurrent` (with RNN memory).

###  Key Classes & Functions

#### 6.1 `AMPDiscriminator`

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

#### 6.2 `AmpMixin`

``` python
class AmpMixin:
    def __init__(..., **kwargs):
        super().__init__(...)
        cfg = kwargs.get('amp_discriminator', {})
        self.discriminator = AMPDiscriminator(**cfg)
```

- A mixin that **injects AMPDiscriminator into Actor-Critic**.
- Initializes the discriminator from config (`amp_discriminator` kwargs).

#### 6.3 `AmpActorCritic` / `AmpActorCriticRecurrent`

``` python
class AmpActorCritic(AmpMixin, ActorCritic):
    pass

class AmpActorCriticRecurrent(AmpMixin, ActorCriticRecurrent):
    pass
```

- **`AmpActorCritic`** → standard actor-critic with discriminator.
- **`AmpActorCriticRecurrent`** → recurrent version with memory (suitable for partially observable tasks).

### Usage Notes

- AMP introduces a **style reward** from the discriminator that complements task rewards.
- `task_reward_lerp` controls interpolation between style and task rewards.
- `discriminator_grad_pen` helps stabilize training via gradient penalty.
- AMP policies must handle both **policy** **optimization** (PPO/APPO) and **adversarial training** of the discriminator.
- `AmpActorCriticRecurrent` requires managing hidden states properly across rollouts.

## 7.conv2d.py

###  General Overview 

This file defines convolutional model components used in `st_rl` to process visual observations (images). It provides a generic **Conv2dModel** for stacked convolutional layers and a higher-level **Conv2dHeadModel** that combines convolutional feature extraction with a fully connected MLP head. These modules are typically used in actor-critic architectures when the policy or value network needs to handle image inputs.

### Key Classes & Functions

#### 7.1 `Conv2dModel`

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

#### 7.2 `Conv2dHeadModel`

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

### Usage Notes

- `Conv2dModel` is useful for building **feature extractors** for images in reinforcement learning environments.
- `Conv2dHeadModel` is especially handy when you want both **convolutional features** and a **flattened** **MLP** **head** (e.g., for actor-critic input).
- If `use_maxpool=True`, convolutions will have stride=1 and downsampling will happen via `MaxPool2d`.
- `conv_out_size` is very useful when you need to compute the number of features before flattening into MLP.

## 8.deterministic_policy.py

### General Overview 

This file defines a simple **mixin class** `DeterministicPolicyMixin` that modifies the behavior of the `act()` method in policy networks. Instead of sampling actions (like in stochastic policies), it enforces **deterministic actions** by always returning the mean action (`self.action_mean`).

This is useful in contexts such as **evaluation/inference**, where deterministic behavior is preferred over exploration.

### Key Classes & Functions

#### 8.1 `DeterministicPolicyMixin`

``` python
class DeterministicPolicyMixin:
    def act(self, *args, **kwargs):
        return_ = super().act(*args, **kwargs)
        return self.action_mean
```

- **Purpose**: Overrides the `act()` method of a policy.
- Calls the **parent’s** **`act()`** **method** (`super().act(...)`) to preserve preprocessing logic.
- Instead of returning the sampled action, it returns `self.action_mean`, i.e., the **mean of the action distribution**.

### Usage Notes

- This mixin is not standalone; it must be combined with a base policy class (e.g., `ActorCritic`) that defines `self.action_mean`.
- Often used for **evaluation** (deterministic rollout) while training may still rely on stochastic policies.
- If `self.action_mean` is not defined in the parent class, this mixin will fail.

## 9.encoder_actor_critic.py

### General Overview 

This file introduces the **EncoderActorCriticMixin**, which extends Actor-Critic architectures by embedding observations (or privileged observations) through dedicated **encoders** (MLPs or CNNs). It allows modular handling of complex observations (like proprioceptive + vision input), replacing raw segments of the observation vector with **latent features** before feeding them into the actor/critic networks.

It also provides concrete combined classes (`EncoderActorCritic`, `EncoderActorCriticRecurrent`, `EncoderAmpActorCriticRecurrent`) by mixing the encoder logic with base Actor-Critic variants.

###  Key Classes & Functions

#### 9.1 EncoderActorCriticMixin

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

#### 9.2 Combined Classes

``` python
class EncoderActorCritic(EncoderActorCriticMixin, ActorCritic): pass
class EncoderActorCriticRecurrent(EncoderActorCriticMixin, ActorCriticRecurrent): pass
class EncoderAmpActorCriticRecurrent(EncoderActorCriticMixin, AmpActorCriticRecurrent): pass
```

These combine the **EncoderActorCriticMixin** with different Actor-Critic variants:

- `EncoderActorCritic`: standard.
- `EncoderActorCriticRecurrent`: with recurrent policy.
- `EncoderAmpActorCriticRecurrent`: for AMP (Adversarial Motion Prior) training.

###  Usage Notes

- Encoders are modular: add more by listing names in `encoder_component_names`.
- `critic_encoder_component_names="shared"` → critic reuses actor encoders.
- Must carefully configure `obs_segments` so slices match actual obs layout.
- Useful when obs is multi-modal (e.g., proprioception + images).

## 10.mlp.py

### General Overview 

This module defines a **Multilayer Perceptron** **(****MLP****)** model as a reusable PyTorch component.

- Supports **flexible** **hidden layer** **configuration** (including none, making it linear).
- Allows **custom nonlinearities** (by class, not functional).
- Last layer can be **linear or nonlinear**, depending on whether `output_size` is provided.
- Provides a clean interface for retrieving the effective output dimensionality.

### Key Classes & Functions

#### `MlpModel`

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

###  Usage Notes

- Flexible: can represent **linear, shallow, or deep MLPs** depending on `hidden_sizes`.
- When `output_size=None`, the model ends in a **nonlinear hidden state**, often used as a feature encoder.
- When `output_size` is set, the model outputs a **linear projection**, suitable for regression or policy/value outputs.
- Useful as a **building block** in RL architectures (e.g., policy networks, critics, encoders).

## 11.normalizer.py

### General Overview 

This module provides different strategies for **normalizing data** in reinforcement learning and machine learning pipelines.

- Implements **empirical** **normalization** (online updates during training).
- Provides a simple **unit vector** **normalization** wrapper.
- Includes **running mean and** **variance** **statistics** for streaming data.
- Extends these with a **normalizer utility** that clips observations and supports both NumPy and PyTorch. These tools are essential for stabilizing learning, avoiding exploding values, and ensuring consistent input scaling.

### Key Classes & Functions

#### 11.1 `EmpiricalNormalization`

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

#### 11.2 `Normalize`

``` python
class Normalize(torch.nn.Module):
    """Wrapper around torch.nn.functional.normalize (L2 norm)."""
```

- Purpose: Applies L2 normalization along the last dimension.
- Methods:
  - forward(x) → Normalizes vectors to unit length (dim=-1).
- Use Case: Feature embedding normalization (e.g., in contrastive learning).

#### 11.3 `RunningMeanStd`

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

#### 11.4 `Normalizer`

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

### Usage Notes

- `EmpiricalNormalization` → Best when normalization should update **in-model during training**.
- `Normalize` → Lightweight L2 normalization, mainly for **embedding scaling**.
- `RunningMeanStd` → Provides a **base algorithm** for streaming stats.
- `Normalizer` → Practical RL tool: combines streaming stats, clipping, and PyTorch integration.
- `update_normalizer` is especially relevant in **adversarial imitation learning (AMP/GAIL)**, where both **expert** and **policy** data streams are combined for consistent normalization.

## 12.state_adaptor.py

### General Overview 

This module introduces an **actor-critic extension with privileged state estimation**.

- Adds a **state adaptor network** that predicts certain hidden/privileged states from available observations.
- Supports both **feedforward** and **recurrent** (memory-based) adaptors.
- Allows probabilistic replacement of raw observations with estimated states, improving robustness and enabling partial observability handling.
- Defines recurrent and non-recurrent composite Actor-Critic classes via mixins.

### Key Classes & Functions

#### 12.1 `AdaptorActorHiddenState`

``` python
AdaptorActorHiddenState = namedarraytuple('AdaptorActorHiddenState', ['adaptor', 'actor'])
```

- **Purpose:** Defines a structured container for hidden states when both **adaptor** and **actor** have their own recurrent states.
- Used in recurrent setups to keep track of adaptor’s memory separately from actor’s RNN state.

#### 12.2 `PrivilegeEstimatorMixin`

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

#### 12.3 PrivilegeStateAcRecurrent`

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

### Usage Notes

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

## 13.state_estimator.py

### General Overview 

This module introduces an **actor-critic extension with an internal state estimator**.

- Adds a **state estimator network** that predicts certain target states (latent or privileged) from a subset of observation components.
- Supports both **feedforward** and **recurrent** estimator variants.
- Can probabilistically **replace raw observation components** with estimated values, improving robustness under partial observability.
- Provides both standard and recurrent Actor-Critic classes with built-in state estimation.

### Key Classes & Functions

#### 13.1 `EstimatorActorHiddenState`

``` python
EstimatorActorHiddenState = namedarraytuple('EstimatorActorHiddenState', ['estimator', 'actor'])
```

- **Purpose:** Container for **estimator** and **actor** hidden states.
- Used in recurrent policies to keep RNN memory of both estimator and actor.

#### 13.2 `EstimatorMixin`

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

#### 13.3 `StateAc`

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

### Usage Notes

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

## 14.utils.py

### General Overview 

This module provides utility functions to support model construction:

- `get_activation_Cls` → Maps string names to PyTorch activation classes.
- `conv2d_output_shape` → Computes output height & width of a Conv2D / Pooling operation given kernel, stride, padding, and dilation.

### Key Classes & Functions

#### 14.1 `get_activation_Cls(activation_name)`

- Maps a string (e.g., `"relu"`, `"tanh"`) to the corresponding `torch.nn` activation class.
- Supports both built-in PyTorch names and custom aliases (`"lrelu" → LeakyReLU`, `"crelu" → ReLU`).
- Returns the **class**, not an instance.
- If the activation name is invalid, prints a warning and returns `None`.

#### 14.2 `activation_utils.py`

- Computes the output `(height, width)` after applying a Conv2D or pooling layer.
- Uses the standard convolution formula accounting for kernel size, stride, padding, and dilation.
- Accepts both integers and `(h, w)` tuples for parameters.

### Usage Notes

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

## 15.visual_actor_critic.py

### General Overview 

This module extends the Actor-Critic framework to handle **visual observations**.

- Encodes image-like inputs (e.g., depth maps, RGB frames) into a compact latent vector via a convolutional encoder.
- Replaces the raw visual input in the observation with its latent representation before passing it to the actor or critic.
- Supports both **feedforward** and **recurrent** variants, with optional deterministic policy behavior.
- Designed for tasks where agents rely on pixel-level or spatial observations alongside state vectors.

### Key Classes & Functions

#### 15.1 `VisualActorCriticMixin`

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

#### 15.2 `VisualDeterministicRecurrent`

- Combines **DeterministicPolicyMixin + VisualActorCriticMixin + ActorCriticRecurrent**.
- A recurrent actor-critic with deterministic actions and visual encoder.

#### 15.3 `VisualDeterministicAC`

- Combines **DeterministicPolicyMixin + VisualActorCriticMixin + ActorCritic**.
- A feedforward actor-critic with deterministic actions and visual encoder.

### Usage Notes

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