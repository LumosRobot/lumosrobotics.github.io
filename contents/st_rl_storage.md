# Storage

## 1.reply_buffer.py

###  **Class: ReplayBuffer**

#### Purpose

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

#### Arguments

- **obs_dim**: Dimension of each state vector.
- **buffer_size**: Maximum number of entries the buffer can hold.
- **device**: Device to store the tensors on (CPU or GPU).

#### Internal variables

- `self.states`: Stores current states (`[buffer_size, obs_dim]`).
- `self.next_states`: Stores next states.
- `self.step`: The current write index in the buffer.
- `self.num_samples`: Number of valid samples currently in the buffer (≤ buffer_size).

###  **Method:** **`insert`**

``` python
def insert(self, states, next_states):
    num_states = states.shape[0]
    start_idx = self.step
    end_idx = self.step + num_states
    ...
```

#### Purpose

Insert a batch of states and their corresponding next states into the buffer.

#### Logic

1. **Determine the** **write** **range**: from `self.step` to `end_idx`.
2. **Check if it exceeds buffer size**:
   1. **Not exceeded** → write directly.
   2. **Exceeded** → split writing into two parts:
      - Fill from `self.step : buffer_size`.
      - Wrap around and fill from the start `[0 : (end_idx - buffer_size)]`.
3. **Update tracking**:
   1. `self.num_samples`: updated to the number of valid samples (max capped at `buffer_size`).
   2. `self.step`: advanced to the new write position (wrapped with modulo `% buffer_size`).

### **Method:** **`feed_forward_generator`**

``` python
def feed_forward_generator(self, num_mini_batch, mini_batch_size):
    for _ in range(num_mini_batch):
        sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
        yield (self.states[sample_idxs].to(self.device),
               self.next_states[sample_idxs].to(self.device))
```

#### Purpose

Randomly sample mini-batches of data for training.

#### Arguments

- **num_mini_batch**: Number of mini-batches to generate.
- **mini_batch_size**: Number of samples per mini-batch.

#### Logic

1. Randomly pick `mini_batch_size` indices from the valid samples (`self.num_samples`).
2. Collect the corresponding states and next states.
3. Yield them one mini-batch at a time.

## 2.rollout_storage.py

### **1.** **`RolloutStorage`**

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

### **2.** **`QueueRolloutStorage`** **(extends** **`RolloutStorage`****)**

Adds support for a **rolling buffer** (like a queue), useful when the rollout length isn’t fixed.

- Can **expand** the buffer size dynamically.
- Can **loop** the buffer (new data overwrites old).
- `untie_buffer_loop()` → reorders buffer so the latest data is continuous.
- Designed for training with **buffered rollouts** instead of strict episode cuts.

### **3.** **`ActionLabelRollout`** **(extends** **`QueueRolloutStorage`****)**

A variant that also stores **action labels** (e.g., for imitation learning).

- Adds an extra `action_labels` tensor.
- MiniBatch now includes `action_labels`.
- Everything else works the same as `QueueRolloutStorage`.

### **4.** **`SarsaRolloutStorage`** **(extends** **`RolloutStorage`****)**

Specialized for algorithms like **SARSA**, where you need both the **current state** and the **next state**.

- Stores `next_observations` and `next_critic_observations`.
- Uses an extended buffer (`all_observations`) so you can easily shift data by 1 timestep.
- Ensures that each transition has `(s, a, r, s')` aligned.

## 3.rollout_files

### 1.base.py

#### **Class:** **`RolloutFileBase`**

This is an **abstract base class** for datasets that load and serve **rollouts (trajectories / sequences)** from files. It inherits from `torch.utils.data.IterableDataset`, so you can iterate over it like a PyTorch dataset.

It’s designed as a **template** — real implementations (subclasses) must implement the abstract methods (`reset_all`, `refresh_handlers`, `get_buffer`, `fill_transition`).

#### **Key Attributes**

- `data_dir` → where the rollout data is stored (file directory).
- `num_envs` → how many environments (parallel envs / agents) to manage.
- `device` → usually `"cuda"` or `"cpu"`.
- `__initialized` → ensures lazy initialization (reset happens on first use).
- `all_env_ids` → tensor with IDs `[0, 1, ..., num_envs-1]` representing environments.

#### **Main Methods**

##### **`reset(env_ids=None)`**

- Resets rollout handlers.
- If no env_ids given → reset **all environments**.
- If env_ids provided → only refresh handlers for those envs.
- (Useful when some envs terminate early but others keep running.)

##### **`get_batch(num_transitions_per_env=None)`**

- Fills a **buffer** with rollout data.
- If `num_transitions_per_env=None` → returns a **single transition per** **env**.
- Else → returns a sequence of transitions of length `num_transitions_per_env`.
- Calls `fill_transition()` internally to populate the buffer.
- First time it’s called, it will automatically call `reset()`.

##### **`get_transition_batch()`**

- Convenience method to simulate **environment stepping**.
- Returns `(s, a, r, d, info)` transitions like a gym env.
- If `"timeout"` field exists in buffer, wraps it in `{"time_outs": buffer.timeout}` for compatibility.

##### **Dataset Interface (****`iter`****,** **`next`****)**

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

### 2.rollout_dataset.py

#### **Class Overview:** **`RolloutDataset`**

- Inherits from **`RolloutFileBase`** (abstract base class for trajectory loaders).
- Purpose: Load, manage, and feed **rollout (trajectory) data** from files into training (e.g., imitation learning, RL).
- Handles multiple environments (`num_envs`), dataset looping, shuffling, and on-demand loading.
- Maintains transitions in a **named** **tuple** (`Transition`) containing:
  - `observation`, `privileged_observation`
  - `action`, `reward`
  - `done`, `timeout`
  - `next_observation`, `next_privileged_observation`

####  **Constructor**

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

####  **Data Reading & Preparation**

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

#### **Handler Management**

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

#### **Buffer & Transition Filling**

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

#### **How It Works in Training**

1. On reset: scans directories, loads available trajectories, sets handlers.
2. On `get_batch`: requests a batch of transitions.
3. On `fill_transition`: loads actual `(s, a, r, d, next_s)` from trajectories.
4. Iteratively feeds these batches into RL training.