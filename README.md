# Pool

The `Pool` class is designed for efficient, parallelized data collection from multiple environments, particularly useful in reinforcement learning settings. It leverages Python's `multiprocessing` module to manage shared memory and execute environment interactions concurrently.

## Features

- **Parallel Data Collection**: Collects `(state, action, next_state, reward, done)` tuples from multiple environments in parallel.
- **Shared Memory Management**: Utilizes `multiprocessing.Manager` to create shared lists for storing collected data, allowing all processes to access and modify the same data structures.
- **Configurable Pool Size**: Allows you to specify the total desired size of the data pool.
- **Windowed Data Management**: Supports keeping only the most recent `window_size` or `window_size_` samples, useful for maintaining a sliding window of experience.
- **Randomized Prioritization (Optional)**: Can prioritize which sub-pool (corresponding to a process) gets new data based on the inverse of its current length, ensuring more balanced data distribution.
- **Clearing Frequency (Optional)**: Periodically clears older data based on a specified frequency, preventing indefinite growth.

## Class Definition

```python
import numpy as np
import multiprocessing as mp
import math

class Pool:
    def __init__(self, env, processes, pool_size, window_size=None, clearing_freq=None, window_size_=None, random=True):
        # ... (rest of the __init__ method)
```

## Constructor (`__init__`)

Initializes the `Pool` object.

**Parameters**:

- **`env`** (List[gym.Env]): A list of environment instances, one for each process. Each environment should implement `reset()` and `step()`.
- **`processes`** (int): The number of parallel processes to use for data collection.
- **`pool_size`** (int): The maximum desired total size of the combined data pools across all processes.
- **`window_size`** (int, optional): If provided, when a sub-pool exceeds its allocated size (based on `pool_size / processes`), it will be truncated to keep only the most recent `window_size` samples. Defaults to `None`, which means only the oldest single sample is removed.
- **`clearing_freq`** (int, optional): If provided, data in each sub-pool will be truncated by `window_size_` samples every `clearing_freq` steps. Defaults to `None`.
- **`window_size_`** (int, optional): Used in conjunction with `clearing_freq`. Specifies the number of oldest samples to remove when clearing. Defaults to `None`.
- **`random`** (bool, optional): If `True`, data will be distributed to sub-pools based on an inverse length probability, aiming to keep sub-pools more evenly sized. If `False`, data is stored sequentially in the process's dedicated sub-pool. Defaults to `True`.

## Methods

### `pool(self, s, a, next_s, r, done, index=None)`

This internal method is responsible for adding new experience tuples to the appropriate data lists and managing their size. It handles concatenation of new data and applies truncation rules based on `window_size`, `clearing_freq`, and `window_size_`.

**Parameters**:

- **`s`** (np.array): The current state.
- **`a`** (any): The action taken.
- **`next_s`** (np.array): The next state observed after taking the action.
- **`r`** (float): The reward received.
- **`done`** (bool): A boolean indicating if the episode terminated.
- **`index`** (int, optional): The index of the sub-pool (and thus the process) to which the data belongs.

### `store_in_parallel(self, env, p, lock_list)`

This method is executed by each parallel process. It continuously interacts with its assigned environment (`self.env[p]`), collects experience tuples, and then calls the `pool` method to store them. If `random` is `True`, it uses locks to ensure thread-safe access to shared data structures when determining which sub-pool to update.

**Parameters**:

- **`p`** (int): The index of the current process.
- **`lock_list`** (List[mp.Lock]): A list of multiprocessing locks used for synchronizing access to shared data when `random` is `True`.

### `store(self)`

This is the main method to initiate the parallel data collection. It creates and starts `self.processes` number of child processes, each running the `store_in_parallel` method. It then waits for all processes to complete (though in a continuous collection scenario, you might manage process lifetimes differently).

### `get_pool(self)`

Retrieves the aggregated data from all sub-pools. This method concatenates all the individual lists (state, action, next state, reward, done) from each process into single NumPy arrays.

**Returns**:

- **`state_pool`** (np.array): Concatenated array of all states.
- **`action_pool`** (np.array): Concatenated array of all actions.
- **`next_state_pool`** (np.array): Concatenated array of all next states.
- **`reward_pool`** (np.array): Concatenated array of all rewards.
- **`done_pool`** (np.array): Concatenated array of all done flags.

## Usage Example

```python
import numpy as np
import multiprocessing as mp
import math
import gym # Assuming gym environments

# Define a dummy environment for demonstration
class DummyEnv:
    def __init__(self):
        self.state = 0
        self.steps = 0

    def reset(self):
        self.state = np.random.rand(4)
        self.steps = 0
        print(f"Env {self.id}: Resetting. Initial state: {self.state}")
        return self.state, 0 # Return state and dummy action

    def step(self, action):
        self.state += np.random.rand(4) * 0.1
        reward = np.sum(self.state)
        self.steps += 1
        done = self.steps >= 10 # Example: episode ends after 10 steps
        print(f"Env {self.id}: Step {self.steps}. State: {self.state}, Reward: {reward}, Done: {done}")
        return np.random.randint(0,2), self.state, reward, done # Dummy action, next_state, reward, done

if __name__ == "__main__":
    num_processes = 2
    envs = [DummyEnv() for _ in range(num_processes)]
    total_pool_size = 50

    # Example with clearing_freq and window_size_
    # pool_manager = Pool(envs, num_processes, total_pool_size, clearing_freq=5, window_size_=2, random=False)
    
    # Example with window_size
    pool_manager = Pool(envs, num_processes, total_pool_size, window_size=5, random=True)

    print("Starting data collection...")

    pool_manager.store()
    
    # Now, get the collected data
    state, action, next_state, reward, done = pool_manager.get_pool()

    print(f"\nCollected States Shape: {state.shape}")
    print(f"Collected Actions Shape: {action.shape}")
    print(f"Collected Next States Shape: {next_state.shape}")
    print(f"Collected Rewards Shape: {reward.shape}")
    print(f"Collected Dones Shape: {done.shape}")

    print("\nSample of collected data:")
    if state.shape[0] > 0:
        print(f"First State: {state[0]}")
        print(f"First Action: {action[0]}")
        print(f"First Next State: {next_state[0]}")
        print(f"First Reward: {reward[0]}")
        print(f"First Done: {done[0]}")
    else:
        print("No data collected yet. Increase sleep time or episode length in DummyEnv.")
```
