import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os

class Grid(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(Grid, self).__init__()
        self.grid_size = 5
        self.obstacles = [(1, 1), (2, 2), (3, 3)]
        self.goal = (4, 4)
        self.max_steps = 50
        self.current_step = 0
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, 
                                            shape=(2,), dtype=np.int32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        while True:
            self.agent_pos = (np.random.randint(0, self.grid_size),
                              np.random.randint(0, self.grid_size))
            if self.agent_pos not in self.obstacles and self.agent_pos != self.goal:
                break
        return np.array(self.agent_pos), {}
    
    def step(self, action):
        self.current_step += 1
        x, y = self.agent_pos
        
        if action == 0 and y < self.grid_size - 1:
            y += 1
        elif action == 1 and y > 0:
            y -= 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.grid_size - 1:
            x += 1
        
        self.agent_pos = (x, y)
        
        done = False
        reward = -1
        
        if self.agent_pos == self.goal:
            reward = 10
            done = True
        elif self.agent_pos in self.obstacles:
            reward = -10
            done = True
        elif self.current_step >= self.max_steps:
            done = True
        
        terminated = done
        truncated = False
        return np.array(self.agent_pos), reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for (ox, oy) in self.obstacles:
            grid[self.grid_size - 1 - oy][ox] = 'X'
        gx, gy = self.goal
        grid[self.grid_size - 1 - gy][gx] = 'G'
        ax, ay = self.agent_pos
        grid[self.grid_size - 1 - ay][ax] = 'A'
        
        print("-" * (self.grid_size * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("-" * (self.grid_size * 2 + 1))

def collect_dataset(env, num_episodes=1000, policy='random', render=False):
    dataset = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': []
    }

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            if policy == 'random':
                action = env.action_space.sample()
            elif policy == 'greedy':
                action = greedy_policy(state, env.goal, env.grid_size)
            else:
                raise ValueError("Unsupported policy type")

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            dataset['states'].append(state)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['next_states'].append(next_state)
            dataset['dones'].append(done)

            if render:
                env.render()

            state = next_state

        if (episode + 1) % 100 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes.")

    for key in dataset:
        dataset[key] = np.array(dataset[key])

    return dataset

def greedy_policy(state, goal, grid_size):
    x, y = state
    goal_x, goal_y = goal

    actions = []
    if y < goal_y and y < grid_size - 1:
        actions.append(0)
    if y > goal_y and y > 0:
        actions.append(1)
    if x > goal_x and x > 0:
        actions.append(2)
    if x < goal_x and x < grid_size - 1:
        actions.append(3)

    if actions:
        return random.choice(actions)
    else:
        return env.action_space.sample()

if __name__ == "__main__":
    env = Grid()

    print("Collecting dataset using a random policy...")
    dataset_random = collect_dataset(env, num_episodes=1000, policy='random', render=False)

    print("Collecting dataset using a greedy policy...")
    dataset_greedy = collect_dataset(env, num_episodes=1000, policy='greedy', render=False)

    print("\nRandom Policy Dataset:")
    print(f"Total transitions: {len(dataset_random['states'])}")
    print(f"Sample transition: State={dataset_random['states'][0]}, Action={dataset_random['actions'][0]}, "
          f"Reward={dataset_random['rewards'][0]}, Next_State={dataset_random['next_states'][0]}, Done={dataset_random['dones'][0]}")

    print("\nGreedy Policy Dataset:")
    print(f"Total transitions: {len(dataset_greedy['states'])}")
    print(f"Sample transition: State={dataset_greedy['states'][0]}, Action={dataset_greedy['actions'][0]}, "
          f"Reward={dataset_greedy['rewards'][0]}, Next_State={dataset_greedy['next_states'][0]}, Done={dataset_greedy['dones'][0]}")

    output_dir = 'datasets'
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, 'dataset_random.npz'), **dataset_random)
    np.savez_compressed(os.path.join(output_dir, 'dataset_greedy.npz'), **dataset_greedy)
    print(f"\nDatasets saved to the '{output_dir}' directory.")
