import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)
    
class GridEnvironment:
    def __init__(self):
        self.grid_size = 5
        set_seed(42)
        self.obstacles = [(1, 1), (2, 2), (3, 3)]
        self.goal = (4, 4)
        self.max_steps = 50
        self.current_step = 0

        self.env = gym.Env()
        self.env.action_space = spaces.Discrete(4)
        self.env.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        while True:
            self.agent_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
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

    def render(self):
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

    def render_video(self, path="env_animation.mp4"):
        fig, ax = plt.subplots(figsize=(5, 5))

        def update(frame):
            ax.clear()
            grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            for (ox, oy) in self.obstacles:
                grid[self.grid_size - 1 - oy][ox] = 'X'
            gx, gy = self.goal
            grid[self.grid_size - 1 - gy][gx] = 'G'
            ax, ay = frame
            grid[self.grid_size - 1 - ay][ax] = 'A'

            ax.set_xticks(range(self.grid_size + 1))
            ax.set_yticks(range(self.grid_size + 1))
            ax.set_xticks([x - 0.5 for x in range(1, self.grid_size)], minor=True)
            ax.set_yticks([y - 0.5 for y in range(1, self.grid_size)], minor=True)
            ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.5)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    content = grid[self.grid_size - 1 - j][i]
                    if content != ' ':
                        ax.text(i, j, content, ha='center', va='center', fontsize=14)

        positions = [(self.agent_pos[0], self.agent_pos[1])]

        for _ in range(self.max_steps):
            action = self.env.action_space.sample()
            next_pos, _, done, _, _ = self.step(action)
            positions.append(next_pos)
            if done:
                break

        ani = FuncAnimation(fig, update, frames=positions, repeat=False)
        ani.save(path, writer="ffmpeg")
        print(f"Video saved to {path}")

    def collect_dataset(self, num_episodes=1000, policy='random', render=False):
        dataset = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }

        for episode in range(num_episodes):
            state, info = self.reset()
            done = False
            truncated = False

            while not (done or truncated):
                if policy == 'random':
                    action = self.env.action_space.sample()
                elif policy == 'greedy':
                    action = self.greedy_policy(state)
                else:
                    raise ValueError("Unsupported policy type")

                next_state, reward, terminated, truncated, info = self.step(action)
                done = terminated or truncated

                dataset['states'].append(state)
                dataset['actions'].append(action)
                dataset['rewards'].append(reward)
                dataset['next_states'].append(next_state)
                dataset['dones'].append(done)

                if render:
                    self.render()

                state = next_state

            if (episode + 1) % 100 == 0:
                print(f"Completed {episode + 1}/{num_episodes} episodes.")

        for key in dataset:
            dataset[key] = np.array(dataset[key])

        return dataset

    def greedy_policy(self, state):
        x, y = state
        goal_x, goal_y = self.goal

        actions = []
        if y < goal_y and y < self.grid_size - 1:
            actions.append(0)
        if y > goal_y and y > 0:
            actions.append(1)
        if x > goal_x and x > 0:
            actions.append(2)
        if x < goal_x and x < self.grid_size - 1:
            actions.append(3)

        if actions:
            return random.choice(actions)
        else:
            return self.env.action_space.sample()

    def save_datasets(self, dataset_random, dataset_greedy):
        output_dir = 'datasets'
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(os.path.join(output_dir, 'dataset_random.npz'), **dataset_random)
        np.savez_compressed(os.path.join(output_dir, 'dataset_greedy.npz'), **dataset_greedy)
        print(f"\nDatasets saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    env = GridEnvironment()
    dataset_random = env.collect_dataset(num_episodes=1000, policy='random', render=False)
    dataset_greedy = env.collect_dataset(num_episodes=1000, policy='greedy', render=False)
    env.render_video()
