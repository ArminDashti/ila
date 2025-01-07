from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import os
import numpy as np
import torch

class Gymnasium:
    def __init__(self, env="HalfCheetah-v5"):
        self.env = gym.make(env)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.accumulated_reward = 0

    def execute_action(self, action, reset_env=False):
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        action = np.array(action, dtype=self.env.action_space.dtype)
        if action.shape == (1, self.action_dim):
            action = action.flatten()
        results = {}
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.accumulated_reward += reward
        results['observation'] = observation
        results['current_reward'] = reward
        results['terminated'] = terminated
        results['truncated'] = truncated
        results['info'] = info
        results['accumulated_reward'] = self.accumulated_reward
        results['step'] = self.env._elapsed_steps

        if terminated or truncated:
            self.env.reset()

        return results




def evaluate(policy=None, env_name='HalfCheetah-v5', 
             render_mode='rgb_array', 
             seeds=[42], max_step=1000, 
             name_prefix='half_cheetah_run', 
             save_dir=r'C:\Users\Armin\step_aware'):
    
    os.makedirs(save_dir, exist_ok=True)

    def only_first_episode_trigger(episode_id):
        return episode_id == 0

    env = gym.make(env_name, render_mode=render_mode)
    env = RecordVideo(env, video_folder=save_dir, name_prefix=name_prefix, episode_trigger=only_first_episode_trigger)

    total_rewards = []

    for seed in seeds:
        observation, info = env.reset(seed=seed)
        episode_reward = 0

        for _ in range(max_step):
            if policy:
                action = policy(observation)
            else:
                action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        print(f"Reward for seed {seed}: {episode_reward}")

    env.close()

    average_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
    print(f"Average reward: {average_reward}")


