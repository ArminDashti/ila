from gymnasium.wrappers import RecordVideo
import torch
import numpy as np



def evaluate_policy(policy, device, env, save_path, seeds=None):
    """Evaluate the current policy over multiple seeds and return average reward."""
    env.close()
    if seeds is None:
        seeds = [42]
    total_rewards = []
    # Wrap the environment with RecordVideo once
    env = RecordVideo(env, video_folder=save_path, episode_trigger=lambda episode_id: True)
    policy.eval()
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        total_reward, done = 0.0, False
        with torch.no_grad():
            step = 0
            while not done:
                step += 1
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                step = torch.tensor([step], dtype=torch.long)
                action = policy.sample_action(state_tensor, step)[0].cpu().numpy().flatten()
                state, reward, terminations, truncations, _ = env.step(action)
                total_reward += reward
                done = terminations or truncations
        total_rewards.append(total_reward)
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    print(f"Average Reward over {len(seeds)} seeds: {avg_reward}")
    env.close()
    return avg_reward