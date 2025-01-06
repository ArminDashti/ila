from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os



# def evaluate_policy(env, policy, save_path, seeds=[42]):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     env.close()
#     total_rewards = []
#     env = RecordVideo(env, video_folder=save_path, episode_trigger=lambda episode_id: True)
#     policy.eval()
#     for seed in seeds:
#         state, _ = env.reset(seed=seed)
#         total_reward, done = 0.0, False
#         with torch.no_grad():
#             while not done:
#                 state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
#                 action = policy.sample_action(state_tensor)[0].cpu().numpy().flatten()
#                 state, reward, terminations, truncations, _ = env.step(action)
#                 total_reward += reward
#                 done = terminations or truncations
#         total_rewards.append(total_reward)
#         print('total_rewards',total_rewards)
#     avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
#     print(f"Average Reward over {len(seeds)} seeds: {avg_reward}")
#     env.close()
#     return avg_reward



def evaluate(policy=None, env_name='HalfCheetah-v4', 
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

evaluate()
