import minari
import numpy as np

class Minari:
    def __init__(self, dataset_id, include_keys=None, normalize_rewards=True):
        self.dataset_id = dataset_id
        self.include_keys = include_keys or [
            'observations', 'actions', 'rewards', 'terminations', 'truncations',
            'next_observations', 'next_actions', 'steps',
            'prev_observations', 'prev_actions', 'prev_rewards',
            'prev_terminations', 'prev_truncations']
        self.normalize_rewards = normalize_rewards
        self.dataset = minari.load_dataset(dataset_id, download=True)
        self.processed_data = {key: [] for key in self.include_keys}

    def _append_to_processed_data(self, key, value):
        if key in self.include_keys:
            self.processed_data[key].append(value)

    def _get_previous_value(self, values, index, default):
        return values[index - 1] if index > 0 else default

    def _normalize_rewards(self, rewards):
        rewards = np.array(rewards)
        if rewards.size == 0:
            return rewards
        return (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))

    def _process_episode(self, episode):
        episode_length = len(episode.observations)

        for step_index in range(min(100, episode_length - 1)):
            self._append_to_processed_data('observations', episode.observations[step_index])
            self._append_to_processed_data('actions', episode.actions[step_index])
            self._append_to_processed_data('rewards', episode.rewards[step_index])
            self._append_to_processed_data('terminations', episode.terminations[step_index])
            self._append_to_processed_data('truncations', episode.truncations[step_index])
            self._append_to_processed_data('next_observations', episode.observations[step_index + 1])

            next_action = episode.actions[step_index + 1] if step_index + 1 < episode_length else np.zeros_like(episode.actions[step_index])
            self._append_to_processed_data('next_actions', next_action)
            self._append_to_processed_data('steps', step_index + 1)

            previous_observation = self._get_previous_value(episode.observations, step_index, np.zeros_like(episode.observations[step_index]))
            self._append_to_processed_data('prev_observations', previous_observation)

            previous_action = self._get_previous_value(episode.actions, step_index, np.zeros_like(episode.actions[step_index]))
            self._append_to_processed_data('prev_actions', previous_action)

            previous_reward = self._get_previous_value(episode.rewards, step_index, 0.0)
            if self.normalize_rewards:
                previous_reward = self._normalize_rewards([previous_reward])[0]
            self._append_to_processed_data('prev_rewards', previous_reward)

            previous_termination = self._get_previous_value(episode.terminations, step_index, 0)
            self._append_to_processed_data('prev_terminations', previous_termination)

            previous_truncation = self._get_previous_value(episode.truncations, step_index, 0)
            self._append_to_processed_data('prev_truncations', previous_truncation)

    def download(self):
        environment = self.dataset.recover_environment()
        return self.dataset, environment

    def download_processed(self):
        for episode in self.dataset.iterate_episodes():
            self._process_episode(episode)

        if 'rewards' in self.include_keys and self.normalize_rewards:
            self.processed_data['rewards'] = self._normalize_rewards(self.processed_data['rewards'])

        if 'terminations' in self.include_keys and 'truncations' in self.include_keys:
            self.processed_data['dones'] = np.logical_or(
                self.processed_data['terminations'], self.processed_data['truncations']
            ).astype(int)

        self.processed_data = {key: np.array(value) for key, value in self.processed_data.items()}

        environment = self.dataset.recover_environment()
        return self.processed_data, environment


#%%
m = Minari('D4RL/door/expert-v2')
v1, v2 = m.download_processed()
#%%
v2
