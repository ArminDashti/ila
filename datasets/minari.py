import minari
import numpy as np



class Minari:
    def __init__(self, dataset_id, include_keys=None, normalize_rewards_flag=True):
        self.dataset_id = dataset_id
        self.include_keys = include_keys or [
            'observations', 'actions', 'rewards', 'terminations', 'truncations',
            'next_observations', 'next_actions', 'next_steps', 'steps',
            'prev_observations', 'prev_actions', 'prev_rewards',
            'prev_terminations', 'prev_truncations']
        self.normalize_rewards_flag = normalize_rewards_flag
        self.dataset = minari.load_dataset(dataset_id, True)
        self.data = {key: [] for key in self.include_keys}

    def _append_data(self, key, value):
        if key in self.include_keys:
            self.data[key].append(value)

    def _get_previous_value(self, values, index, default):
        return values[index - 1] if index > 0 else default

    def _process_episode(self, episode):
        episode_length = len(episode.observations)

        for i in range(min(100, episode_length - 1)):
            self._append_data('observations', episode.observations[i])
            self._append_data('actions', episode.actions[i])
            self._append_data('rewards', episode.rewards[i])
            self._append_data('terminations', episode.terminations[i])
            self._append_data('truncations', episode.truncations[i])
            self._append_data('next_observations', episode.observations[i + 1])

            next_act = episode.actions[i + 1] if i + 1 < episode_length else np.zeros_like(episode.actions[i])
            self._append_data('next_actions', next_act)
            self._append_data('next_steps', i + 2)
            self._append_data('steps', i + 1)

            prev_obs = self._get_previous_value(episode.observations, i, np.zeros_like(episode.observations[i]))
            self._append_data('prev_observations', prev_obs)

            prev_act = self._get_previous_value(episode.actions, i, np.zeros_like(episode.actions[i]))
            self._append_data('prev_actions', prev_act)

            prev_rew = self._get_previous_value(episode.rewards, i, 0.0)
            self._append_data('prev_rewards', prev_rew)

            prev_term = self._get_previous_value(episode.terminations, i, 0)
            self._append_data('prev_terminations', prev_term)

            prev_trunc = self._get_previous_value(episode.truncations, i, 0)
            self._append_data('prev_truncations', prev_trunc)

    def download(self):
        for episode in self.dataset.iterate_episodes():
            self._process_episode(episode)

        if 'rewards' in self.include_keys and self.normalize_rewards_flag:
            self.data['rewards'] = normalize_rewards(self.data['rewards'])

        if 'terminations' in self.include_keys and 'truncations' in self.include_keys:
            self.data['dones'] = np.logical_or(
                self.data['terminations'], self.data['truncations']
            ).astype(int)

        self.data = {key: np.array(value) for key, value in self.data.items()}

        return self.dataset, self.data
