import minari
import numpy as np

class Minari:
    def __init__(self, dataset_id, include_keys=None, normalize_rewards=True):
        self.dataset_id = dataset_id
        self.include_keys = include_keys or [
            'states', 'actions', 'rewards', 'termins', 'truncs',
            'next_states', 'next_actions', 'steps', 'next_steps', 'prev_steps',
            'prev_states', 'prev_actions', 'prev_rewards',
            'prev_termins', 'prev_truncs']
        self.normalize_rewards = normalize_rewards
        self.dataset = minari.load_dataset(dataset_id, download=True)
        self.processed_data = {key: [] for key in self.include_keys}

    def _append_to_processed_data(self, key, value):
        if key in self.include_keys:
            self.processed_data[key].append(value)

    @staticmethod
    def _get_previous_value(values, index, default):
        return values[index - 1] if index > 0 else default

    @staticmethod
    def _normalize_rewards(rewards):
        rewards = np.array(rewards)
        if rewards.size == 0:
            return rewards
        return (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))

    def _process_episode(self, episode):
        episode_length = len(episode.observations)

        for step_index in range(min(100, episode_length - 1)):
            self._append_to_processed_data('states', episode.observations[step_index])
            self._append_to_processed_data('actions', episode.actions[step_index])
            self._append_to_processed_data('rewards', episode.rewards[step_index])
            self._append_to_processed_data('termins', episode.terminations[step_index])
            self._append_to_processed_data('truncs', episode.truncations[step_index])
            self._append_to_processed_data('next_states', episode.observations[step_index + 1])

            next_action = episode.actions[step_index + 1] if step_index + 1 < episode_length else np.zeros_like(episode.actions[step_index])
            self._append_to_processed_data('next_actions', next_action)
            self._append_to_processed_data('steps', step_index + 1)

            prev_state = self._get_previous_value(episode.observations, step_index, np.zeros_like(episode.observations[step_index]))
            self._append_to_processed_data('prev_states', prev_state)

            prev_action = self._get_previous_value(episode.actions, step_index, np.zeros_like(episode.actions[step_index]))
            self._append_to_processed_data('prev_actions', prev_action)

            prev_reward = self._get_previous_value(episode.rewards, step_index, 0.0)
            if self.normalize_rewards:
                prev_reward = self._normalize_rewards([prev_reward])[0]
            self._append_to_processed_data('prev_rewards', prev_reward)

            prev_termin = self._get_previous_value(episode.terminations, step_index, 0)
            self._append_to_processed_data('prev_termins', prev_termin)

            prev_trunc = self._get_previous_value(episode.truncations, step_index, 0)
            self._append_to_processed_data('prev_truncs', prev_trunc)

            next_step = step_index + 1 if step_index + 1 < episode_length else None
            prev_step = step_index if step_index > 0 else None

            self._append_to_processed_data('next_steps', next_step)
            self._append_to_processed_data('prev_steps', prev_step)

    def download(self):
        return self.dataset

    def download_processed(self):
        for episode in self.dataset.iterate_episodes():
            self._process_episode(episode)

        if 'rewards' in self.include_keys and self.normalize_rewards:
            self.processed_data['rewards'] = self._normalize_rewards(self.processed_data['rewards'])

        if 'termins' in self.include_keys and 'truncs' in self.include_keys:
            self.processed_data['dones'] = np.logical_or(
                self.processed_data['termins'], self.processed_data['truncs']
            ).astype(int)

        self.processed_data = {key: np.array(value) for key, value in self.processed_data.items()}

        return self.processed_data

    def env(self):
        return self.dataset.recover_environment()

if __name__ == "__main__":
    minari_instance = Minari('D4RL/door/expert-v2')
    processed_data = minari_instance.download_processed()
    print(processed_data['prev_states'][1] == processed_data['states'][0])
