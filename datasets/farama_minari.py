#%%
import minari
import numpy as np

class Dataset:
    def __init__(self, dataset_id, include_keys=None):
        self.dataset_id = dataset_id
        self.include_keys = include_keys or [
            'states', 'actions', 'rewards', 'terminations', 'truncations',
            'next_states', 'next_actions', 'steps', 'next_steps', 'prev_steps',
            'prev_states', 'prev_actions', 'prev_rewards',
            'prev_terminations', 'prev_truncations', 'past_returns', 'future_returns']
        self.dataset = minari.load_dataset(dataset_id, download=True)
        self.processed_data = {key: [] for key in self.include_keys}
        self._is_processed = False

    def _append_to_processed_data(self, key, value):
        if key in self.include_keys:
            self.processed_data[key].append(value)

    @staticmethod
    def _get_previous_value(values, index, default):
        return values[index - 1] if index > 0 else default

    def _process_episode(self, episode):
        actions_length = len(episode.actions)
        observations_length = len(episode.observations)
        assert observations_length == actions_length + 1, (
            f"Expected observations_length to be actions_length + 1, "
            f"got observations_length={observations_length}, actions_length={actions_length}"
        )
        rewards = episode.rewards
        past_returns = np.cumsum(rewards)
        future_returns = np.array([np.sum(rewards[i:]) for i in range(len(rewards))])
        for step_index in range(actions_length):
            self._append_to_processed_data('states', episode.observations[step_index])
            self._append_to_processed_data('actions', episode.actions[step_index])
            self._append_to_processed_data('rewards', rewards[step_index])
            self._append_to_processed_data('terminations', episode.terminations[step_index])
            self._append_to_processed_data('truncations', episode.truncations[step_index])

            next_state = episode.observations[step_index + 1]
            self._append_to_processed_data('next_states', next_state)

            if step_index + 1 < actions_length:
                next_action = episode.actions[step_index + 1]
            else:
                next_action = np.zeros_like(episode.actions[step_index])
            self._append_to_processed_data('next_actions', next_action)

            self._append_to_processed_data('steps', step_index + 1)

            prev_state = self._get_previous_value(
                episode.observations, step_index,
                default=np.zeros_like(episode.observations[step_index])
            )
            self._append_to_processed_data('prev_states', prev_state)

            prev_action = self._get_previous_value(
                episode.actions, step_index,
                default=np.zeros_like(episode.actions[step_index])
            )
            self._append_to_processed_data('prev_actions', prev_action)

            prev_reward = self._get_previous_value(
                rewards, step_index, default=0.0
            )
            self._append_to_processed_data('prev_rewards', prev_reward)

            prev_termin = self._get_previous_value(
                episode.terminations, step_index, default=0
            )
            self._append_to_processed_data('prev_terminations', prev_termin)

            prev_trunc = self._get_previous_value(
                episode.truncations, step_index, default=0
            )
            self._append_to_processed_data('prev_truncations', prev_trunc)

            next_step = step_index + 1
            self._append_to_processed_data('next_steps', next_step)

            prev_step = step_index - 1 if step_index > 0 else None
            self._append_to_processed_data('prev_steps', prev_step)

            self._append_to_processed_data('past_returns', past_returns[step_index])
            self._append_to_processed_data('future_returns', future_returns[step_index])

    def download(self):
        return self.dataset

    def download_processed(self):
        for episode in self.dataset.iterate_episodes():
            self._process_episode(episode)

        if 'terminations' in self.include_keys and 'truncations' in self.include_keys:
            self.processed_data['dones'] = np.logical_or(
                self.processed_data['terminations'], self.processed_data['truncations']
            ).astype(int)

        self.processed_data = {key: np.array(value) for key, value in self.processed_data.items()}
        self._is_processed = True

    def dict_data(self):
        if not self._is_processed:
            raise RuntimeError("Data has not been processed yet. Call `download_processed()` first.")
        return self.processed_data

    def env(self):
        return self.dataset

    @property
    def data(self):
        return self.dataset

    @property
    def dataset_shapes(self):
        if not self._is_processed:
            raise RuntimeError("Data has not been processed yet. Call `download_processed()` first.")
        return {key: value.shape for key, value in self.processed_data.items()}

    @property
    def dataset_ranges(self):
        if not self._is_processed:
            raise RuntimeError("Data has not been processed yet. Call `download_processed()` first.")
        ranges = {}
        for key, value in self.processed_data.items():
            array = np.array(value)
            if array.size > 0 and np.issubdtype(array.dtype, np.number):
                ranges[key] = (np.min(array), np.max(array))
        return ranges

# Function to normalize a specific key in a processed dataset and add it as a new key
def add_normalized_key(processed_data, key):
    if key not in processed_data:
        raise KeyError(f"Key '{key}' not found in the processed data.")
    values = np.array(processed_data[key])
    min_value = np.min(values)
    max_value = np.max(values)
    if max_value - min_value < 1e-8:
        normalized_values = np.zeros_like(values)
    else:
        normalized_values = (values - min_value) / (max_value - min_value)
    processed_data[f"normalized_{key}"] = normalized_values
    return processed_data


minari_instance = Dataset('D4RL/door/expert-v2')
dsp = minari_instance.download_processed()
#%%
processed_data = minari_instance.dict_data()
raw_data = minari_instance.data
environment = minari_instance.env()
shapes = minari_instance.dataset_shapes
ranges = minari_instance.dataset_ranges
#%%
# Example usage of the function
processed_data = add_normalized_key(processed_data, 'rewards')
#%%
