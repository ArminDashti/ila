import minari
import numpy as np


def add_nexts(data, keys):
    for key in keys:
        if key in data:
            shifted_data = np.empty_like(data[key], dtype=float)
            shifted_data[:-1] = data[key][1:]
            shifted_data[-1] = None
            data[f"next_{key}"] = shifted_data
    return data

def add_prevs(data, keys):
    for key in keys:
        if key in data:
            shifted_data = np.empty_like(data[key], dtype=object)
            shifted_data[1:] = data[key][:-1]
            shifted_data[0] = None
            data[f"prev_{key}"] = shifted_data
    return data

def add_accumulated_rewards(data):
    if 'rewards' not in data:
        raise KeyError("Key 'rewards' not found in the data.")
    data['accumulated_rewards'] = []
    accumulated_reward = 0.0
    for reward in data['rewards']:
        accumulated_reward += reward
        data['accumulated_rewards'].append(accumulated_reward)
    return data

class Dataset:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.dataset = minari.load_dataset(dataset_id, download=True)
        self.processed_data = {}
        self._process_all_data()

    def _add_to_processed_data(self, key, value):
        if key not in self.processed_data:
            self.processed_data[key] = []
        self.processed_data[key].append(value)

    def _process_single_episode(self, episode):
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
            self._add_to_processed_data('states', episode.observations[step_index])
            self._add_to_processed_data('actions', episode.actions[step_index])
            self._add_to_processed_data('rewards', rewards[step_index])
            self._add_to_processed_data('terminations', episode.terminations[step_index])
            self._add_to_processed_data('truncations', episode.truncations[step_index])
            self._add_to_processed_data('steps', step_index + 1)
            self._add_to_processed_data('past_returns', past_returns[step_index])
            self._add_to_processed_data('future_returns', future_returns[step_index])

    def _process_all_data(self):
        for episode in self.dataset.iterate_episodes():
            self._process_single_episode(episode)

        if 'terminations' in self.processed_data and 'truncations' in self.processed_data:
            self.processed_data['dones'] = np.logical_or(
                self.processed_data['terminations'], self.processed_data['truncations']
            ).astype(int)

        self.processed_data = {key: np.array(value) for key, value in self.processed_data.items()}

    def to_dict(self):
        return self.processed_data

    def env(self):
        return self.dataset.recover_environment(render_mode='rgb_array')

    @property
    def minari_instance(self):
        return self.dataset

    @property
    def shapes(self):
        return {key: value.shape for key, value in self.processed_data.items()}

    @property
    def ranges(self):
        ranges = {}
        for key, value in self.processed_data.items():
            array = np.array(value)
            if array.size > 0 and np.issubdtype(array.dtype, np.number):
                ranges[key] = (np.min(array), np.max(array))
        return ranges

# Function to normalize specific keys in a processed dataset and add them as new keys
def normalize(processed_data, keys, replace=False):
    for key in keys:
        if key not in processed_data:
            raise KeyError(f"Key '{key}' not found in the processed data.")
        values = np.array(processed_data[key])
        min_value = np.min(values)
        max_value = np.max(values)
        if max_value - min_value < 1e-8:
            normalized_values = np.zeros_like(values)
        else:
            normalized_values = (values - min_value) / (max_value - min_value)
        if replace:
            processed_data[key] = normalized_values
        else:
            processed_data[f"normalized_{key}"] = normalized_values
    return processed_data





if __name__ == '__main__':
    minari_instance = Dataset('D4RL/door/human-v2')
    data = minari_instance.to_dict()
    data = add_nexts(data, ['states'])
    print(data.keys())

