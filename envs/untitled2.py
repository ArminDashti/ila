import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ila.envs import grid
import numpy as np
import random
import sys


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OfflineRLDataset(Dataset):
    def __init__(self, data):
        self.states = torch.tensor(data['states'], dtype=torch.float32)
        self.actions = torch.tensor(data['actions'], dtype=torch.float32)
        self.rewards = torch.tensor(data['rewards'], dtype=torch.float32)
        self.next_states = torch.tensor(data['next_states'], dtype=torch.float32)
        self.dones = torch.tensor(data['dones'], dtype=torch.float32)

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def train_offline_rl(data, state_dim, action_dim, max_action, epochs=50, batch_size=64, lr=1e-3, gamma=0.99):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OfflineRLDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic = Critic(state_dim, action_dim).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        for states, actions, rewards, next_states, dones in dataloader:
            states = states.to(device)
            actions = actions.to(device).unsqueeze(1)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            with torch.no_grad():
                next_actions = actor(next_states)
                target_q = rewards + gamma * critic(next_states, next_actions) * (1 - dones)

            current_q = critic(states, actions)

            critic_loss = criterion(current_q, target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            epoch_critic_loss += critic_loss.item()

            # Update Actor
            actor_loss = -critic(states, actor(states)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            epoch_actor_loss += actor_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Actor Loss: {epoch_actor_loss:.4f}, Critic Loss: {epoch_critic_loss:.4f}")

    return actor, critic

set_seed(42)
env = grid.GridEnvironment()

ds = env.collect_dataset(num_episodes=10000, policy='random', render=False)
train_offline_rl(ds, 2, 1, 3)
