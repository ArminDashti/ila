import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal



class GaussianPolicy(nn.Module):
    def __init__(self, layers, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_std = nn.Parameter(torch.zeros(layers[-1]))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        mean = self.layers[-1](x)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def sample_action(self, x, rsample=True, apply_tanh=True):
        mean, std = self.forward(x)
        normal = Normal(mean, std)
        
        if rsample:
            z = normal.rsample()
        else:
            z = normal.sample()
        
        if apply_tanh:
            action = torch.tanh(z)
        else:
            action = z
        
        if apply_tanh:
            # Reference: Appendix C of https://arxiv.org/pdf/1801.01290.pdf
            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        else:
            log_prob = normal.log_prob(z)
        
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

    def sample_deterministic_action(self, x, apply_tanh=True):
        mean, _ = self.forward(x)
        if apply_tanh:
            action = torch.tanh(mean)
        else:
            action = mean
        return action



if __name__ == "__main__":
    state_dim = 64 
    action_dim = 2 
    policy = GaussianPolicy([state_dim, 128, 256, 256, 128, action_dim])
    x = torch.randn(1, state_dim)
    action, log_prob = policy.sample_action(x)
    print("Sampled Action:", action)
    print("Log Probability:", log_prob)
    deterministic_action = policy.sample_deterministic_action(x)
    print("Deterministic Action:", deterministic_action)
