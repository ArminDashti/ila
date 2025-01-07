import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueFunction(nn.Module):
    def __init__(self, layers):
        super(Value, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
    
    def forward(self, state, action=None):
        x = state
        if action is not None:
            x = torch.cat([x, action], dim=-1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        value = self.layers[-1](x)
        return value
