import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, activation=nn.ReLU,
                 spectral=False):
        super(MLP, self).__init__()
        f = []
        for i in range(n_layers):
            if spectral:
                f.append(spectral_norm(nn.Linear(hidden_size if i != 0 else input_size, hidden_size)))
            else:
                f.append(nn.Linear(hidden_size if i != 0 else input_size, hidden_size))
            f.append(activation())
        f.append(nn.Linear(hidden_size if n_layers != 0 else input_size, output_size))
        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)
