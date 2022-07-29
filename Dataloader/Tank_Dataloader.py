import os
import torch
from torch.utils.data import Dataset
import pickle
import random


class Tank(Dataset):
    def __init__(self, split="train"):
        super(Tank, self).__init__()

        self.dataloc = f"Data/Tank/{split}.pkl"

        self.data = pickle.load(open(self.dataloc, 'rb'))

        self.cmd_size = 1
        self.obs_size = 1
        self.state_size = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]

        inputs = data["command"]
        observation = data["observation"]

        inputs, observation = torch.tensor(inputs), torch.tensor(observation)
        inputs, observation = self.normalize(inputs, observation)
        output = {'inputs': inputs.reshape((-1, 1)),
                  'observations': observation}
        return output

    def craft_inputs(self, command, observation):
        k0 = 15
        observation = observation.view(-1, 1)
        command = command.view(-1, 1)

        Y = torch.stack([observation[k0 - i: - i] for i in range(1, self.na + 1)], dim=1)
        U = torch.stack([command[k0 - i:- i] for i in range(1, self.nb + 1)], dim=1)

        I = torch.cat([Y, U], dim=-1)

        O = observation[k0:]
        return I, O

    def normalize(self, inputs, observations):
        inputs = (inputs - torch.tensor([2.76462479])).to(inputs.device) / torch.tensor([1.30709925]).to(inputs.device)
        observations = (observations - torch.tensor([2.23151521])).to(observations.device) / torch.tensor(
            [1.33107377]).to(observations.device)
        return inputs, observations

    def denormalize(self, inputs=None, observations=None):
        if inputs is not None:
            inputs = inputs * torch.tensor([1.14607733]).to(inputs.device) + torch.tensor(
                [1.30709925]).to(inputs.device)
        if observations is not None:
            observations = observations * torch.tensor([1.33107377]).to(
                observations.device) + torch.tensor([2.23151521]).to(observations.device)
        return inputs, observations