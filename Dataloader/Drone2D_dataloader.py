import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


class Drone_2D(Dataset):
    def __init__(self, split="train"):
        super(Drone_2D, self).__init__()

        self.dataloc = f"Data/Drone2D/{split}.pkl"
        self.data = pickle.load(open(self.dataloc, 'rb'))

        self.cmd_size = 2
        self.obs_size = 3
        self.state_size = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]

        inputs = data["command"]
        observation = data["observation"]

        inputs, observation = torch.tensor(inputs), torch.tensor(observation)
        inputs, observation = self.normalize(inputs, observation)
        output = {'inputs': inputs,
                  'observations': observation}
        return output

    def normalize(self, inputs, observations):
        inputs = (inputs - torch.tensor([110.86366, 110.86604])).to(inputs.device) / torch.tensor(
            [4.1976333, 4.195533]).to(inputs.device)
        observations = (observations - torch.tensor([-0.00282778, -0.02173044, 0.00027592])).to(
            observations.device) / torch.tensor(
            [1.1263357, 1.1627737, 0.08194371]).to(observations.device)
        return inputs, observations

    def denormalize(self, inputs=None, observations=None):
        if inputs is not None:
            inputs = inputs * torch.tensor([4.1976333, 4.195533]).to(inputs.device) + torch.tensor(
                [110.86366, 110.86604]).to(inputs.device)
        if observations is not None:
            observations = observations * torch.tensor([1.1263357, 1.1627737, 0.08194371]).to(
                observations.device) + torch.tensor(
                [-0.00282778, -0.02173044, 0.00027592]).to(observations.device)
        return inputs, observations