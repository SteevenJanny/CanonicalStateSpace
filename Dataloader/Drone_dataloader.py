import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


class Drone(Dataset):
    def __init__(self, split="train", frequency_subdiv=2, windows_length=100, sampling_mode="full"):
        super(Drone, self).__init__()

        self.dataloc = f"Data/Drone/"
        self.data = pickle.load(open(f'{self.dataloc}data.pkl', 'rb'))
        self.stats = pickle.load(open(f'{self.dataloc}stats.pkl', 'rb'))

        with open(f"Data/Drone/{split}.txt", 'r') as f:
            idx = [int(i) for i in f.readlines()]

        self.data = [self.data[i] for i in idx]

        self.frequency_subdiv = frequency_subdiv
        self.windows_length = windows_length
        self.sampling_mode = sampling_mode
        self.frequency = 100 // self.frequency_subdiv
        # dict_keys(['t', 'imu_ang_acc', 'motor', 'state', 'trajectory', 'orientation', 'speed'])
        splitted_data = []
        trajectory_length = 200
        for d in self.data:
            for i in range(0, d['state'][::self.frequency_subdiv].shape[0], trajectory_length):
                new_data = {}
                for k in d:
                    if k in ['imu_ang_acc', 'motor', 'state']:
                        new_data[k] = torch.from_numpy(d[k][::self.frequency_subdiv][i:i + trajectory_length]).float()
                    elif k == 't':
                        new_data[k] = torch.from_numpy(d[k][::self.frequency_subdiv][i:i + trajectory_length])
                    else:
                        new_data[k] = d[k]
                if new_data["state"].shape[0] == trajectory_length:
                    splitted_data.append(new_data)
        self.data = splitted_data
        self.cmd_size = 4
        self.obs_size = 6  # 7
        self.state_size = 13

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]

        # dict_keys(['t', 'imu_ang_acc', 'motor', 'state', 'trajectory', 'orientation', 'speed'])
        inputs = data["motor"]
        observations = data['imu_ang_acc']
        inputs, observations = self.normalize(inputs, observations)
        output = {'inputs': inputs,
                  'observations': observations,
                  'state': data['state']}
        return output

    def normalize(self, inputs, observations):
        mean, std = self.stats['motor'][0].to(inputs.device), self.stats['motor'][1].to(inputs.device)
        inputs = (inputs - mean) / std

        mean = self.stats['imu_ang_acc'][0].to(observations.device)
        std = self.stats['imu_ang_acc'][1].to(observations.device)
        observations = (observations - mean) / std
        return inputs, observations

    def denormalize(self, inputs=None, observations=None):
        if inputs is not None:
            mean, std = self.stats['motor'][0].to(inputs.device), self.stats['motor'][1].to(inputs.device)
            inputs = inputs * std + mean
        if observations is not None:
            mean = self.stats['imu_ang_acc'][0].to(observations.device)
            std = self.stats['imu_ang_acc'][1].to(observations.device)
            observations = observations * std + mean
        return inputs, observations
