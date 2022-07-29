import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Dataloader.Drone_dataloader import Drone
from Dataloader.Tank_Dataloader import Tank
from Dataloader.Drone2D_dataloader import Drone_2D
from models.utils import MLP
import sys
import os

plt.style.use("seaborn")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--lr', default=-3, type=float)
parser.add_argument('--window_size', default=20, type=int)
parser.add_argument('--compression', default=0.75, type=float)
parser.add_argument('--forward', default=100, type=int)
parser.add_argument('--n_layers', default=2, type=int)
parser.add_argument('--units', default=512, type=int)
parser.add_argument('--dataset', default='Drone2D', type=str)
parser.add_argument('--name', default='', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MSE = nn.MSELoss()
datasets = {'Tank': Tank, "Drone": Drone, "Drone2D": Drone_2D}


class Autoencoder(nn.Module):
    def __init__(self, obs_size, cmd_size):
        super(Autoencoder, self).__init__()
        self.z_size = obs_size + cmd_size

        state_size = int(np.floor(self.z_size * args.window_size * (1 - args.compression)) + 1)
        self.encoder = MLP(input_size=args.window_size * self.z_size,
                           output_size=state_size,
                           hidden_size=args.units,
                           n_layers=args.n_layers)
        self.decoder = MLP(input_size=state_size,
                           output_size=args.window_size * self.z_size,
                           hidden_size=args.units,
                           n_layers=args.n_layers)
        self.h0 = MLP((cmd_size + obs_size) * args.window_size, output_size=obs_size, hidden_size=256, n_layers=3).to(
            device)
        self.h0.load_state_dict(torch.load(f"trained_models/regressor/{args.dataset}/H0/{args.name}.nn",
                                               map_location=device))

    def forward(self, z):
        B, P, T, S = z.shape
        x = self.encoder(z.reshape(B, P, T * S))
        z_decoder = self.decoder(x)
        return x, z_decoder.reshape(B, P, T, S)


def evaluate():
    dataset = datasets[args.dataset](split="valid")
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False)
    cmd_size, obs_size = dataset.cmd_size, dataset.obs_size
    model = Autoencoder(obs_size, cmd_size).to(device)

    model.load_state_dict(torch.load(f"trained_models/regressor/{args.dataset}/{args.name}.nn", map_location=device))
    validate(model, dataloader)


def validate(model, dataloader):
    l = args.window_size
    with torch.no_grad():
        model.eval()
        for i, x in enumerate(dataloader):
            inputs = x["inputs"].to(device).float()
            observation = x["observations"].to(device).float()
            observation_noisy = observation

            z_groundtruth = torch.cat([observation_noisy, inputs], dim=-1)
            z_groundtruth = torch.stack([z_groundtruth[:, i - l:i] for i in range(l, z_groundtruth.shape[1])], dim=1)
            B, P, T, S = z_groundtruth.shape
            x_direct, z_decode = model(z_groundtruth)

            start_time = 10
            x_forward = [x_direct[:, start_time]]
            y_forward = [observation[:, l - 1 + start_time]]
            for t in range(args.forward):
                z = model.decoder(x_forward[-1]).view(B, T, S)
                y_next = y_forward[-1] + model.h0(z.view(B, T * S))
                small_z = torch.cat([y_next, inputs[:, start_time + l + t]], dim=-1).unsqueeze(-2)
                z_next = torch.cat([z[:, 1:], small_z], dim=1)

                xf = model.encoder(z_next.view(B, T * S))
                x_forward.append(xf)
                y_forward.append(y_next)

            x_forward = torch.stack(x_forward, dim=1)
            y_forward = torch.stack(y_forward, dim=1)

            _, observation = dataloader.dataset.denormalize(observations=observation)
            _, y_forward = dataloader.dataset.denormalize(observations=y_forward)

            losses = get_loss(z_groundtruth, z_decode, x_direct,
                              x_forward, y_forward, observation, start_time=start_time)
            print(losses)


def get_loss(z_groundtruth, z_decode,
             x_direct, x_forward,
             y_forward, observation,
             start_time):
    loss_ae = MSE(z_groundtruth, z_decode)
    l = args.window_size
    n = y_forward.shape[1]
    loss_rollout_x = MSE(x_direct[:, start_time:start_time + n], x_forward)
    loss_rollout_y = MSE(observation[:, l - 1 + start_time:l - 1 + start_time + n], y_forward)
    loss = loss_ae + loss_rollout_x + loss_rollout_y
    return {'loss': loss, 'MSE_AE': loss_ae, 'MSE_Rollout_X': loss_rollout_x, 'MSE_Rollout_y': loss_rollout_y}


def main():
    print(args)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    mode = "online"
    if sys.platform == "darwin":
        mode = "disabled"

    train_dataset = datasets[args.dataset](split="train")
    valid_dataset = datasets[args.dataset](split="valid")

    batch_size = 50 if args.dataset == 'Drone2D' else 250
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    cmd_size, obs_size = train_dataset.cmd_size, train_dataset.obs_size
    model = Autoencoder(obs_size, cmd_size).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=10 ** args.lr)
    l = args.window_size
    for epoch in range(args.epoch):
        model.train()
        if epoch < args.epoch * 0.75:
            for p in model.h0.parameters():
                p.requires_grad = \
                    False
        else:
            for p in model.h0.parameters():
                p.requires_grad = True

        for i, x_direct in enumerate(train_dataloader):
            inputs = x_direct["inputs"].to(device).float()
            observation = x_direct["observations"].to(device).float()
            observation_noisy = observation

            z_groundtruth = torch.cat([observation_noisy, inputs], dim=-1)
            z_groundtruth = torch.stack([z_groundtruth[:, i - l:i] for i in range(l, z_groundtruth.shape[1])], dim=1)
            B, P, T, S = z_groundtruth.shape

            x_direct, z_decode = model(z_groundtruth)

            start_time = random.randint(0, P - args.forward - 1)

            x_forward = [x_direct[:, start_time]]
            y_forward = [observation[:, l - 1 + start_time]]

            length = min(int(epoch / (args.epoch * 0.75) * 100), 100)

            for t in range(length):
                z = model.decoder(x_forward[-1]).view(B, T, S)
                y_next = y_forward[-1] + model.h0(z.view(B, T * S))
                small_z = torch.cat([y_next, inputs[:, start_time + l + t]], dim=-1).unsqueeze(-2)
                z_next = torch.cat([z[:, 1:], small_z], dim=1)

                xf = model.encoder(z_next.view(B, T * S))
                x_forward.append(xf)
                y_forward.append(y_next)

            x_forward = torch.stack(x_forward, dim=1)
            y_forward = torch.stack(y_forward, dim=1)

            losses = get_loss(z_groundtruth, z_decode,
                              x_direct, x_forward,
                              y_forward, observation, start_time=start_time)

            optim.zero_grad()
            losses["loss"].backward()
            optim.step()
        validate(model, valid_dataloader)
        torch.save(model.state_dict(), f"../trained_models/regressor/{args.dataset}/{args.name}.nn")

if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
