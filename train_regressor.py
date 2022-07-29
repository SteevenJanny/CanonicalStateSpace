import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.utils import MLP
from Dataloader.Drone_dataloader import Drone
from Dataloader.Tank_Dataloader import Tank
from Dataloader.Drone2D_dataloader import Drone_2D

plt.style.use("seaborn")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--lr', default=-4.0, type=float)
parser.add_argument('--window_size', default=5, type=int)
parser.add_argument('--forward', default=100, type=int)
parser.add_argument('--n_layers', default=3, type=int)
parser.add_argument('--units', default=256, type=int)
parser.add_argument('--dataset', default="Tank", type=str)
parser.add_argument('--name', default="", type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MSE = nn.MSELoss()
datasets = {'Tank': Tank, "Drone": Drone, "Drone2D": Drone_2D}


def evaluate():
    dataset = datasets[args.dataset](split="valid")
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False)

    cmd_size, obs_size = dataset.cmd_size, dataset.obs_size
    model = MLP((cmd_size + obs_size) * args.window_size, output_size=obs_size, hidden_size=args.units,
                n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(f"trained_models/regressor/{args.dataset}/{args.name}.nn", map_location=device))
    validate(model, dataloader)


def validate(model, dataloader):
    l = args.window_size
    with torch.no_grad():
        model.eval()
        for i, x in enumerate(dataloader):
            inputs = x["inputs"].to(device).float()
            observation = x["observations"].to(device).float()
            noise = torch.randn_like(observation) * torch.std(observation) / 1
            observation_noisy = observation + noise.to(device) * 0

            z = torch.cat([observation_noisy, inputs], dim=-1)
            z = torch.stack([z[:, i - l:i] for i in range(l, z.shape[1])], dim=1)
            B, P, T, S = z.shape
            y_regressor = observation[:, l - 1:-1] + model(z.view(B * P, T * S)).view(B, P, -1)

            start_time = 10
            z0 = z[:, start_time]
            y = [observation[:, start_time + l - 1:start_time + l]]
            length = 100
            for t in range(length):
                y_next = y[-1] + model(z0.view(B, T * S)).view(B, 1, -1)
                small_z = torch.cat([y_next, inputs[:, t + start_time + l:start_time + 1 + t + l]], dim=-1)
                z0 = torch.cat([z0[:, 1:], small_z], dim=1)
                y.append(y_next)

            y = torch.cat(y, dim=1)
            y_target = observation[:, start_time + l - 1:start_time + l + length]

            _, observation = dataloader.dataset.denormalize(observations=observation)
            _, y_regressor = dataloader.dataset.denormalize(observations=y_regressor)
            _, y = dataloader.dataset.denormalize(observations=y)
            _, y_target = dataloader.dataset.denormalize(observations=y_target)

            losses = get_loss(observation, y_regressor, y, y_target)
            print(losses)


def get_loss(observations, output, y, y_target):
    w = args.window_size

    loss_h0 = MSE(output, observations[:, w:])
    loss_rollout = MSE(y, y_target)
    loss = 10 * loss_h0 + loss_rollout
    return {'loss': loss, 'MSE_H0': loss_h0, 'MSE_Rollout': loss_rollout}


def main():
    print(args)

    train_dataset = datasets[args.dataset](split="train")
    valid_dataset = datasets[args.dataset](split="valid")

    batch_size = 100 if args.dataset == 'Drone' else 100
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    cmd_size, obs_size = train_dataset.cmd_size, train_dataset.obs_size
    model = MLP((cmd_size + obs_size) * args.window_size, output_size=obs_size, hidden_size=args.units,
                n_layers=args.n_layers, spectral=False).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=10 ** args.lr)
    l = args.window_size
    step = 0
    for epoch in range(args.epoch):
        model.train()

        for i, x in enumerate(train_dataloader):
            inputs = x["inputs"].to(device).float()
            observation = x["observations"].to(device).float()
            noise = torch.randn_like(observation)
            observation_noisy = observation + 0 * noise.to(device) / max([25, 100 - epoch / 100])

            z = torch.cat([observation_noisy, inputs], dim=-1)
            z = torch.stack([z[:, i - l:i] for i in range(l, z.shape[1])], dim=1)
            B, P, T, S = z.shape
            y_regressor = observation[:, l - 1:-1] + model(z.view(B * P, T * S)).view(B, P, -1)

            start_time = random.randint(0, z.shape[1] - args.forward - 1 -l)
            z0 = z[:, start_time]
            y = [observation[:, start_time + l - 1:start_time + l]]

            if args.dataset != 'Drone':
                length = epoch // 100
            else:
                length = min(step // 100, args.forward)

            # print(length)
            for t in range(length):
                y_next = y[-1] + model(z0.view(B, T * S)).view(B, 1, -1)
                small_z = torch.cat([y_next, inputs[:, start_time + l + t:start_time + l + t + 1]], dim=-1)
                z0 = torch.cat([z0[:, 1:], small_z], dim=1)
                y.append(y_next)
            y = torch.cat(y, dim=1)
            y_target = observation[:, start_time + l - 1:start_time + l + length]
            losses = get_loss(observation, y_regressor, y, y_target)


            optim.zero_grad()
            losses["loss"].backward()
            optim.step()
            step += 1

        validate(model, valid_dataloader)
        torch.save(model.state_dict(), f"../trained_models/regressor/{args.dataset}/{args.name}.nn")

if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
