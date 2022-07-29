import pickle
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import torch
import matplotlib.pyplot as plt
import numpy as np
import random

plt.style.use("seaborn")

G = 9.81
MASS = 1
Kthrust = 4e-4
MAX_SPEED = 7000 * np.pi / 60
DRAG = 1e-9
L = 0.15
A = 0.1
J = MASS * (L ** 2 + A ** 2) / 12
DT = 1 / 30

MODE = "train"

COMMAND_HORIZON = 5
PREDICTION_HORIZON = 10


def simulate(state, command, dt, accurate=True):
    x, z, theta, dx, dz, DTheta = state
    ## LINEAR PFD
    propeller_force = Kthrust / MASS * torch.sum(command ** 2)
    drag_force = -DRAG * torch.sum(command) / MASS
    ddx = -propeller_force * torch.sin(theta) + drag_force * dx * (accurate == True)
    ddz = propeller_force * torch.cos(theta) - G + drag_force * dz * (accurate == True)

    ## ANGULAR PFD
    dDTheta = (1 / J) * Kthrust * L * (command[1] ** 2 - command[0] ** 2)

    ## INTEGRATION
    next_dx = dx + dt * ddx
    next_dz = dz + dt * ddz
    next_DTheta = DTheta + dt * dDTheta

    next_x = x + dt * dx
    next_z = z + dt * dz
    next_theta = (theta + dt * DTheta)

    return torch.stack([next_x, next_z, next_theta, next_dx, next_dz, next_DTheta], dim=-1)


def compute_command_mpc(x0, command, target, t, dt, state_weights, command_weights):
    target = target[t:t + PREDICTION_HORIZON]

    def cost(u):
        states = [x0]
        for t in range(target.shape[0]):
            cmd_time = t if t < COMMAND_HORIZON else -1
            next_state = simulate(states[-1], u[cmd_time], dt=dt, accurate=False)
            states.append(next_state)
        states = torch.stack(states[1:], dim=0)
        error_states = torch.sum(state_weights * (target - states).sum(0) ** 2)
        error_command = command_weights * command.sum(0)
        cost = error_states.sum() + error_command.sum()
        return cost

    LR = 1
    command.requires_grad = True
    error = np.inf
    cpt = 0
    while error > 100 and cpt < 1:
        error = cost(command)
        error.backward()
        command = command - LR * command.grad
        command = command.detach()
        command.requires_grad = True
        cpt += 1
    # print(error, cpt)
    return command.detach()


def visualize(states, command, target_trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    t = 0
    X = np.array([[L, A / 2, 0], [L, -A / 2, 0], [-L, -A / 2, 0], [-L, A / 2, 0], [L, A / 2, 0]])
    r = Rotation.from_euler('zyx', [states[0, 2], 0, 0])
    X = r.apply(X) + states[t, :3]
    patch = patches.Polygon(X[:, :2], closed=True, fc='r', ec='r', alpha=0.5)

    ax.plot(target_trajectory[:, 0], target_trajectory[:, 1])
    ax.plot(states[:, 0], states[:, 1])
    ax.add_patch(patch)

    m = np.array([[-L, 0, 0], [-L, -command[t, 0] / 500, 0]])
    m = r.apply(m) + states[t, :3]
    m1, = ax.plot(m[:, 0], m[:, 1])

    m = np.array([[L, 0, 0], [L, -command[t, 1] / 500, 0]])
    m = r.apply(m) + states[t, :3]
    m2, = ax.plot(m[:, 0], m[:, 1])

    # plt.show()

    def init():
        return patch, m1, m2

    def animate(i):
        X = np.array([[L, A / 2, 0], [L, -A / 2, 0], [-L, -A / 2, 0], [-L, A / 2, 0], [L, A / 2, 0]])
        r = Rotation.from_euler('zyx', [states[i, 2], 0, 0])
        X = r.apply(X) + states[i, :3]

        patch.set_xy(X[:, :2])

        m = np.array([[-L, 0, 0], [-L, -command[i, 0] / 500, 0]])
        m = r.apply(m) + states[i, :3]
        m1.set_data(m[:, :2].transpose())

        m = np.array([[L, 0, 0], [L, -command[i, 1] / 500, 0]])
        m = r.apply(m) + states[i, :3]
        m2.set_data(m[:, :2].transpose())

        return patch, m1, m2

    ani = animation.FuncAnimation(fig, animate, states.shape[0] - 1, init_func=init,
                                  interval=DT * 1000, blit=True, repeat=True)
    plt.show()


def complete_traj(t, target_trajectory):
    t = np.concatenate([t, [t[-1] * 10]], axis=0)
    target_trajectory = np.concatenate([target_trajectory, [target_trajectory[-1]] * 10], axis=0)
    return t, target_trajectory


def procedural_traj(dt, T, N=10):
    waypoints = 2 * (2 * np.random.rand(N, 2) - 1)
    t = np.linspace(0, T, N)

    time = np.arange(0, T, dt)
    target = interp1d(t, waypoints, axis=0, kind='cubic')(time)

    pose_target = np.zeros((time.shape[0], 3))
    pose_target[:, :2] = target

    speed_target = np.concatenate([(pose_target[1:] - pose_target[:-1]) / dt, np.zeros((1, 3))], axis=0)
    state_target = np.concatenate([pose_target, speed_target], axis=-1)

    return time, state_target


if __name__ == '__main__':
    splits = {'train': 500, 'test': 20, 'valid': 20}

    data = []
    while len(data) < splits[MODE]:
        loose_traj = random.random() < 0.2 and MODE == "train"
        t, target_trajectory = procedural_traj(DT, 20, N=random.randint(5, 10))
        t, target_trajectory = complete_traj(t, target_trajectory)
        t, target_trajectory = torch.from_numpy(t), torch.from_numpy(target_trajectory)

        command = torch.ones(COMMAND_HORIZON, 2) * np.sqrt(G * MASS / (2 * Kthrust))
        states = [target_trajectory[0]]

        state_weights = torch.tensor([5, 5, 0.0, 1, 1, 0.01])
        cmd_weights = torch.tensor([0.0, 0.0])

        list_command = [command[0]]
        clear_cmd = [command[0]]
        for t in range(t.shape[0] - 1):
            if loose_traj is not True:
                command = compute_command_mpc(states[-1], command, target_trajectory, t=t, dt=DT,
                                              state_weights=state_weights, command_weights=cmd_weights)

            else:
                if random.random() < 0.1:
                    command = command + np.random.randn(*command[0].shape) * 10
                else:
                    command = compute_command_mpc(states[-1], command, target_trajectory, t=t, dt=DT,
                                                  state_weights=state_weights, command_weights=cmd_weights)
            cmd = command[0] + np.random.randn(*command[0].shape)
            next_state = simulate(states[-1], cmd, dt=DT)
            states.append(next_state)
            list_command.append(cmd)
            clear_cmd.append(command[0])
            command = torch.cat([command[1:], command[-1:]], dim=0)

        states = torch.stack(states, dim=0).detach().numpy()
        command = torch.stack(list_command, dim=0).detach().numpy()
        clear_cmd = torch.stack(clear_cmd, dim=0).detach().numpy()

        error = np.mean((states - target_trajectory.numpy()[:states.shape[0]]) ** 2)
        print(error, loose_traj)
        if not np.isnan(error):
            data.append(({"states": states.astype(np.float32), "command": command.astype(np.float32),
                          "observation": states[:, :3].astype(np.float32)}))

        # print(len(data), "/", splits[MODE])
        # visualize(states, command, target_trajectory)
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.plot(command[:, 0])
        # plt.plot(clear_cmd[:, 0])
        # plt.subplot(2, 1, 2)
        # plt.plot(command[:, 1])
        # plt.plot(clear_cmd[:, 1])
        # plt.show()
    pickle.dump(data, open(f"../Data/Drone2D/{MODE}_noisy.pkl", 'wb'))
