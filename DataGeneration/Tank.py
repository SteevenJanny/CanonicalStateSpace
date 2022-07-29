import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d

plt.style.use("seaborn")
K1, K2, K3, K4 = 0.5, 0.4, 0.2, 0.3

MODE = "test"


def f(X, u):
    x1 = X[0] - K1 * np.sqrt(X[0]) + K2 * u
    x2 = X[1] + K3 * np.sqrt(X[0]) - K4 * np.sqrt(X[1])
    x1 = np.clip(x1, 0, np.inf)
    x2 = np.clip(x2, 0, np.inf)
    return np.stack([x1, x2])


def simulate(target):
    states = [np.zeros((2))]
    error = [0]
    command = [0]
    for t in range(target.shape[0]):
        error.append(target[t] - states[-1][-1])
        integ = sum(error)
        deriv = error[-1] - error[-2]

        u = 4 * error[-1] + 0.6 * integ + 4 * deriv
        command.append(u)
        x = f(states[-1], u)
        states.append(x)
    states = np.stack(states, axis=0)
    command = np.stack(command, axis=0)
    return states, command


if __name__ == '__main__':
    splits = {'train': 60, 'test': 20, 'valid': 20}

    data = []
    for _ in range(splits[MODE]):
        keypoints = np.concatenate([np.zeros((1)), np.random.rand(5) * 5])
        positions = np.array([40 * i for i in range(0, 6)])
        target = interp1d(positions, keypoints, kind='cubic')(np.arange(200))

        while np.sum(target < 0) != 0:
            keypoints = np.concatenate([np.zeros((1)), np.random.rand(5) * 4])
            positions = np.array([40 * i for i in range(0, 6)])
            target = interp1d(positions, keypoints, kind='cubic')(np.arange(200))

        states, command = simulate(target)
        data.append({"states": states, "command": command, "observation": states[:, -1:]})
        plt.scatter(positions, keypoints, label="Waypoints")
        plt.plot(target, label="Target")
        plt.plot(states[:, -1], label="System")
        plt.legend()
        plt.show()
    pickle.dump(data, open(f"../Data/Tank_V2/{MODE}.pkl", 'wb'))
