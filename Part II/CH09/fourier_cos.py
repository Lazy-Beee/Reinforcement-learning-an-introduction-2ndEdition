"""Figure 9.4"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import time


def fourier_cos(c=(0, 0)):
    resolution = 100
    c = np.array(c)
    x = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            s = np.array((j / resolution, i / resolution))
            s.transpose()
            x[i, j] = np.cos(np.pi * np.dot(s, c))
    return x


def plot():
    start_time = time.time()

    _, axes = plt.subplots(2, 3, figsize=(30, 20))
    axes = axes.flatten()

    cs = [(0, 1), (1, 0), (1, 1), (0, 5), (2, 5), (5, 2)]

    for i, c in enumerate(cs):
        fig = sns.heatmap(np.flipud(fourier_cos(c)), cmap="YlGnBu", ax=axes[i], xticklabels=False, yticklabels=False,
                          cbar=False)
        fig.set_title(f'c = {c}', fontsize=30)

    plt.suptitle(f'run time = {int(time.time() - start_time)} s', fontsize=30)
    plt.savefig('images/fourier cosine features')
    plt.close()


if __name__ == '__main__':
    plot()