"""Figure 8.7"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def sample(b):
    """Generate next 2b samples, return error"""
    sample_values = np.random.randn(b)
    true_value = np.mean(sample_values)

    samples = np.zeros(2 * b)
    errors = np.zeros(2 * b)
    for i in range(2 * b):
        samples[i] = np.random.choice(sample_values)
        # errors[i] = np.sqrt(np.sum((samples[:i + 1] - true_value) ** 2) / (i + 1))
        errors[i] = np.abs(np.average(samples[:i + 1]) - true_value)
    return errors


def plot():
    """Plot Figure 8.7"""
    repeat = 10000
    branches = [2, 10, 100, 1000, 10000]

    plt.figure()
    for branch in branches:
        error = np.zeros(2 * branch)
        runs = int(repeat/np.sqrt(branch))
        for _ in trange(runs):
            error += sample(branch)
        error /= runs
        x_axis = (np.arange(len(error)) + 1) / float(branch)
        plt.plot(x_axis, error, label=f'b = {branch}')
    plt.xlabel('number of computations')
    plt.xticks([0, 1.0, 2.0], ['0', 'b', '2b'])
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('images/figure_8_7.png')
    plt.close()


if __name__ == "__main__":
    plot()