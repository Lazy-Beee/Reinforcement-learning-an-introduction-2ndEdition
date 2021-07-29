"""Figure 9.8, RBF"""
import bisect

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


class CoarseCoding:
    def __init__(self):
        self.interval = 2
        self.n_features = 50
        self.high = 1
        self.low = 0
        self.step_size = 0.2 / self.n_features
        self.weights = np.zeros(self.n_features)

        self.left_bound = 0
        self.right_bound = 0
        self.feature_width = 0
        self.features = []
        self.pos = 0
        self.std_dev = 1

    def reset(self):
        self.weights = np.zeros(self.n_features)
        self.left_bound = 0
        self.right_bound = 0
        self.feature_width = 0
        self.features = []
        self.pos = 0

    def init_coarse(self):
        self.left_bound = - self.interval / 2
        self.right_bound = self.interval / 2

    def get_sample(self, n):
        samples = []
        for _ in range(n):
            pos = np.random.random() * self.interval + self.left_bound
            if self.left_bound / 2 <= pos < self.right_bound / 2:
                samples.append([pos, self.high])
            else:
                samples.append([pos, self.low])
        return samples

    def init_value_func(self):
        self.init_coarse()
        feature_step = self.interval / (self.n_features - 1)
        for i in range(self.n_features):
            self.features.append(self.left_bound + i * feature_step)

    def get_func_value(self, x=None):
        if x is not None:
            self.pos = x
        value = 0
        for i in range(self.n_features):
            value += np.exp(- np.power(self.pos - self.features[i], 2) / (2 * np.power(self.std_dev, 2))) \
                     * self.weights[i]
        return value

    def update_func_values(self, change):
        for i in range(self.n_features):
            change_ = change * np.exp(- np.power(self.pos - self.features[i], 2) / (2 * np.power(self.std_dev, 2)))
            self.weights[i] += change_

    def approximate(self, n):
        self.reset()
        self.init_value_func()
        samples = self.get_sample(n)
        for x, y in samples:
            self.pos = x
            change = self.step_size * (y - self.get_func_value())
            self.update_func_values(change)

    def figure_9_8(self):
        start_time = time.time()

        repeat = 10
        num_of_samples = [40, 160, 640, 2560, 10240, 40960]
        feature_widths = [0.05, 0.2, 0.4]
        plt.figure(figsize=(20, 10))
        for index, num_of_sample in enumerate(num_of_samples):
            plt.subplot(2, 3, index + 1)
            plt.title('%d samples' % num_of_sample)
            for feature_width in tqdm(feature_widths):
                self.init_coarse()
                self.std_dev = feature_width / 2
                axis_x = np.arange(self.left_bound, self.right_bound, 0.01)
                values = np.zeros(axis_x.shape)
                for _ in range(repeat):
                    self.approximate(num_of_sample)
                    values += [self.get_func_value(x) for x in axis_x]
                values /= repeat
                plt.plot(axis_x, values, label=str(feature_width))
            axis_x = np.arange(self.left_bound, self.right_bound, 0.01)
            true_y = []
            for x in axis_x:
                if -0.5 <= x < 0.5:
                    true_y.append(1)
                else:
                    true_y.append(0)
            plt.plot(axis_x, true_y, '--', label='true value')
            plt.ylim(-0.4, 1.4)
            plt.legend(loc='lower center')

        plt.suptitle(f'run time = {int(time.time() - start_time)} s')
        plt.savefig('images/figure_9_8_RBF.png')
        plt.close()


if __name__ == '__main__':
    a = CoarseCoding()
    a.figure_9_8()
