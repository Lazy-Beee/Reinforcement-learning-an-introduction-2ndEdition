"""Figure 9.8, RBF not in function"""
import bisect

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


class CoarseCoding:
    def __init__(self, rbf=False):
        self.rbf = rbf
        self.interval = 2
        self.n_features = 50
        self.high = 1
        self.low = 0
        self.step_size = 0.2
        self.std_dev = 0.1
        self.weights = np.zeros(self.n_features)

        self.left_bound = 0
        self.right_bound = 0
        self.feature_width = 0
        self.features = []
        self.pos = 0
        self.active_features = []
        self.feature_step = 0

    def reset(self):
        self.weights = np.zeros(self.n_features)
        self.left_bound = 0
        self.right_bound = 0
        self.feature_width = 0
        self.features = []
        self.pos = 0
        self.active_features = []
        self.feature_step = 0

    def init_coarse(self, feature_width):
        self.feature_width = feature_width
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

    def init_value_func(self, feature_width):
        self.init_coarse(feature_width)

        self.feature_step = (self.interval - self.feature_width) / self.n_features
        for i in range(self.n_features - 1):
            left = i * self.feature_step + self.left_bound
            self.features.append([left, left + self.feature_width])
        self.features.append([(self.n_features - 1) * self.feature_step, self.right_bound])

        if self.rbf:
            for i in range(self.n_features):
                self.features[i].append((self.features[i][0] + self.features[i][1]) / 2)

    def get_active_features(self):
        left_bounds = [self.features[i][0] for i in range(self.n_features)]
        right_bounds = [self.features[i][1] for i in range(self.n_features)]
        start = bisect.bisect_left(right_bounds, self.pos)
        if self.pos < self.features[-1][0]:
            end = bisect.bisect_left(left_bounds, self.pos)
        else:
            end = self.n_features - 1
        self.active_features = [i for i in range(start, end + 1)]

    def get_func_value(self, x=None):
        if x is not None:
            self.pos = x
            self.get_active_features()

        if self.rbf:
            value = 0
            for i in self.active_features:
                value += np.exp(- np.power(self.pos - self.features[i][2], 2) / (2 * np.power(self.std_dev, 2)))
            return value
        else:
            return np.sum(self.weights[self.active_features])

    def update_func_values(self, change):
        change /= len(self.active_features)

        if self.rbf:
            for i in self.active_features:
                change *= np.exp(- np.power(self.pos - self.features[i][2], 2) / (2 * np.power(self.std_dev, 2)))
                # change *= - (self.pos - self.features[i][2]) / np.power(self.std_dev, 2)
                self.weights[i] += change
        else:
            for i in self.active_features:
                self.weights[i] += change

    def approximate(self, feature_width, n):
        self.reset()
        self.init_value_func(feature_width)
        samples = self.get_sample(n)
        for x, y in samples:
            self.pos = x
            self.get_active_features()
            change = self.step_size * (y - self.get_func_value())
            self.update_func_values(change)

    def figure_9_8(self):
        start_time = time.time()

        repeat = 10
        num_of_samples = [10, 40, 160, 640, 2560, 10240]
        feature_widths = [0.2, 0.4, 1.0]
        plt.figure(figsize=(20, 10))
        for index, num_of_sample in enumerate(num_of_samples):
            plt.subplot(2, 3, index + 1)
            plt.title('%d samples' % num_of_sample)
            for feature_width in tqdm(feature_widths):
                self.init_coarse(feature_width)
                self.std_dev = feature_width / 5
                axis_x = np.arange(self.left_bound, self.right_bound, 0.01)
                values = np.zeros(axis_x.shape)
                for _ in range(repeat):
                    self.approximate(feature_width, num_of_sample)
                    values += [self.get_func_value(x) for x in axis_x]
                values /= repeat
                plt.plot(axis_x, values, label='feature width %.01f' % feature_width)
            axis_x = np.arange(self.left_bound, self.right_bound, 0.01)
            true_y = []
            for x in axis_x:
                if -0.5 <= x < 0.5:
                    true_y.append(1)
                else:
                    true_y.append(0)
            plt.plot(axis_x, true_y, '--', label='true value')
            # plt.ylim(-0.4, 1.4)
            plt.legend(loc='lower center')

        plt.suptitle(f'run time = {int(time.time() - start_time)} s')
        if self.rbf:
            plt.savefig('images/figure_9_8_RBF.png')
        else:
            plt.savefig('images/figure_9_8.png')
        plt.close()


if __name__ == '__main__':
    a = CoarseCoding()
    a.figure_9_8()
