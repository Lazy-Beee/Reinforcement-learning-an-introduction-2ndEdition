"""Example 9.1 and 9.2"""

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


class RandomWalk:
    """1000-state Random Walk"""

    def __init__(self, max_episodes):
        """Define parameters of the class"""
        self.n_states = 1000
        self.n_groups = 10
        self.max_episodes = max_episodes
        self.sampling_rate = self.max_episodes / 100
        self.group_size = self.n_states // self.n_groups

        self.states = np.arange(0, self.n_states)
        self.state_values = np.zeros(self.n_states)
        self.group_values = np.zeros(self.n_groups)

        self.true_value = np.linspace(-1, 1, num=self.n_states)
        self.policy = [0.5, 0.5]
        self.move_range = [1, 100]
        self.actions = [-1, 1]
        self.discount = 1

        self.start_state = self.n_states / 2
        self.state = self.start_state
        self.reached_terminal = False
        self.action = 0
        self.small_threshold = 1e-3
        self.step_size = 2e-4

    def compute_true_values(self):
        while True:
            old_values = self.true_value.copy()
            for state in self.states:
                self.true_value[state] = 0
                for action in self.actions:
                    for step in range(self.move_range[0], self.move_range[1] + 1):
                        move = action * step
                        next_state = state + move
                        d = len(self.actions) * (self.move_range[1] - self.move_range[0] + 1)
                        if next_state < self.states[0]:
                            self.true_value[state] += - 1 / d
                        elif next_state >= self.states[-1]:
                            self.true_value[state] += 1 / d
                        else:
                            self.true_value[state] += self.true_value[next_state] / d
            error = np.sum(np.abs(old_values - self.true_value))
            if error < self.small_threshold:
                print('True values estimate complete.')
                break

    def rms_error(self):
        """Find the total error of mid five state values compared with true values"""
        error = 0
        for i in range(self.n_states):
            error += np.square(self.true_value[i] - self.group_values[i // self.group_size])
        error = np.sqrt(error / self.n_states)
        return error

    def reset(self):
        """Reset class parameters"""
        self.group_values = np.zeros(self.n_groups)

    def start_episode(self):
        """Get ready for a new episode"""
        self.state = self.start_state
        self.action = 0
        self.reached_terminal = False

    def step(self):
        """Take a random move and end the episode if reached terminal. Return reward."""
        self.action = np.random.choice(self.actions, p=self.policy)
        self.state += self.action * np.random.randint(self.move_range[0], self.move_range[1] + 1)
        if self.state > self.states[-1]:
            self.reached_terminal = True
            return 1, self.action
        elif self.state < self.states[0]:
            self.reached_terminal = True
            return -1, self.action
        else:
            return 0, self.action

    def walk(self):
        self.start_episode()
        states, actions, rewards = [], [], []
        while not self.reached_terminal:
            state = self.state
            reward, action = self.step()
            states.append(state)
            rewards.append(reward)
            actions.append(action)
        return states, rewards, actions

    def one_hot(self, s):
        one_hot = np.zeros((self.n_groups, 1))
        one_hot[int(s // self.group_size), 0] = 1
        return one_hot

    def gradient_mc(self):
        self.reset()
        errors = []
        times = []
        time_1 = time.time()
        for i in trange(self.max_episodes):
            states, rewards, _ = self.walk()
            states.reverse()
            rewards.reverse()
            returns = 0
            for j, state in enumerate(states):
                returns = self.discount * returns + rewards[j]
                group_idx = int(state // self.group_size)
                change = self.step_size * (returns - self.group_values[group_idx])
                self.group_values[group_idx] += change
            if (i + 1) % self.sampling_rate == 0:
                errors.append(self.rms_error())
                times.append(time.time() - time_1)
        return errors, times

    def semi_gradient_td(self):
        self.reset()
        errors = []
        times = []
        time_1 = time.time()
        for i in trange(self.max_episodes):
            states, rewards, _ = self.walk()
            for j, state in enumerate(states):
                group_idx_curr = int(state // self.group_size)
                if j == len(states) - 1:
                    next_state_value = 0
                else:
                    group_idx_next = int(states[j+1] // self.group_size)
                    next_state_value = self.group_values[group_idx_next]
                change = self.step_size * (rewards[j] + self.discount * next_state_value
                                           - self.group_values[group_idx_curr])
                self.group_values[group_idx_curr] += change

            if (i + 1) % self.sampling_rate == 0:
                errors.append(self.rms_error())
                times.append(time.time() - time_1)
        return errors, times

    def least_square_td(self):
        self.reset()
        errors = []
        times = []
        a_mat_inv = np.dot(np.power(self.small_threshold, -1), np.identity(self.n_groups))
        b_vec = np.zeros((self.n_groups, 1))
        time_1 = time.time()
        for i in trange(self.max_episodes):
            self.start_episode()
            while not self.reached_terminal:
                x = self.one_hot(self.state)
                reward, _ = self.step()
                if not self.reached_terminal:
                    x_next = self.one_hot(self.state)
                else:
                    x_next = 0

                v = np.matmul(np.transpose(a_mat_inv), (x - self.discount * x_next))
                a_mat_inv -= np.matmul(np.matmul(a_mat_inv, x), np.transpose(v)) / (1 + np.matmul(np.transpose(v), x))
                b_vec += reward * x
                self.group_values = np.matmul(a_mat_inv, b_vec)[:, 0]

            if (i + 1) % self.sampling_rate == 0:
                errors.append(self.rms_error())
                times.append(time.time() - time_1)
        return errors, times

    def figure(self):
        start_time = time.time()
        # self.compute_true_values()

        n_data_point = int(self.max_episodes // self.sampling_rate)
        x_val = np.arange(1, n_data_point + 1) * self.sampling_rate
        errors_mc, errors_sg, errors_ls = np.zeros(n_data_point), np.zeros(n_data_point), np.zeros(n_data_point)
        times_mc, times_sg, times_ls = np.zeros(n_data_point), np.zeros(n_data_point), np.zeros(n_data_point)

        repeat = 1
        for _ in range(repeat):
            a, b = self.gradient_mc()
            errors_mc += a
            times_mc += b
            a, b = self.semi_gradient_td()
            errors_sg += a
            times_sg += b
            a, b = self.least_square_td()
            errors_ls += a
            times_ls += b

        errors_mc /= repeat
        errors_sg /= repeat
        errors_ls /= repeat
        times_mc /= repeat
        times_sg /= repeat
        times_ls /= repeat

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x_val, errors_mc, label='Gradient MC')
        plt.plot(x_val, errors_sg, label='Semi-Gradient TD')
        plt.plot(x_val, errors_ls, label='Least Squares TD')
        plt.xlabel('Episodes')
        plt.ylabel('RMS Error')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x_val, times_mc, label='Gradient MC')
        plt.plot(x_val, times_sg, label='Semi-Gradient TD')
        plt.plot(x_val, times_ls, label='Least Squares TD')
        plt.xlabel('Episodes')
        plt.ylabel('Run Time (s)')
        plt.legend()

        plt.suptitle(f'run time = {int(time.time() - start_time)} s')
        plt.savefig(f'images/figure_LSTD_{self.max_episodes}.png')
        plt.close()


if __name__ == '__main__':
    k = RandomWalk(1000000)
    k.figure()

