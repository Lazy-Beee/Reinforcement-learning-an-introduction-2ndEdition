"""Figure 11.1"""

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange


class Baird:
    def __init__(self):
        # Process Parameters
        self.state = 0
        self.weight = []
        self.gradient_correction = []

        # Settings
        self.n_states = 7
        self.n_weights = 8
        self.states = np.arange(self.n_states)
        self.max_steps = 1000
        self.eps = 0.1
        self.step_size_a = 0.01
        self.step_size_b = 0.01
        self.discount = 0.99
        self.feature = np.zeros((self.n_states, self.n_weights))

        # Records
        self.weight_record = []
        self.ve_record = []
        self.pbe_record = []

    def reset(self):
        self.state = 0
        self.weight = np.asarray([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)
        self.weight_record = []
        self.ve_record = []
        self.pbe_record = []
        self.gradient_correction = np.zeros(self.n_weights)
        for i, state in enumerate(self.states):
            if state in self.states[:-1]:
                self.feature[i, i] = 2
                self.feature[i, -1] = 1
            elif state == self.states[-1]:
                self.feature[i, i] = 1
                self.feature[i, -1] = 2

    def get_state_values(self, state):
        return np.dot(self.feature[state, :], self.weight)

    def update_weight(self, state, change):
        self.weight += np.dot(self.feature[state, :], change)

    def behavior_policy(self):
        if np.random.binomial(1, 1/self.n_states):
            self.state = self.states[-1]
        else:
            self.state = np.random.choice(self.states[:-1])
        return self.state

    def mean_square_ve(self):
        ve = np.sqrt(np.mean(np.power(np.dot(self.feature, self.weight), 2)))
        self.ve_record.append(ve)

    def mean_square_pbe(self):
        be = np.zeros(self.n_states)
        for state in self.states:
            value, next_value = self.get_state_values(state), self.get_state_values(self.states[-1])
            be[state] += 0 + self.discount * next_value - value
        state_distribution = np.asarray([1 / self.n_states] * self.n_states)
        state_distribution_mat = np.matrix(np.diag(state_distribution))
        projection = np.matrix(self.feature) * \
            np.linalg.pinv((self.feature.T * state_distribution_mat) * self.feature) * \
            np.matrix(self.feature.T) * state_distribution_mat
        be = np.dot(np.asarray(projection), be)
        self.pbe_record.append(np.sqrt(np.dot(np.power(be, 2), state_distribution)))

    def semi_gradient_off_policy_td(self):
        self.reset()
        for _ in trange(self.max_steps):
            state = self.state
            value = self.get_state_values(state)
            next_state = self.behavior_policy()
            next_value = self.get_state_values(next_state)
            if next_state == self.states[-1]:
                change = 1 / (1/self.n_states) * self.step_size_a * (0 + self.discount * next_value - value)
            else:
                change = 0
            self.update_weight(state, change)
            self.weight_record.append(self.weight.copy())

    def semi_gradient_dp(self):
        self.reset()
        for _ in trange(self.max_steps):
            change = []
            for state in self.states:
                expected_return = 0 + self.discount * self.get_state_values(self.states[-1])
                change.append(self.step_size_a * (expected_return - self.get_state_values(state)) / self.n_states)
            for i, state in enumerate(self.states):
                self.update_weight(state, change[i])
            self.weight_record.append(self.weight.copy())

    def gradient_correction_td(self):
        self.reset()
        for _ in trange(self.max_steps):
            state = self.state
            value = self.get_state_values(state)
            next_state = self.behavior_policy()
            next_value = self.get_state_values(next_state)
            if next_state == self.states[-1]:
                x = self.feature[state, :]
                next_x = self.feature[next_state, :]
                delta = 0 + self.discount * next_value - value
                rho = 1 / (1 / self.n_states)
                change = rho * self.step_size_a
                change *= delta * x - self.discount * next_x * np.dot(x, self.gradient_correction)
                self.gradient_correction += self.step_size_b * rho * (delta - np.dot(self.gradient_correction, x) * x)
            else:
                change = 0
            self.weight += np.asarray(change)
            self.weight_record.append(self.weight.copy())
            self.mean_square_ve()
            self.mean_square_pbe()

    def gradient_correction_td_expected(self):
        self.reset()
        update_weight = 1 / self.n_states
        for _ in trange(self.max_steps):
            changes = np.zeros(self.n_weights)
            changes_gc = np.zeros(self.n_weights)
            for state in self.states:
                x, next_x = self.feature[state, :], self.feature[-1, :]
                value, next_value = self.get_state_values(state), self.get_state_values(self.states[-1])
                delta = 0 + self.discount * next_value - value
                rho = 1 / (1 / self.n_states)
                change = rho * self.step_size_a
                change *= delta * x - self.discount * next_x * np.dot(x, self.gradient_correction)
                change_gc = self.step_size_b * rho * (delta - np.dot(self.gradient_correction, x) * x)
                changes += change
                changes_gc += change_gc
            self.weight += changes * update_weight / self.n_states
            self.gradient_correction += changes_gc * update_weight / self.n_states
            self.weight_record.append(self.weight.copy())
            self.mean_square_ve()
            self.mean_square_pbe()

    def one_step_emphatic_td_expected(self):
        self.reset()
        prev_m = 0
        update_weight = 1 / self.n_states
        for _ in trange(self.max_steps):
            expected_m = 0
            change = np.zeros(self.n_weights)
            for state in self.states:
                value, next_value = self.get_state_values(state), self.get_state_values(self.states[-1])
                delta = 0 + self.discount * next_value - value
                rho = 1 / (1 / self.n_states)
                if state == self.states[-1]:
                    prev_rho = 1 / (1 / self.n_states)
                else:
                    prev_rho = 0
                m = self.discount * prev_rho * prev_m + 1
                expected_m += m
                change += self.step_size_a * m * rho * delta * self.feature[state, :]
            prev_m = expected_m / self.n_states
            self.weight += change * update_weight / self.n_states
            self.weight_record.append(self.weight.copy())
            self.mean_square_ve()
            self.mean_square_pbe()

    def figure_11_2(self):
        start_time = time.time()
        self.max_steps = 1000
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        self.semi_gradient_off_policy_td()
        record = np.asarray(self.weight_record)
        for i in range(8):
            plt.plot(record[:, i], label=f'w{i+1}')
        plt.xlabel('Steps')
        plt.ylabel('Weight')
        plt.title('Semi-gradient Off-policy TD')
        plt.legend()

        plt.subplot(1, 2, 2)
        self.semi_gradient_dp()
        record = np.asarray(self.weight_record)
        for i in range(8):
            plt.plot(record[:, i], label=f'w{i + 1}')
        plt.xlabel('Steps')
        plt.ylabel('Weight')
        plt.title('Semi-gradient DP')
        plt.legend()

        plt.suptitle(f'run time = {int(time.time() - start_time)} s')
        plt.savefig(f'images/figure 11.2.2.png')
        plt.close()

    def figure_11_5(self):
        start_time = time.time()
        self.max_steps = 1000
        self.step_size_a = 0.005
        self.step_size_b = 0.05
        plt.figure(figsize=(20, 8))

        for i in range(2):
            ax = plt.subplot(1, 2, i+1)
            if i == 0:
                self.gradient_correction_td()
                plt.title('Gradient correction TD (TDC)')
            else:
                self.gradient_correction_td_expected()
                plt.title('Expected TDC')
            record = np.asarray(self.weight_record)
            for j in range(6):
                plt.plot(record[:, j], '--', label=f'w{j + 1}')
            for j in range(6, 8):
                plt.plot(record[:, j], label=f'w{j + 1}')
            plt.plot(self.ve_record, label='msVE')
            plt.plot(self.pbe_record, label='msPBE')
            plt.xlabel('Steps')
            plt.ylabel('Weight')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle(f'run time = {int(time.time() - start_time)} s')
        plt.savefig(f'images/figure 11.5.png')
        plt.close()

    def figure_11_6(self):
        start_time = time.time()
        self.max_steps = 1000
        self.step_size_a = 0.01

        plt.figure(figsize=(10, 8))
        self.one_step_emphatic_td_expected()
        record = np.asarray(self.weight_record)
        for j in range(6):
            plt.plot(record[:, j], '--', label=f'w{j + 1}')
        for j in range(6, 8):
            plt.plot(record[:, j], label=f'w{j + 1}')
        plt.plot(self.ve_record, label='msVE')
        plt.plot(self.pbe_record, label='msPBE')
        plt.xlabel('Steps')
        plt.ylabel('Weight')
        plt.title('Gradient correction TD (TDC)')
        plt.legend(loc='upper right')

        plt.suptitle(f'run time = {int(time.time() - start_time)} s')
        plt.savefig(f'images/figure 11.6.png')
        plt.close()


if __name__ == "__main__":
    a = Baird()
    a.figure_11_6()
