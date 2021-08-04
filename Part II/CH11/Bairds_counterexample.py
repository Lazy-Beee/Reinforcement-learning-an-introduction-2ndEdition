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

        # Settings
        self.n_states = 7
        self.states = np.arange(self.n_states)
        self.max_steps = 1000
        self.eps = 0.1
        self.step_size_alpha = 0.01
        self.step_size_beta = 0.01
        self.discount = 0.99

        # Records
        self.weight_record = []

    def reset(self):
        self.state = 0
        self.weight = [1, 1, 1, 1, 1, 1, 10, 1]
        self.weight_record = []

    def get_state_values(self, state):
        if state in self.states[:-1]:
            return 2 * self.weight[state] + self.weight[-1]
        elif state == self.states[-1]:
            return self.weight[state] + self.weight[-1] * 2
        else:
            print(f"Invalid state index in get_state_values({state}).")
            quit()

    def update_weight(self, state, change):
        if state in self.states[:-1]:
            self.weight[state] += change * 2
            self.weight[-1] += change
        elif state == self.states[-1]:
            self.weight[state] += change
            self.weight[-1] += change * 2
        else:
            print(f"Invalid state index in update_weight({state, change}).")
            quit()

    def behavior_policy(self):
        if np.random.binomial(1, 1/self.n_states):
            self.state = self.states[-1]
        else:
            self.state = np.random.choice(self.states[:-1])
        return self.state

    def semi_gradient_off_policy_td(self):
        self.reset()
        for _ in trange(self.max_steps):
            state = self.state
            value = self.get_state_values(state)
            next_state = self.behavior_policy()
            next_value = self.get_state_values(next_state)
            if next_state == self.states[-1]:
                change = 1/(1/self.n_states) * self.step_size_alpha * (0 + self.discount * next_value - value)
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
                change.append(self.step_size_alpha * (expected_return - self.get_state_values(state)) / self.n_states)
            for i, state in enumerate(self.states):
                self.update_weight(state, change[i])
            self.weight_record.append(self.weight.copy())

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
        plt.savefig(f'images/figure 11.2.png')
        plt.close()


if __name__ == "__main__":
    a = Baird()
    a.figure_11_2()




