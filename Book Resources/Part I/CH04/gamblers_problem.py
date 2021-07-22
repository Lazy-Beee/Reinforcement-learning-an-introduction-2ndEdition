"""Example 4.3"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')


class Gambler:
    """The Gambler class"""

    def __init__(self, goal=100, head_percentage=50):
        self.goal = goal
        self.states = np.arange(self.goal + 1)
        self.head_prob = head_percentage / 100
        self.tail_prob = 1 - self.head_prob
        self.sweep_history = []
        self.value = np.zeros(self.states.shape)
        self.value[-1] = 1
        self.policy = np.zeros(self.states.shape, dtype=int)
        self.small_threshold = 1e-9
        self.discount = 1

    def find_best_option(self, state):
        actions = np.arange(1, min(state, self.goal - state) + 1)
        returns = np.zeros(actions.shape)
        for i in range(len(actions)):
            head_value = self.head_prob * self.discount * self.value[state + actions[i]]
            tail_value = self.tail_prob * self.discount * self.value[state - actions[i]]
            returns[i] = head_value + tail_value
        opt = np.argmax(returns)
        return returns[opt], actions[opt]

    def value_iteration(self):
        while True:
            old_value = self.value.copy()
            self.sweep_history.append(old_value)
            for i in range(1, len(self.states) - 1):
                self.value[i], self.policy[i] = self.find_best_option(self.states[i])
            if abs(old_value - self.value).max() < self.small_threshold:
                break

    def find_opt_policy(self):
        for i in range(1, len(self.states) - 1):
            _, self.policy[i] = self.find_best_option(self.states[i])

    def plot(self):
        plt.figure(figsize=(10, 20))

        plt.subplot(2, 1, 1)
        for sweep, state_value in enumerate(self.sweep_history):
            plt.plot(state_value, label='sweep {}'.format(sweep))
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
        # plt.legend(loc='best')

        plt.subplot(2, 1, 2)
        plt.scatter(self.states[1:-1], self.policy[1:-1])
        plt.xlabel('Capital')
        plt.ylabel('Final policy (stake)')

        plt.savefig(f'images/gambler_heads_{self.head_prob}.png')
        plt.close()


if __name__ == "__main__":
    gambler = Gambler(head_percentage=40)
    gambler.value_iteration()
    gambler.find_opt_policy()
    gambler.plot()
