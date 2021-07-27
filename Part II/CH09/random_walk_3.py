"""Figure 9.5"""

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


class RandomWalk:
    """1000-state Random Walk"""

    def __init__(self, max_episodes=10000):
        """Define parameters of the class"""
        self.n_states = 1000
        self.max_episodes = max_episodes
        self.sampling_rate = 100

        self.states = np.arange(0, self.n_states)
        self.true_value = np.linspace(-1, 1, num=self.n_states)
        self.policy = [0.5, 0.5]
        self.move_range = [1, 100]
        self.actions = [-1, 1]
        self.discount = 1

        self.start_state = self.n_states / 2
        self.state = 0
        self.reached_terminal = False
        self.action = 0
        self.small_threshold = 1e-2
        self.step_size = 0.001

        self.weights = []
        self.bases = []

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

    def reset(self):
        """Reset class parameters"""

    def start_episode(self):
        """Get ready for a new episode"""
        self.state = self.start_state
        self.action = 0
        self.reached_terminal = False

    def rms_error(self):
        """Find the total error of mid five state values compared with true values"""
        error = 0
        for i in range(self.n_states):
            error += np.square(self.true_value[i] - self.get_func_value(i))
        error = np.sqrt(error / self.n_states)
        return error

    def init_value_func(self, mode, order):
        """Initiate bases value function. polynomial-0, fourier-1."""
        self.weights = np.zeros(order + 1)
        if mode == 0:
            self.bases = [lambda s, i=i: np.power(s, i) for i in range(order + 1)]
        else:
            self.bases = [lambda s, i=i: np.cos(s * i * np.pi) for i in range(order + 1)]

    def get_func_value(self, state):
        """Get value of state in value function"""
        state /= self.n_states
        return np.dot(self.weights, np.array([func(state) for func in self.bases]))

    def update_value_func(self, state, delta):
        """Update value of state in value function with change delta"""
        state /= self.n_states
        self.weights += delta * np.array([func(state) for func in self.bases])

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
        """Run full episode"""
        self.start_episode()
        states, actions, rewards = [], [], []
        while not self.reached_terminal:
            state = self.state
            reward, action = self.step()
            states.append(state)
            rewards.append(reward)
            actions.append(action)
        return states, rewards, actions

    def gradient_mc(self, mode, order):
        """Gradient Monte Carlo method"""
        self.reset()
        self.init_value_func(mode, order)
        error = []

        for episode in range(self.max_episodes):
            states, rewards, _ = self.walk()
            states.reverse()
            rewards.reverse()
            returns = 0
            for i, state in enumerate(states):
                returns = self.discount * returns + rewards[i]
                change = self.step_size * (returns - self.get_func_value(state))
                self.update_value_func(state, change)

            if (episode + 1) % self.sampling_rate == 0:
                error.append(self.rms_error())

        return error

    def figure_9_5(self):
        """Plot the rms error of value prediction"""
        start_time = time.time()

        self.max_episodes = 5000
        repeat = 10
        modes = [0, 1]
        step_sizes = [1e-4, 5e-5]
        orders = [5, 10, 20]
        errors = np.zeros((len(modes), len(orders), self.max_episodes // self.sampling_rate))

        for _ in trange(repeat):
            for i in range(len(modes)):
                mode = modes[i]
                self.step_size = step_sizes[i]
                for j in range(len(orders)):
                    order = orders[j]
                    errors[i, j] += self.gradient_mc(mode, order)
        errors /= repeat

        plt.figure()
        x_axes = np.arange(self.max_episodes // self.sampling_rate) * self.sampling_rate
        for i in range(len(modes)):
            mode = modes[i]
            if mode == 0:
                func_name = 'POLY'
            else:
                func_name = 'FOURIER'
            self.step_size = step_sizes[i]
            for j in range(len(orders)):
                order = orders[j]
                plt.plot(x_axes, errors[i, j], label=func_name + ' ' + str(order))
        plt.xlabel('Time Step')
        plt.ylabel('RMS error')
        plt.title(f'run time = {int(time.time() - start_time)} s')
        plt.legend(loc='upper right')

        plt.savefig('images/figure_9_5')
        plt.close()


if __name__ == '__main__':
    a = RandomWalk()
    a.figure_9_5()