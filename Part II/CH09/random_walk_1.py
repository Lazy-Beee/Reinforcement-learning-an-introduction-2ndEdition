"""Example 9.1 and 9.2"""

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


class RandomWalk:
    """1000-state Random Walk"""

    def __init__(self, max_episodes=100000, true_value=True):
        """Define parameters of the class"""
        self.n_states = 1000
        self.n_groups = 10
        self.max_episodes = max_episodes
        self.group_size = self.n_states // self.n_groups

        self.states = np.arange(0, self.n_states)
        self.state_values = np.zeros(self.n_states)
        self.group_values = np.zeros(self.n_groups)
        self.state_visit = np.zeros(self.n_states)

        self.true_value = np.linspace(-1, 1, num=self.n_states)
        self.policy = [0.5, 0.5]
        self.move_range = [1, 100]
        self.actions = [-1, 1]
        self.discount = 1

        self.start_state = self.n_states / 2
        self.state = self.start_state
        self.reached_terminal = False
        self.action = 0
        self.small_threshold = 1e-2
        self.step_size = 2e-5

        if true_value:
            self.compute_true_values()

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
        self.group_values = np.zeros(self.n_groups)
        self.state_visit = np.zeros(self.n_states)

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

    def gradient_mc(self):
        self.reset()
        for _ in trange(self.max_episodes):
            states, rewards, _ = self.walk()
            states.reverse()
            rewards.reverse()
            returns = 0
            for i, state in enumerate(states):
                self.state_visit[int(state)] += 1
                returns = self.discount * returns + rewards[i]
                group_idx = int(state // self.group_size)
                change = self.step_size * (returns - self.group_values[group_idx])
                self.group_values[group_idx] += change

    def semi_gradient_td(self):
        self.reset()
        for _ in trange(self.max_episodes):
            states, rewards, _ = self.walk()
            for i, state in enumerate(states):
                self.state_visit[int(state)] += 1
                group_idx_curr = int(state // self.group_size)
                if i == len(states) - 1:
                    next_state_value = 0
                else:
                    group_idx_next = int(states[i+1] // self.group_size)
                    next_state_value = self.group_values[group_idx_next]
                change = self.step_size * (rewards[i] + self.discount * next_state_value
                                           - self.group_values[group_idx_curr])
                self.group_values[group_idx_curr] += change

    def figure_9_1(self):
        start_time = time.time()

        # Left Fig 1
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.states + 1, self.true_value, label='True Value')

        # Left Fig 2
        self.gradient_mc()
        values = [self.group_values[int(i // self.group_size)] for i in self.states]
        plt.plot(self.states + 1, values, label='Approximate MC Value')
        plt.xlabel('State')
        plt.ylabel('Value')
        plt.legend(loc='upper left')

        # Right Fig
        plt.subplot(1, 2, 2)
        plt.step(self.states + 1, self.state_visit / np.sum(self.state_visit) * 100)
        plt.xlabel('State')
        plt.ylabel('Visit Distribution (%)')

        plt.suptitle(f'run time = {int(time.time() - start_time)} s')
        plt.savefig('images/figure_9_1.png')
        plt.close()

    def figure_9_2_left(self):
        start_time = time.time()

        # Left Fig 1
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.states + 1, self.true_value, label='True Value')

        # Left Fig 2
        self.semi_gradient_td()
        values = [self.group_values[int(i // self.group_size)] for i in self.states]
        plt.plot(self.states + 1, values, label='Approximate MC Value')
        plt.xlabel('State')
        plt.ylabel('Value')
        plt.legend(loc='upper left')

        # Right Fig
        plt.subplot(1, 2, 2)
        plt.step(self.states + 1, self.state_visit / np.sum(self.state_visit) * 100)
        plt.xlabel('State')
        plt.ylabel('Visit Distribution (%)')

        plt.suptitle(f'run time = {int(time.time() - start_time)} s')
        plt.savefig('images/figure_9_2_left.png')
        plt.close()


if __name__ == '__main__':
    a = RandomWalk(100000, False)
    a.figure_9_2_left()