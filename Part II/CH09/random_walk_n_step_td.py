"""Example 9.1 and 9.2"""

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


class RandomWalk:
    """1000-state Random Walk"""

    def __init__(self, max_episodes=10000):
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
        self.state = int
        self.reached_terminal = False
        self.action = 0
        self.small_threshold = 1e-2
        self.step_size = 0.001

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

    def rms_error(self):
        """Find the total error of mid five state values compared with true values"""
        error = 0
        for i in range(self.n_states):
            error += np.square(self.true_value[i] - self.group_values[i // self.group_size])
        error = np.sqrt(error / self.n_states)
        return error

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

    def semi_gradient_td(self, n, p=False):
        """Estimate state values under given policy using n-step TD."""
        self.reset()

        if p:
            r = trange(self.max_episodes)
        else:
            r = range(self.max_episodes)

        for _ in r:
            self.start_episode()
            terminal = float('inf')
            t, update_time = 0, 0
            states = [self.state] * (n + 1)
            rewards = [0] * (n + 1)
            while True:
                if t < terminal:
                    rewards[(t+1) % (n+1)], _ = self.step()
                    states[(t+1) % (n+1)] = self.state
                    if self.reached_terminal:
                        terminal = t + 1
                update_time = t - n + 1
                if update_time >= 0:
                    state_to_update = states[update_time % (n + 1)]
                    if state_to_update in self.states:
                        returns = 0
                        for j in range(update_time + 1, min(update_time + n, terminal) + 1):
                            returns += (self.discount ** (j - update_time - 1)) * rewards[j % (n+1)]
                        if update_time + n < terminal:
                            group_idx = int(states[(update_time + n) % (n+1)] // self.group_size)
                            returns += (self.discount ** n) * self.group_values[group_idx]
                        self.state_visit[int(state_to_update)] += 1
                        group_idx = int(state_to_update // self.group_size)
                        self.group_values[group_idx] += self.step_size * (returns - self.group_values[group_idx])
                if update_time == terminal - 1:
                    break
                t += 1
        return self.rms_error()

    def figure_9_2_left(self):
        start_time = time.time()
        self.compute_true_values()
        # Left Fig 1
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.states + 1, self.true_value, label='True Value')

        # Left Fig 2
        self.semi_gradient_td(1, p=True)
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

    def figure_9_2_right(self):
        """Plot the rms error of value prediction"""
        start_time = time.time()
        self.n_groups = 20
        self.max_episodes = 10
        repeat = 100
        steps = np.power(2, np.arange(0, 5))
        time_steps = np.logspace(-2, 0, num=20)
        errors = np.zeros((len(steps), len(time_steps)))
        plt.figure()
        for i in range(len(steps)):
            step = steps[i]
            for j in trange(len(time_steps)):
                self.step_size = time_steps[j]
                for k in range(repeat):
                    errors[i, j] += self.semi_gradient_td(step)
                errors[i, j] /= repeat
            plt.plot(time_steps, errors[i, :], label='n='+str(step))
        plt.xlabel('Time Step')
        plt.ylabel('RMS error')
        plt.title(f'run time = {int(time.time() - start_time)} s')
        plt.ylim([0.1, 0.6])
        plt.legend(loc='center right')

        plt.savefig('images/figure_9_2_right')
        plt.close()


if __name__ == '__main__':
    a = RandomWalk()
    # a.figure_9_2_left()
    a.figure_9_2_right()