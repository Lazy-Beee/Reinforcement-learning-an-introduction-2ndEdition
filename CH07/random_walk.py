"""Example 7.1"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


class RandomWalk:
    """
    a larger random walk process, with 19 states instead of 5 (and with a -1 outcome on the
    left, all values initialized to 0)
    """

    def __init__(self):
        """Define parameters of the class"""
        self.number_of_states = 19
        self.states = np.arange(0, self.number_of_states + 2)
        self.true_values = np.arange(-9, 10) / 10
        self.state_values = np.zeros(self.number_of_states + 2)
        self.policy = [0.5, 0.5]
        self.actions = [-1, 1]
        self.discount = 1

        self.state = 10
        self.reached_terminal = False
        self.action = 0
        self.small_threshold = 1e-3

    def reset(self):
        """Reset class parameters"""
        self.state_values = np.zeros(self.number_of_states + 2)

    def start_episode(self):
        """Get ready for a new episode"""
        self.state = 10
        self.action = 0
        self.reached_terminal = False

    def rms_error(self):
        """Find the total error of mid five state values compared with true values"""
        error = 0
        for i in range(1, self.number_of_states + 1):
            error += np.square(self.state_values[i] - self.true_values[i - 1])
        error = np.sqrt(error / self.number_of_states)
        return error

    def step(self):
        """Take a random move and end the episode if reached terminal. Return reward."""
        self.action = np.random.choice(self.actions, p=self.policy)
        self.state += self.action
        if self.state == self.states[-1]:
            self.reached_terminal = True
            return 1
        elif self.state == self.states[0]:
            self.reached_terminal = True
            return -1
        else:
            return 0

    def n_step_td(self, episode, step, time_step):
        """Estimate state values under given policy using n-step TD."""
        self.reset()
        for _ in range(episode):
            self.start_episode()
            terminal = float('inf')
            time, update_time = 0, 0
            states = [self.state]
            rewards = [0]
            while True:
                if time < terminal:
                    rewards.append(self.step())
                    states.append(self.state)
                    if self.reached_terminal:
                        terminal = time + 1
                update_time = time - step + 1
                if update_time >= 0:
                    returns = 0
                    for j in range(update_time + 1, min(update_time + step, terminal) + 1):
                        returns += (self.discount ** (j - update_time - 1)) * rewards[j]
                    if update_time + step < terminal:
                        returns += (self.discount ** step) * self.state_values[states[update_time + step]]
                    state = states[update_time]
                    if state != self.states[0] and state != self.states[-1]:
                        self.state_values[state] += time_step * (returns - self.state_values[state])
                if update_time == terminal - 1:
                    break
                time += 1
        return self.rms_error()

    def figure_7_2(self):
        """Plot the rms error of value prediction"""
        episode = 10
        repeat = 100
        steps = np.power(2, np.arange(0, 6))
        time_steps = np.logspace(-2, 0, num=30)
        errors = np.zeros((len(steps), len(time_steps)))
        plt.figure()
        for i in range(len(steps)):
            step = steps[i]
            for j in trange(len(time_steps)):
                time_step = time_steps[j]
                # print(f'step:{step} time_step:{time_step}')
                for k in range(repeat):
                    errors[i, j] += self.n_step_td(episode, step, time_step)
                errors[i, j] /= repeat
            plt.plot(time_steps, errors[i, :], label='n='+str(step))
        plt.xlabel('Time Step')
        plt.ylabel('RMS error')
        plt.ylim([0.1, 0.6])
        plt.legend(loc='center right')

        plt.savefig('images/figure_7_2')
        plt.close()


if __name__ == '__main__':
    system = RandomWalk()
    system.figure_7_2()
    # system.n_step_td(10, 10, 0.5)

