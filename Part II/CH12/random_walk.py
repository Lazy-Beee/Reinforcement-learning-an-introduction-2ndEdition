"""Figure 12.3"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import time


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

        self.features = np.zeros((self.number_of_states + 2, self.number_of_states + 2))
        for i in range(self.number_of_states):
            self.features[i+1, i+1] = 1

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

    def random_walk(self):
        """Simulate an episode"""
        self.start_episode()
        states = [self.state]
        rewards = [0]
        while not self.reached_terminal:
            rewards.append(self.step())
            states.append(self.state)
        return states, rewards

    def n_step_td(self, episode, step, time_step):
        """Estimate state values under given policy using n-step TD."""
        self.reset()
        for _ in range(episode):
            self.start_episode()
            terminal = float('inf')
            time_, update_time = 0, 0
            states = [self.state]
            rewards = [0]
            while True:
                if time_ < terminal:
                    rewards.append(self.step())
                    states.append(self.state)
                    if self.reached_terminal:
                        terminal = time_ + 1
                update_time = time_ - step + 1
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
                time_ += 1
        return self.rms_error()

    def n_step_return(self, states, t, n, rewards=None, non_discount=True):
        """Return n-step return(non-discount/discount)"""
        if non_discount:
            if rewards is None:
                return self.state_values[states[t+n]]
            else:
                # Include rewards only if h can be terminal (in case of random walk).
                return self.state_values[states[t+n]] + np.sum(rewards[t+1:t+n+1])
        g = 0
        for i in range(n):
            idx = t + i + 1
            g += rewards[idx] * np.power(self.discount, i)
        g += self.state_values[states[t + n]] * np.power(self.discount, n)
        return g

    def complete_return(self, rewards, t=None, terminal=None):
        """Return complete return"""
        if t is None and terminal is None:
            return rewards[-1]
        g = 0
        for i in range(terminal - t):
            idx = t + i + 1
            g += rewards[idx] * np.power(self.discount, i)
        return g

    def lambda_return(self, states, rewards, t, terminal, lamb):
        """Return lambda-return"""
        lamb_return = 0
        for n in range(1, terminal - t):
            if np.power(lamb, n - 1) < self.small_threshold:
                return lamb_return
            lamb_return += (1 - lamb) * np.power(lamb, n - 1) * self.n_step_return(states, t, n)
        lamb_return += np.power(lamb, terminal - t - 1) * self.complete_return(rewards)
        return lamb_return

    def h_lambda_return(self, states, rewards, t, h, lamb):
        """Return truncated-lambda-return"""
        lamb_return = 0
        for n in range(1, h - t):
            if np.power(lamb, n - 1) < self.small_threshold:
                return lamb_return
            lamb_return += (1 - lamb) * np.power(lamb, n - 1) * self.n_step_return(states, t, n)
        lamb_return += np.power(lamb, h - t - 1) * self.n_step_return(states, t, h-t, rewards)
        return lamb_return

    def offline_lambda_return(self, episode, lamb, step_size):
        """Estimate state values under given policy using Off-line lambda-return."""
        self.reset()
        for _ in range(episode):
            states, rewards = self.random_walk()
            terminal = len(states)
            for t in range(terminal - 1):
                lamb_return = self.lambda_return(states, rewards, t, terminal, lamb)
                state_value, state_value_diff = self.state_values[states[t]], 1
                self.state_values[states[t]] += step_size * (lamb_return - state_value) * state_value_diff
        return self.rms_error()

    def semi_gradient_td_lambda(self, episode, lamb, step_size):
        self.reset()
        for _ in range(episode):
            self.start_episode()
            trace = np.zeros(self.state_values.shape)
            while not self.reached_terminal:
                state = self.state
                reward = self.step()
                next_state = self.state

                trace = self.discount * lamb * trace
                trace[state] += 1
                delta = reward + self.discount * self.state_values[next_state] - self.state_values[state]
                self.state_values += step_size * delta * trace

        return self.rms_error()

    def online_td_lambda(self, episode, lamb, step_size):
        self.reset()
        for _ in range(episode):
            self.start_episode()
            states = [self.state]
            rewards = [0]
            h = 0
            while not self.reached_terminal:
                h += 1
                rewards.append(self.step())
                states.append(self.state)
                for t in range(h):
                    state = states[t]
                    change = step_size * (self.h_lambda_return(states, rewards, t, h, lamb) - self.state_values[state])
                    self.state_values[state] += change

        return self.rms_error()

    def true_online_td_lambda(self, episode, lamb, step_size):
        self.reset()
        for _ in range(episode):
            self.start_episode()
            trace = np.zeros(self.number_of_states + 2)
            value_old = 0
            while True:
                feature = self.features[self.state]
                reward = self.step()
                feature_next = self.features[self.state]
                value = np.dot(self.state_values, feature)
                value_next = np.dot(self.state_values, feature_next)

                delta = reward + self.discount * value_next - value
                trace *= self.discount * lamb
                trace += (1 - step_size * self.discount * lamb * np.dot(trace, feature)) * feature
                self.state_values += step_size * ((delta + value - value_old) * trace
                                                  - (value - value_old) * feature)
                value_old = value_next
                if sum(feature_next) == 0:
                    break

        return self.rms_error()

    def figure_12_3(self):
        """Plot the rms error of value prediction"""
        start_time = time.time()
        episode = 10
        repeat = 100
        steps = np.power(2, np.arange(1, 5))
        lambs = [0.4, 0.8, 0.9, 0.95]
        time_steps = np.logspace(-2, 0, num=20)
        errors = np.zeros((len(steps), len(time_steps)))
        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        for i in range(len(lambs)):
            lamb = lambs[i]
            time_r = time.time()
            for j in trange(len(time_steps)):
                time_step = time_steps[j]
                for k in range(repeat):
                    errors[i, j] += self.offline_lambda_return(episode, lamb, time_step)
                errors[i, j] /= repeat
            plt.plot(time_steps, errors[i, :], label=f'\u03BB={lamb}, t={int(time.time() - time_r)}s')
        plt.xlabel('Step Size')
        plt.ylabel('RMS Error')
        plt.title('Off-line \u03BB-return')
        plt.ylim([0.1, 0.6])
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        for i in range(len(steps)):
            step = steps[i]
            time_r = time.time()
            for j in trange(len(time_steps)):
                time_step = time_steps[j]
                for k in range(repeat):
                    errors[i, j] += self.n_step_td(episode, step, time_step)
                errors[i, j] /= repeat
            plt.plot(time_steps, errors[i, :], label=f'n={step}, t={int(time.time() - time_r)}s')
        plt.xlabel('Step Size')
        plt.ylabel('RMS Error')
        plt.title('n-step TD')
        plt.ylim([0.1, 0.6])
        plt.legend(loc='upper right')

        plt.suptitle(f't = {int(time.time() - start_time)}s')
        plt.savefig('images/figure 12.3.png')
        plt.close()

    def figure_12_6(self):
        """Plot the rms error of value prediction"""
        start_time = time.time()
        episode = 10
        repeat = 100
        lambs = [0, 0.4, 0.8, 0.9, 0.95]
        time_steps = np.linspace(0, 1, num=100)
        errors = np.zeros((len(lambs), len(time_steps)))
        plt.figure(figsize=(10, 8))

        for i in range(len(lambs)):
            lamb = lambs[i]
            time_r = time.time()
            for j in trange(len(time_steps)):
                time_step = time_steps[j]
                for k in range(repeat):
                    errors[i, j] += self.semi_gradient_td_lambda(episode, lamb, time_step)
                errors[i, j] /= repeat
            plt.plot(time_steps, errors[i, :], label=f'\u03BB={lamb}, t={int(time.time() - time_r)}s')
        plt.xlabel('Step Size')
        plt.ylabel('RMS Error')
        plt.ylim([0.1, 0.6])
        plt.legend(loc='upper right')

        plt.title(f'TD(\u03BB) t = {int(time.time() - start_time)}s')
        plt.savefig('images/figure 13.1.png')
        plt.close()

    def figure_12_8(self):
        """Plot the rms error of value prediction"""
        start_time = time.time()
        episode = 10
        repeat = 10
        lambs = [0, 0.4, 0.8, 0.9, 0.95]
        time_steps = np.linspace(0, 1, num=20)
        errors = np.zeros((len(lambs), len(time_steps)))
        plt.figure(figsize=(20, 8))

        plt.subplot(1, 2, 1)
        for i in range(len(lambs)):
            lamb = lambs[i]
            time_r = time.time()
            for j in trange(len(time_steps)):
                time_step = time_steps[j]
                for k in range(repeat):
                    errors[i, j] += self.online_td_lambda(episode, lamb, time_step)
                errors[i, j] /= repeat
            plt.plot(time_steps, errors[i, :], label=f'\u03BB={lamb}, t={int(time.time() - time_r)}s')
        plt.xlabel('Step Size')
        plt.ylabel('RMS Error')
        plt.title(f'On-line \u03BB-return algorithm ({repeat*len(lambs)*len(time_steps)} runs)')
        plt.ylim([0.1, 0.6])
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        for i in range(len(lambs)):
            lamb = lambs[i]
            time_r = time.time()
            for j in trange(len(time_steps)):
                time_step = time_steps[j]
                for k in range(repeat):
                    errors[i, j] += self.semi_gradient_td_lambda(episode, lamb, time_step)
                errors[i, j] /= repeat
            plt.plot(time_steps, errors[i, :], label=f'\u03BB={lamb}, t={int(time.time() - time_r)}s')
        plt.xlabel('Step Size')
        plt.ylabel('RMS Error')
        plt.title(f'Off-line \u03BB-return algorithm ({repeat*len(lambs)*len(time_steps)} runs)')
        plt.ylim([0.1, 0.6])
        plt.legend(loc='upper right')

        plt.suptitle(f'TD(\u03BB) t = {int(time.time() - start_time)}s')
        plt.savefig('images/figure 12.8.png')
        plt.close()

    def figure_12_8_2(self):
        """Plot the rms error of value prediction"""
        start_time = time.time()
        episode = 10
        repeat = 1000
        lambs = [0, 0.4, 0.8, 0.9, 0.95]
        time_steps = np.linspace(0, 1, num=30)
        errors = np.zeros((len(lambs), len(time_steps)))
        plt.figure(figsize=(20, 8))

        plt.subplot(1, 2, 1)
        for i in range(len(lambs)):
            lamb = lambs[i]
            time_r = time.time()
            for j in trange(len(time_steps)):
                time_step = time_steps[j]
                for k in range(repeat):
                    errors[i, j] += self.true_online_td_lambda(episode, lamb, time_step)
                errors[i, j] /= repeat
            plt.plot(time_steps, errors[i, :], label=f'\u03BB={lamb}, t={int(time.time() - time_r)}s')
        plt.xlabel('Step Size')
        plt.ylabel('RMS Error')
        plt.title(f'True on-line TD(\u03BB) ({repeat*len(lambs)*len(time_steps)} runs)')
        plt.ylim([0.1, 0.6])
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        for i in range(len(lambs)):
            lamb = lambs[i]
            time_r = time.time()
            for j in trange(len(time_steps)):
                time_step = time_steps[j]
                for k in range(repeat):
                    errors[i, j] += self.semi_gradient_td_lambda(episode, lamb, time_step)
                errors[i, j] /= repeat
            plt.plot(time_steps, errors[i, :], label=f'\u03BB={lamb}, t={int(time.time() - time_r)}s')
        plt.xlabel('Step Size')
        plt.ylabel('RMS Error')
        plt.title(f'Off-line \u03BB-return algorithm ({repeat*len(lambs)*len(time_steps)} runs)')
        plt.ylim([0.1, 0.6])
        plt.legend(loc='upper right')

        plt.suptitle(f'TD(\u03BB) t = {int(time.time() - start_time)}s')
        plt.savefig('images/figure 12.8.2.png')
        plt.close()


if __name__ == '__main__':
    system = RandomWalk()
    system.figure_12_8_2()
