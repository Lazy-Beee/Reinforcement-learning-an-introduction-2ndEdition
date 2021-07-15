"""Example 6.2"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


class RandomWalk:
    """
    All episodes start in the center state, C, then proceed either left
    or right by one state on each step, with equal probability. Episodes terminate either
    on the extreme left or the extreme right. When an episode terminates on the right,
    a reward of +1 occurs; all other rewards are zero
    """

    def __init__(self):
        """Define parameters of the class"""
        self.number_of_states = 7
        self.true_values = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
        self.state_values = [0] * self.number_of_states
        self.positions = [i - 1 for i in range(self.number_of_states)]
        self.states = [i - 1 for i in range(self.number_of_states)]
        self.policy = [0.5, 0.5]
        self.actions = [-1, 1]
        self.discount = 1

        self.position = 3
        self.reached_terminal = False
        self.action = 0
        self.small_threshold = 1e-3

    def reset_episode(self):
        """Reset class for a new episode"""
        self.position = 3
        self.action = 0
        self.reached_terminal = False

    def reset_state_values(self):
        self.state_values = [0] * self.number_of_states
        self.state_values[1:6] = [0.5] * 5

    def position_to_state(self):
        return self.positions.index(self.position)

    def rms_error(self):
        """Find the total error of mid five state values compared with true values"""
        error = 0
        for i in range(1, self.number_of_states - 1):
            error += np.square(self.state_values[i] - self.true_values[i])
        error = np.sqrt(error / (self.number_of_states - 2))
        return error

    def step(self):
        """Take a random move and end the episode if reached terminal. Return reward."""
        self.action = np.random.choice(self.actions, p=self.policy)
        self.position += self.action
        if self.position > 4:
            self.reached_terminal = True
            return 1
        elif self.position < 0:
            self.reached_terminal = True
        return 0

    def random_walk(self):
        """Simulate an episode"""
        states = []
        rewards = []
        actions = []

        self.reset_episode()
        while not self.reached_terminal:
            state = self.position_to_state()
            reward = self.step()
            action = self.action

            states.append(state)
            rewards.append(reward)
            actions.append(action)

        return states, rewards, actions

    def tabular_td_0(self, episodes, step_size=0.1, rms_error=False, get_full_episode=True):
        """Estimate value of states using TD(0), return state values and error"""
        error = []
        self.reset_state_values()
        for _ in range(episodes):
            self.reset_episode()
            if get_full_episode:
                states, rewards, _ = self.random_walk()
                for i in range(len(states)):
                    state = states[i]
                    reward = rewards[i]
                    if i == len(states) - 1:
                        new_state_value = 0
                    else:
                        new_state = states[i+1]
                        new_state_value = self.state_values[new_state]
                    self.state_values[state] += step_size * (reward + self.discount * new_state_value -
                                                             self.state_values[state])
            else:
                state = self.position_to_state()
                while not self.reached_terminal:
                    reward = self.step()
                    new_state = self.position_to_state()
                    self.state_values[state] += step_size * (reward + self.discount * self.state_values[new_state] -
                                                             self.state_values[state])
                    state = new_state
            error.append(self.rms_error())

        if not rms_error:
            return self.state_values[1:6]
        else:
            return error

    def constant_step_size_mc(self, episodes, step_size=0.01, rms_error=False):
        """Estimate value of states using MC, return state values and error"""
        error = []
        self.reset_state_values()
        for _ in range(episodes):
            states, rewards, _ = self.random_walk()
            returns = 0
            for i in range(len(states) - 1, -1, -1):
                state = states[i]
                returns = self.discount * returns + rewards[i]
                self.state_values[state] += step_size * (returns - self.state_values[state])
            error.append(self.rms_error())

        if not rms_error:
            return self.state_values[1:6]
        else:
            return error

    def td_batch(self, episodes, step_size=0.001, rms_error=False):
        """Estimate values with TD(batch), return values or RMS error"""
        error = []
        self.reset_state_values()
        states = []
        rewards = []
        for _ in range(episodes):
            self.reset_episode()
            curr_states, curr_rewards, _ = self.random_walk()
            states.append(curr_states)
            rewards.append(curr_rewards)
            while True:
                update = [0] * self.number_of_states
                for i in range(len(states)):
                    for j in range(len(states[i])):
                        state = states[i][j]
                        reward = rewards[i][j]
                        if j == len(states[i]) - 1:
                            new_state_value = 0
                        else:
                            new_state = states[i][j + 1]
                            new_state_value = self.state_values[new_state]
                        update[state] += step_size * (reward + self.discount * new_state_value - self.state_values[state])
                for i in range(self.number_of_states):
                    self.state_values[i] += update[i]
                if np.sum(np.abs(update)) < self.small_threshold:
                    break
            error.append(self.rms_error())

        if not rms_error:
            return self.state_values[1:6]
        else:
            return error

    def mc_batch(self, episodes, step_size=0.001, rms_error=False):
        """Estimate values with MC batch, return state values or RMS error"""
        error = []
        self.reset_state_values()
        states = []
        rewards = []
        for _ in range(episodes):
            curr_states, curr_rewards, _ = self.random_walk()
            states.append(curr_states)
            rewards.append(curr_rewards)
            while True:
                update = [0] * self.number_of_states
                for i in range(len(states)):
                    returns = 0
                    for j in range(len(states[i]) - 1, -1, -1):
                        state = states[i][j]
                        returns = self.discount * returns + rewards[i][j]
                        update[state] += step_size * (returns - self.state_values[state])
                for i in range(self.number_of_states):
                    self.state_values[i] += update[i]
                if np.sum(np.abs(update)) < self.small_threshold:
                    break
            error.append(self.rms_error())

        if not rms_error:
            return self.state_values[1:6]
        else:
            return error

    def figure_6_2_1(self, method='TD'):
        """Plot the result of value prediction"""
        episodes = [0, 1, 10, 100]
        repeat = 1000
        plt.figure()
        for episode in episodes:
            values = 0
            if method == 'TD':
                values = self.tabular_td_0(episode)
            else:
                values = self.constant_step_size_mc(episode)
            plt.plot(values, label=str(episode) + ' episode(s)')
        plt.plot(self.true_values[1:6], label='true values')
        plt.xlabel('state')
        plt.ylabel('estimated value')
        plt.legend()

        plt.savefig('images/random_walk_1')
        plt.close()

    def figure_6_2_2(self):
        """Compare TD and MC methods."""
        td_step_size = [0.15, 0.1, 0.05]
        mc_step_size = [0.01, 0.02, 0.03, 0.04]
        episodes = 100
        repeat = 100

        plt.figure()
        for step_size in tqdm(td_step_size):
            error_td = 0
            for i in range(repeat):
                error_td += (np.array(self.tabular_td_0(episodes, step_size=step_size, rms_error=True)) - error_td) /\
                            (i + 1)
            plt.plot(error_td, linestyle='solid', label='TD, alpha = %.02f' % step_size)
        for step_size in tqdm(mc_step_size):
            error_mc = 0
            for i in range(repeat):
                error_mc += (np.array(self.constant_step_size_mc(episodes, step_size=step_size, rms_error=True))
                             - error_mc) / (i + 1)
            plt.plot(error_mc, linestyle='solid', label='MC, alpha = %.02f' % step_size)
        plt.xlabel('episodes')
        plt.ylabel('RMS')
        plt.legend()

        plt.savefig('images/random_walk_2')
        plt.close()

    def figure_batch(self):
        """Compare batch training of TD and MC"""
        episodes = 100
        repeat = 1000

        td_error = 0
        mc_error = 0
        for i in trange(repeat):
            td_error += (np.array(self.td_batch(episodes, rms_error=True)) - td_error) / (i + 1)
            mc_error += (np.array(self.mc_batch(episodes, rms_error=True)) - mc_error) / (i + 1)

        plt.plot(td_error, label='TD')
        plt.plot(mc_error, label='MC')
        plt.xlabel('episodes')
        plt.ylabel('RMS error')
        plt.legend()

        plt.savefig('images/random_walk_batch.png')
        plt.close()


if __name__ == '__main__':
    system = RandomWalk()
    # system.figure_6_2_1()
    # system.figure_6_2_2()
    system.figure_batch()

