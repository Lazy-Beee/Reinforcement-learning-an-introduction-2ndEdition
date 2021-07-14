"""Example 6.2"""

import numpy as np
import matplotlib.pyplot as plt


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
        self.value_difference = [0] * self.number_of_states

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

    def find_value_difference(self):
        for i in range(self.number_of_states):
            self.value_difference[i] = self.state_values[i] - self.true_values[i]

    def rms_error(self):
        """Find the total error of mid five state values compared with true values"""
        error = 0
        for i in range(1, self.number_of_states - 1):
            error += abs(self.state_values[i] - self.true_values[i])
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

            self.find_value_difference()
            error.append(np.sqrt(np.sum(np.power(self.value_difference[1:5], 2)) / (self.number_of_states - 2)))

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
            self.find_value_difference()
            error.append(np.sqrt(np.sum(np.power(self.value_difference[1:5], 2)) / (self.number_of_states - 2)))

        if not rms_error:
            return self.state_values[1:6]
        else:
            return error

    def figure_6_2_1(self, method='TD'):
        """Plot the result of value prediction"""
        episodes = [0, 1, 10, 100]
        plt.figure()
        for episode in episodes:
            if method == 'TD':
                values = self.tabular_td_0(episode)
            elif method == 'MC':
                values = self.constant_step_size_mc(episode)
            else:
                print(f'Cannot identify method "{method}".')
                exit()
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
        plt.figure()
        for step_size in td_step_size:
            plt.plot(self.tabular_td_0(episodes, step_size=step_size, rms_error=True),
                     linestyle='solid', label='TD, alpha = %.02f' % step_size)
        for step_size in mc_step_size:
            plt.plot(self.constant_step_size_mc(episodes, step_size=step_size, rms_error=True),
                     linestyle='dashdot', label='MC, alpha = %.02f' % step_size)
        plt.xlabel('episodes')
        plt.ylabel('RMS')
        plt.legend()

        plt.savefig('images/random_walk_2')
        plt.close()


if __name__ == '__main__':
    system = RandomWalk()
    system.figure_6_2_1()
    system.figure_6_2_2()


