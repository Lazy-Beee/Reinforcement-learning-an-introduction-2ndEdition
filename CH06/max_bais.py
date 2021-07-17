"""Exercise 6.9 & 6.10"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class WindGridWorld:
    """
    The MDP has two non-terminal states A and B. Episodes
    always start in A with a choice between two actions, left and right. The right action
    transitions immediately to the terminal state with a reward and return of zero. The
    left action transitions to B, also with a reward of zero, from which there are many
    possible actions all of which cause immediate termination with a reward drawn from a
    normal distribution with mean -0.1 and variance 1.0
    """

    def __init__(self, episodes, repeat):
        """Initialize class"""
        self.episodes = episodes
        self.repeat = repeat
        self.b_terminal = [-0.1, 1]

        self.states = [0, 1, 2]
        self.actions = [[0, 1], [0, 1, 2, 3, 4], [0]]
        self.init_state = 0
        self.state = 0
        self.value_1 = np.zeros((3, 5))
        self.value_2 = np.zeros((3, 5))
        self.policy = np.zeros(2)
        self.action = 0
        self.moving = True

        self.discount = 1
        self.eps = 0.1
        self.step_size = 0.1
        self.sum_of_lefts = 0
        self.sum_of_actions = 0
        self.episodes_record = []

    def reset(self):
        """Reset class"""
        self.value_1 = np.zeros((3, 5))
        self.value_2 = np.zeros((3, 5))
        self.policy = np.zeros(2)
        self.action = 0
        self.sum_of_lefts = 0
        self.sum_of_actions = 0
        self.episodes_record = []

    def start(self):
        """Start a new episode."""
        self.state = self.init_state
        self.moving = True

    def step(self):
        """Move according to action. Return reward."""
        if self.state == 0:
            self.sum_of_actions += 1
            if self.action == 1:
                self.moving = False
                self.state = 2
                return 0
            else:
                self.state = 1
                self.sum_of_lefts += 1
                return 0
        else:
            self.moving = False
            self.state = 2
            return np.random.normal(self.b_terminal[0], self.b_terminal[1])

    def eps_greedy(self):
        """Determine action with eps-greedy."""
        value = self.value_1 + self.value_2
        self.policy = np.argmax(value, axis=-1)
        if value[0, 0] == value[0, 1]:
            self.policy[0] = np.random.choice([0, 1])

        if np.random.random() < self.eps:
            self.action = np.random.choice(len(self.actions[self.state]))
        else:
            self.action = self.policy[self.state]

    def q_learning(self, double=False):
        """Apply Q-Learning (off-policy TD control)."""
        self.reset()
        for _ in range(self.episodes):
            self.start()
            while self.moving:
                self.eps_greedy()
                state = self.state
                action = self.action
                reward = self.step()
                new_state = self.state
                if not double:
                    self.value_1[state, action] += self.step_size * (reward + self.discount
                                                                     * np.max(self.value_1[new_state])
                                                                     - self.value_1[state, action])
                else:
                    if np.random.choice([True, False]):
                        self.value_1[state, action] += self.step_size * \
                            (reward + self.discount * self.value_2[new_state, np.argmax(self.value_1[new_state])]
                             - self.value_1[state, action])
                    else:
                        self.value_2[state, action] += self.step_size * \
                            (reward + self.discount * self.value_1[new_state, np.argmax(self.value_2[new_state])]
                             - self.value_2[state, action])

            self.episodes_record.append(self.sum_of_lefts / self.sum_of_actions)
        return self.episodes_record

    def plot_rewards(self):
        """Plot route based on policy."""
        repeat = self.repeat
        rewards_q_d = np.zeros(self.episodes)
        rewards_q = np.zeros(self.episodes)
        for _ in trange(repeat):
            rewards_q_d += np.array(self.q_learning(double=True))
            rewards_q += np.array(self.q_learning())
        rewards_q_d /= repeat
        rewards_q /= repeat

        plt.figure()
        plt.plot(rewards_q, label="Q_learning")
        plt.plot(rewards_q_d, label="Double Q-learning")
        plt.xlabel('Episodes')
        plt.ylabel('Left actions from A')
        plt.legend()
        plt.savefig('images/max bias')
        plt.close()


if __name__ == '__main__':
    agent = WindGridWorld(300, 10000)
    agent.plot_rewards()






