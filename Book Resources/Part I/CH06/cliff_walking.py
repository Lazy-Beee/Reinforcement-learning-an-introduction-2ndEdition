"""Exercise 6.9 & 6.10"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class WindGridWorld:
    """A grid world with wind blowing along each column."""
    def __init__(self, episodes, repeat):
        """Initialize class"""
        self.episodes = episodes
        self.repeat = repeat

        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        self.world_size = [12, 4]
        self.start_position = [0, 0]
        self.target = [11, 0]
        self.cliff = [[i, 0] for i in range(1, self.world_size[0] - 1)]

        self.state_action_value = np.zeros((self.world_size[0], self.world_size[1], 4))
        self.policy = np.zeros((self.world_size[0], self.world_size[1]))
        self.position = self.start_position
        self.action = 0
        self.moving = True

        self.discount = 1
        self.eps = 0.1
        self.step_size = 0.5
        self.sum_of_rewards = 0
        self.episodes_record = []

    def reset(self):
        """Reset class"""
        self.state_action_value = np.zeros((self.world_size[0], self.world_size[1], 4))
        self.policy = np.zeros((self.world_size[0], self.world_size[1]))
        self.position = self.start_position
        self.action = 0
        self.moving = True
        self.sum_of_rewards = 0
        self.episodes_record = []

    def start(self):
        """Start a new episode."""
        self.position = self.start_position
        self.moving = True

    def step(self):
        """Move on grid according to action. Return reward."""
        pos_x, pos_y = self.position
        action_x, action_y = self.actions[self.action]
        new_pos_x = max(0, min(self.world_size[0] - 1, pos_x + action_x))
        new_pos_y = max(0, min(self.world_size[1] - 1, pos_y + action_y))
        self.position = [new_pos_x, new_pos_y]

        if self.position == self.target:
            self.moving = False
            return 0
        elif self.position in self.cliff:
            self.position = self.start_position
            return -100
        else:
            return -1

    def eps_greedy(self):
        """Determine action with eps-greedy."""
        self.policy = np.argmax(self.state_action_value, axis=-1)
        if np.random.random() < self.eps:
            self.action = np.random.choice(len(self.actions))
        else:
            self.action = self.policy[self.position[0], self.position[1]]

    def sarsa(self):
        """Apply Sarsa (on-policy TD control)."""
        self.reset()
        for _ in range(self.episodes):
            self.start()
            self.eps_greedy()
            while self.moving:
                state = self.position
                action = self.action
                reward = self.step()
                self.eps_greedy()
                new_state = self.position
                new_action = self.action
                self.state_action_value[state[0], state[1], action] += self.step_size * \
                    (reward + self.discount * self.state_action_value[new_state[0], new_state[1], new_action] -
                     self.state_action_value[state[0], state[1], action])

                self.sum_of_rewards += reward
            self.episodes_record.append(self.sum_of_rewards)
            self.sum_of_rewards = 0

        return self.episodes_record

    def q_learning(self):
        """Apply Q-Learning (off-policy TD control)."""
        self.reset()
        for _ in range(self.episodes):
            self.start()
            while self.moving:
                self.eps_greedy()
                state = self.position
                action = self.action
                reward = self.step()
                new_state = self.position
                self.state_action_value[state[0], state[1], action] += \
                    self.step_size * \
                    (reward + self.discount * np.max(self.state_action_value[new_state[0], new_state[1]]) -
                     self.state_action_value[state[0], state[1], action])

                self.sum_of_rewards += reward
            self.episodes_record.append(self.sum_of_rewards)
            self.sum_of_rewards = 0

        return self.episodes_record

    def plot_rewards(self):
        """Plot route based on policy."""
        repeat = self.repeat
        rewards_sarsa = np.zeros(self.episodes)
        rewards_q = np.zeros(self.episodes)
        for _ in trange(repeat):
            rewards_sarsa += np.array(self.sarsa())
            rewards_q += np.array(self.q_learning())
        rewards_sarsa /= repeat
        rewards_q /= repeat

        # Remove rewards less than -100
        rewards_sarsa = [max(i, -100) for i in rewards_sarsa]
        rewards_q = [max(i, -100) for i in rewards_q]

        plt.figure()
        plt.plot(rewards_sarsa, label="Sarsa")
        plt.plot(rewards_q, label="Q-learning")
        plt.xlabel('Episodes')
        plt.ylabel('Rewards during episode')
        plt.savefig('images/cliff walk')
        plt.close()


if __name__ == '__main__':
    agent = WindGridWorld(200, 500)
    agent.plot_rewards()






