"""Exercise 6.9 & 6.10"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from matplotlib.table import Table


class WindGridWorld:
    """A grid world with wind blowing along each column."""
    def __init__(self, episodes, kings_move=False, random_wind=False):
        """Initialize class"""
        self.kings_move = kings_move
        self.random_wind = random_wind
        self.episodes = episodes

        if self.kings_move:
            self.actions = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        else:
            self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        self.world_size = [10, 7]
        self.start_position = [0, 3]
        self.target = [7, 3]
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.wind_variance = [1, 0, -1]

        self.state_action_value = np.zeros((self.world_size[0], self.world_size[1], 4))
        self.policy = np.zeros((self.world_size[0], self.world_size[1]))
        self.position = self.start_position
        self.action = 0
        self.moving = True

        self.discount = 1
        self.eps = 0.1
        self.step_size = 0.5
        self.policy_optimized = False

        self.time_steps = 0
        self.episodes_record = []

    def reset(self):
        """Reset class"""
        self.state_action_value = np.zeros((self.world_size[0], self.world_size[1], 4))
        self.policy = np.zeros((self.world_size[0], self.world_size[1]))
        self.position = self.start_position
        self.action = 0
        self.moving = True
        self.policy_optimized = False
        self.time_steps = 0
        self.episodes_record = []

    def start(self):
        """Start a new episode."""
        self.position = self.start_position
        self.moving = True

    def step(self):
        """Move on grid according to action. Return reward."""
        pos_x, pos_y = self.position
        action_x, action_y = self.actions[self.action]

        if self.random_wind:
            wind_effect = np.random.choice(self.wind_variance) + self.wind[pos_x]
        else:
            wind_effect = self.wind[pos_x]
        new_pos_x = max(0, min(self.world_size[0] - 1, pos_x + action_x))
        new_pos_y = max(0, min(self.world_size[1] - 1, pos_y + action_y + wind_effect))
        self.position = [new_pos_x, new_pos_y]

        if self.position == self.target:
            self.moving = False
            return 0
        return -1

    def eps_greedy(self):
        """Determine action with eps-greedy."""
        self.policy = np.argmax(self.state_action_value, axis=-1)
        if np.random.random() < self.eps:
            self.action = np.random.choice(len(self.actions))
        else:
            self.action = self.policy[self.position[0], self.position[1]]

    def sarsa(self):
        """Apply Sarsa (on-policy TD control), return policy"""
        self.reset()
        for _ in trange(self.episodes):
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

                self.time_steps += 1
            self.episodes_record.append(self.time_steps)

        # update policy before exit Sarsa
        self.eps_greedy()
        return self.policy

    def plot_policy_route(self):
        """Plot route based on policy."""
        if not self.policy_optimized:
            self.sarsa()

        self.start()
        route = [self.position]
        action = []
        while self.moving:
            self.eps_greedy()
            self.step()
            action.append(self.action)
            route.append(self.position)

        text = [["" for _ in range(self.world_size[0])] for _ in range(self.world_size[1])]
        for i in range(len(route) - 1):
            text[route[i][1]][route[i][0]] = str(i+1) + str(self.actions[action[i]])
        text[route[-1][1]][route[-1][0]] = f"{len(route)} END"

        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.table(cellText=text, colLabels=self.wind, cellLoc='center', loc='upper left')
        ax.set_title('matplotlib.axes.Axes.table() function Example', fontweight="bold")
        plt.savefig(f"images/wind grid world-king={self.kings_move}-random={self.random_wind}")
        plt.close()

        plt.figure()
        plt.plot(self.episodes_record, range(1, self.episodes + 1))
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.savefig('images/wind grid world time steps')
        plt.close()


if __name__ == '__main__':
    agent = WindGridWorld(100000, kings_move=False, random_wind=True)
    agent.plot_policy_route()






