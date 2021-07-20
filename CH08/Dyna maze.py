"""Example 8.1 (Dyna-Q+ not working)"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class Maze:
    """A 6*9 maze."""
    def __init__(self, runs, repeat, mode=1):
        """Initialize class, datum is at bottom left"""
        self.episodes = runs
        self.repeat = repeat
        self.mode = mode

        if self.mode == 1:
            self.world_size = [9, 6]
            self.start_state = [0, 3]
            self.goal_state = [8, 5]
            self.obstacles = [[2, 2], [2, 3], [2, 4], [5, 1], [7, 3], [7, 4], [7, 5]]
        elif self.mode == 2:
            self.world_size = [9, 6]
            self.start_state = [3, 0]
            self.goal_state = [8, 5]
            self.obstacles_before = [[i, 2] for i in range(8)]
            self.obstacles_after = [[i + 1, 2] for i in range(8)]
            self.obstacles = []
            self.change_steps = 1000
            self.max_steps = runs
        elif self.mode == 3:
            self.world_size = [9, 6]
            self.start_state = [3, 0]
            self.goal_state = [8, 5]
            self.obstacles_before = [[i, 2] for i in range(1, 9)]
            self.obstacles_after = [[i, 2] for i in range(1, 8)]
            self.obstacles = []
            self.change_steps = 3000
            self.max_steps = runs

        self.actions = [[0, -1], [0, 1], [1, 0], [-1, 0]]
        self.states = [[i, j] for i in range(self.world_size[0]) for j in range(self.world_size[1])]

        self.state_action_value = np.zeros((self.world_size[0], self.world_size[1], len(self.actions)))
        self.state_action_model = np.zeros((self.world_size[0], self.world_size[1], len(self.actions), 3))
        self.state_action_seen = np.zeros((len(self.states), len(self.actions)))
        self.state_action_bonus = np.zeros((len(self.states), len(self.actions)))
        self.policy = np.zeros((self.world_size[0], self.world_size[1]))
        self.state = self.start_state
        self.action = 0

        self.discount = 0.95
        self.eps = 0.1
        self.step_size = 0.1
        self.time_bonus = 1e-3
        self.total_steps = 0
        self.cumulative_reward = 0
        self.episode_finished = False
        self.episode_steps = 0
        self.episode_count = 0
        self.episode_record = []

    def reset(self):
        """Reset class"""
        self.state_action_value = np.zeros((self.world_size[0], self.world_size[1], len(self.actions)))
        self.state_action_model = np.zeros((self.world_size[0], self.world_size[1], len(self.actions), 3))
        self.state_action_seen = np.zeros((len(self.states), len(self.actions)))
        self.state_action_bonus = np.zeros((len(self.states), len(self.actions)))
        self.state_action_bonus[:, :] += 1
        self.policy = np.zeros((self.world_size[0], self.world_size[1]))
        self.state = self.start_state
        self.action = 0
        self.episode_finished = False
        self.episode_steps = 0
        self.episode_count = 0
        self.episode_record = []
        self.total_steps = 0
        self.cumulative_reward = 0

        if self.mode in [2, 3]:
            self.obstacles = self.obstacles_before

    def start(self):
        """Start a new episode."""
        self.state = self.start_state
        self.episode_steps = 0
        self.episode_finished = False

    def move(self):
        """Move on grid according to action. Return reward."""
        self.episode_steps += 1
        self.total_steps += 1
        pos_x, pos_y = self.state
        action_x, action_y = self.actions[self.action]
        new_pos_x = max(0, min(self.world_size[0] - 1, pos_x + action_x))
        new_pos_y = max(0, min(self.world_size[1] - 1, pos_y + action_y))
        new_state = [new_pos_x, new_pos_y]

        if self.mode in [2, 3]:
            if self.total_steps == self.change_steps:
                self.obstacles = self.obstacles_after

        if new_state == self.goal_state:
            self.state = self.start_state
            self.episode_count += 1
            self.episode_finished = True
            return self.state, 1
        elif new_state not in self.obstacles:
            self.state = new_state
        return self.state, 0

    def eps_greedy(self):
        """Determine action with eps-greedy, update target policy."""
        self.policy = np.argmax(self.state_action_value, axis=-1)
        if np.random.random() < self.eps:
            self.action = np.random.choice(len(self.actions))
        else:
            values = self.state_action_value[self.state[0], self.state[1], :]
            self.action = np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

    def dyna_q(self, n, plus=False):
        """Apply Dyna-Q method"""
        self.reset()
        self.start()
        while True:
            self.eps_greedy()
            state = self.state
            action = self.action
            new_state, reward = self.move()
            self.state_action_value[state[0], state[1], action] += self.step_size * \
                (reward + self.discount * np.max(self.state_action_value[new_state[0], new_state[1]]) -
                 self.state_action_value[state[0], state[1], action])
            self.state_action_model[state[0], state[1], action] = [reward, new_state[0], new_state[1]]
            state_idx = self.states.index(state)
            self.state_action_seen[state_idx, action] = 1
            self.cumulative_reward += reward

            self.state_action_bonus[state_idx, :] += 1
            self.state_action_bonus[state_idx, action] = 1
            self.model_learning(n, plus)

            if self.mode == 1:
                if self.episode_finished:
                    self.episode_record.append(self.episode_steps)
                    self.start()
                if self.episode_count >= self.episodes:
                    break
            elif self.mode in [2, 3]:
                self.episode_record.append(self.cumulative_reward)
                if self.total_steps >= self.max_steps:
                    break
        return self.episode_record

    def model_learning(self, n, plus):
        """Model learning in Dyna-Q process."""
        for _ in range(n):
            state_idx = np.random.choice([i for i in range(len(self.state_action_seen))
                                          if sum(self.state_action_seen[i]) > 0])
            state = self.states[state_idx]

            if plus:
                action = np.random.choice(len(self.actions))
                if self.state_action_seen[state_idx][action] == 0:
                    reward, new_state = 0, state
                else:
                    new_state = [10, 10]
                    reward, new_state[0], new_state[1] = self.state_action_model[state[0], state[1], action]
                reward += self.time_bonus * np.sqrt(self.state_action_bonus[state_idx, action])
            else:
                action = np.random.choice([i for i in range(len(self.state_action_seen[state_idx]))
                                           if self.state_action_seen[state_idx][i] > 0])
                new_state = [10, 10]
                reward, new_state[0], new_state[1] = self.state_action_model[state[0], state[1], action]

            self.state_action_value[state[0], state[1], action] += self.step_size * \
                (reward + self.discount * np.max(self.state_action_value[int(new_state[0]), int(new_state[1])]) -
                 self.state_action_value[state[0], state[1], action])

    def figure_8_2(self):
        """Plot efficiency of different planning steps."""
        n = [0, 5, 50]
        repeat = self.repeat
        steps_per_episode = np.zeros((len(n), self.episodes))
        for _ in trange(repeat):
            for i, step in enumerate(n):
                steps_per_episode[i] += np.array(self.dyna_q(step))
        steps_per_episode /= repeat

        plt.figure()
        for i, step in enumerate(n):
            plt.plot(range(1, self.episodes + 1), steps_per_episode[i], label=f"Dyna-Q(n={step})")
        plt.title(f'runs={self.repeat}')
        plt.xlabel('Episodes')
        plt.ylabel('Steps per episode')
        plt.legend(loc='upper right')
        plt.ylim(10, 600)
        plt.savefig('images/maze-Dyna-Q')
        plt.close()

    def figure_8_4(self):
        """Plot efficiency of different methods."""
        n = 50
        repeat = self.repeat
        rewards = np.zeros((2, self.episodes))
        for _ in trange(repeat):
            rewards[0] += np.array(self.dyna_q(n, plus=True))
            rewards[1] += np.array(self.dyna_q(n, plus=False))
        rewards /= repeat

        plt.figure()
        plt.plot(rewards[0], label=f"Dyna-Q+(n={n})")
        plt.plot(rewards[1], label=f"Dyna-Q(n={n})")
        plt.title(f'runs={self.repeat}')
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.legend(loc='upper left')
        plt.savefig('images/maze-Dyna-Q+1')
        plt.close()

    def figure_8_5(self):
        """Plot efficiency of different methods."""
        n = 50
        repeat = self.repeat
        rewards = np.zeros((2, self.episodes))
        for _ in trange(repeat):
            rewards[0] += np.array(self.dyna_q(n, plus=True))
            rewards[1] += np.array(self.dyna_q(n, plus=False))
        rewards /= repeat

        plt.figure()
        plt.plot(rewards[0], label=f"Dyna-Q+(n={n})")
        plt.plot(rewards[1], label=f"Dyna-Q(n={n})")
        plt.title(f'runs={self.repeat}')
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.legend(loc='upper left')
        plt.savefig('images/maze-Dyna-Q+2')
        plt.close()


if __name__ == '__main__':
    # agent1 = Maze(runs=30, repeat=100, mode=1)
    # agent1.figure_8_2()

    # agent2 = Maze(runs=3000, repeat=5, mode=2)
    # agent2.figure_8_4()

    # agent3 = Maze(runs=6000, repeat=5, mode=3)
    # agent3.figure_8_5()

    agent4 = Maze(runs=6000, repeat=5, mode=3)
    agent4.figure_8_5()
