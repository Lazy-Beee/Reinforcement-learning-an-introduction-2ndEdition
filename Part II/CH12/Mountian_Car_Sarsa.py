"""Example 10.1"""

import time
import numpy as np
import matplotlib.pyplot as plt
from TileCodingSoftware import *
from tqdm import trange


def bound(val, bou):
    """Bound function: restrict parameter in given boundary"""
    return max(bou[0], min(bou[1], val))


def argmax(value):
    max_q = np.max(value)
    return np.random.choice([a for a, q in enumerate(value) if q == max_q])


class MountainCar:
    """Under powered car in a valley"""

    def __init__(self):
        """Initiate class"""
        # Process Parameters
        self.pos = 0
        self.vel = 0
        self.action = 0
        self.reached_terminal = False

        # Settings
        self.pos_bound = [-1.2, 0.5]
        self.terminal = self.pos_bound[1]
        self.vel_bound = [-0.07, 0.07]
        self.pos_start = [-0.6, -0.4]
        self.vel_start = 0
        self.action_acc = [-1, 0, 1]
        self.actions = np.arange(len(self.action_acc))
        self.small_threshold = 1e-4
        self.max_episodes = 1000
        self.eps = 0.1
        self.discount = 1
        self.n_tilings = 8
        self.step_size = 0.3 / self.n_tilings
        self.grid_size = 100

        # Records
        self.weight = []

        # Tiling
        self.max_size = 4096
        self.iht = IHT(self.max_size)
        self.pos_scale = self.n_tilings / abs(self.pos_bound[1] - self.pos_bound[0])
        self.vel_scale = self.n_tilings / abs(self.vel_bound[1] - self.vel_bound[0])

    def reset(self):
        """Reset/Set class parameters"""
        self.weight = np.zeros(self.max_size)
        self.iht = IHT(self.max_size)

    def start_new_episode(self):
        """Set class ready for a new episode"""
        self.pos = np.random.uniform(low=self.pos_start[0], high=self.pos_start[1])
        self.vel = self.vel_start
        self.reached_terminal = False

    def eps_greedy(self, pos, vel):
        """Determine action according to eps-greedy policy"""
        if np.random.random() < self.eps:
            self.action = np.random.choice(self.actions)
        else:
            values = []
            for action in self.actions:
                values.append(self.get_state_action_value(pos, vel, action))
            self.action = argmax(values)
        return self.action

    def move_one_time_step(self, pos, vel, action):
        """Move one time step, return reward"""
        acc = self.action_acc[action]
        new_vel = bound(vel + 0.001 * acc - 0.0025 * np.cos(3 * pos), self.vel_bound)
        new_pos = bound(pos + new_vel, self.pos_bound)
        if new_pos == self.pos_bound[0]:
            new_vel = 0
        elif new_pos == self.terminal:
            self.reached_terminal = True
        self.pos, self.vel = new_pos, new_vel
        return new_pos, new_vel, -1

    def get_active_tiles(self, pos, vel, action):
        """Obtain active tiles of given state-action pair"""
        active_tiles = tiles(self.iht, self.n_tilings, [pos * self.pos_scale, vel * self.vel_scale], [action])
        return active_tiles

    def get_state_action_value(self, pos, vel, action):
        """Approximate state-action value"""
        active_tiles = self.get_active_tiles(pos, vel, action)
        if pos == self.terminal:
            return 0
        else:
            return np.sum(self.weight[active_tiles])

    def sarsa_lambda(self, lamb):
        """Sarsa(lambda) with binary features and linear function approximation"""
        self.reset()
        episode_length = []
        for _ in range(self.max_episodes):
            count = 0
            self.start_new_episode()
            z = np.zeros(self.max_size)
            pos, vel = self.pos, self.vel
            action = self.eps_greedy(pos, vel)
            while True:
                count += 1
                pos_next, vel_next, reward = self.move_one_time_step(pos, vel, action)
                delta = reward

                for i in self.get_active_tiles(pos, vel, action):
                    delta -= self.weight[i]
                    # Replacing traces
                    z[i] = 1
                    # Accumulating traces
                    # z[i] += 1

                if self.reached_terminal:
                    self.weight += self.step_size * delta * z
                    break

                action_next = self.eps_greedy(pos_next, vel_next)
                delta += self.discount * np.sum(self.weight[self.get_active_tiles(pos_next, vel_next, action_next)])
                self.weight += self.step_size * delta * z
                z *= self.discount * lamb
                pos, vel, action = pos_next, vel_next, action_next

            episode_length.append(count)
        return episode_length

    def figure_12_10(self):
        start_time = time.time()
        self.max_episodes = 50
        repeat = 1
        steps = [0, 0.68, 0.84, 0.96, 0.98]
        time_steps = np.linspace(0.5, 1, 10)
        runs = np.zeros((len(steps), len(time_steps)))
        plt.figure()
        for _ in range(repeat):
            for i in range(len(steps)):
                step = steps[i]
                for j in trange(len(time_steps)):
                    self.step_size = time_steps[j] / self.n_tilings
                    runs[i, j] += np.mean(self.sarsa_lambda(step))
        runs /= repeat

        for i, step in enumerate(steps):
            plt.plot(time_steps, runs[i, :], label='n=' + str(step))

        plt.xlabel('Step size * 8 (number of tilings)')
        plt.ylabel('Average steps over 50th episodes')
        plt.ylim(100, 400)
        plt.title(f'run time = {int(time.time() - start_time)} s')
        plt.legend()

        plt.savefig('images/figure 12.10.png')
        plt.close()


if __name__ == "__main__":
    o = MountainCar()
    o.figure_12_10()
