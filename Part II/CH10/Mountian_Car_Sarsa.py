"""Example 10.1"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    def start_new_episode(self):
        """Set class ready for a new episode"""
        self.pos = np.random.uniform(low=self.pos_start[0], high=self.pos_start[1])
        self.vel = self.vel_start
        self.reached_terminal = False

    def eps_greedy(self):
        """Determine action according to eps-greedy policy"""
        if np.random.random() < self.eps:
            self.action = np.random.choice(self.actions)
        else:
            pos, vel = self.pos, self.vel
            values = []
            for action in self.actions:
                values.append(self.get_state_action_value(pos, vel, action))
            self.action = argmax(values)
        return self.action

    def move_one_time_step(self):
        """Move one time step, return reward"""
        pos, vel = self.pos, self.vel
        acc = self.action_acc[self.action]
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

    def traverse_all_state_values(self):
        positions = np.linspace(self.pos_bound[0], self.pos_bound[1], self.grid_size)
        velocities = np.linspace(self.vel_bound[0], self.vel_bound[1], self.grid_size)
        values = np.zeros((self.grid_size, self.grid_size))
        for i, pos in enumerate(positions):
            for j, vel in enumerate(velocities):
                val = []
                for action in self.actions:
                    val.append(self.get_state_action_value(pos, vel, action))
                values[i, j] = max(val)
        return - values

    def episodic_semi_gradient_sarsa(self, record_points=None):
        """Apply Episodic Semi_gradient Sarsa"""
        self.reset()
        value_record = []
        episode_length = []
        for i in range(self.max_episodes):
            count = 0
            self.start_new_episode()
            while not self.reached_terminal:
                pos, vel = self.pos, self.vel
                action = self.eps_greedy()
                new_pos, new_vel, reward = self.move_one_time_step()
                new_action = self.eps_greedy()

                value = self.get_state_action_value(pos, vel, action)
                value_diff = 1
                new_value = self.get_state_action_value(new_pos, new_vel, new_action)
                change = self.step_size * (reward + self.discount * new_value - value) * value_diff
                active_tiles = self.get_active_tiles(pos, vel, action)
                self.weight[active_tiles] += change
                count += 1
            episode_length.append(count)
            if record_points is not None and (i + 1) in record_points:
                value_record.append(self.traverse_all_state_values())
        return value_record, episode_length

    def figure_10_1(self):
        start_time = time.time()
        self.max_episodes = 10000
        episodes = np.power(10, np.arange(5))
        value_record, _ = self.episodic_semi_gradient_sarsa(episodes)

        fig = plt.figure(figsize=(10, 50))

        for i, value in enumerate(value_record):
            positions = np.linspace(self.pos_bound[0], self.pos_bound[1], self.grid_size)
            velocities = np.linspace(self.vel_bound[0], self.vel_bound[1], self.grid_size)
            ax = fig.add_subplot(5, 1, i+1, projection='3d')
            axis_x, axis_y = np.meshgrid(positions, velocities)
            ax.plot_wireframe(axis_x, axis_y, np.asarray(value_record[i]), color='blue', linewidth=0.3)
            ax.set_xlabel('Position')
            ax.set_ylabel('Velocity')
            ax.set_zlabel('Cost to go')
            ax.set_title(f'Episode {episodes[i]}')

        plt.suptitle(f'run time = {int(time.time() - start_time)} s', fontsize=30)
        plt.savefig('images/figure 10.1.png')
        plt.close()

    def figure_10_2(self):
        start_time = time.time()
        self.max_episodes = 500
        repeat = 100
        step_sizes = [0.1, 0.2, 0.5]
        step_number = np.zeros((len(step_sizes), self.max_episodes))
        for _ in trange(repeat):
            for i, step_size in enumerate(step_sizes):
                self.step_size = step_size / self.n_tilings
                _, episode_length = self.episodic_semi_gradient_sarsa()
                step_number[i, :] += episode_length
        step_number /= repeat

        plt.figure()
        for i in range(len(step_sizes)):
            plt.plot(step_number[i, :], label=f'Step Size = {step_sizes[i]}/{self.n_tilings}')
        plt.xlabel('Episodes')
        plt.ylabel('Run Time (s)')
        plt.ylim(100, 1000)
        plt.legend()

        plt.suptitle(f'run time = {int(time.time() - start_time)} s')
        plt.savefig(f'images/figure 10.2.png')
        plt.close()


if __name__ == "__main__":
    o = MountainCar()
    o.figure_10_2()











