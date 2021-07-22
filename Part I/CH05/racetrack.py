"""Exercise 5.12"""

import numpy as np
from tqdm import trange
from shapely.geometry import LineString
from matplotlib.table import Table
import matplotlib.pyplot as plt


class RaceCar:
    """A race car traveling on race track in Figure 5.5"""
    def __init__(self, energy_save_factor=0.1):
        self.acc = [1, 0, -1]
        self.actions = [[i, j] for i in self.acc for j in self.acc]
        self.speed_lower_limit = 0
        self.speed_upper_limit = 5
        self.speed = [0, 1, 2, 3, 4, 5]

        # Race track boundaries. Datum is at bottom left
        self.track_size = [12, 36]
        self.left_bound = LineString([(-0.5, -0.5), (-0.5, 35.5)])
        self.right_bound_1 = LineString([(5.5, -0.5), (5.5, 23.5)])
        self.right_bound_2 = LineString([(11.5, 23.5), (11.5, 35.5)])
        self.upper_bound = LineString([(-0.5, 35.5), (11.5, 35.5)])
        self.lower_bound_1 = LineString([(-0.5, -0.5), (5.5, -0.5)])
        self.lower_bound_2 = LineString([(5.5, 23.5), (11.5, 23.5)])
        self.start_line = self.lower_bound_1
        self.end_line = self.right_bound_2
        self.start_positions = [[i, 0] for i in range(6)]

        # State: x-pos, y-pos, x-vel, y-vel; Action: 0-8
        self.state_action_value = np.zeros((self.track_size[0], self.track_size[1], 6, 6, 9))
        self.state_action_count = np.zeros(self.state_action_value.shape, dtype=int)
        self.state_action_value_unweighted = np.zeros(self.state_action_value.shape)
        self.state_action_count_unweighted = np.zeros(self.state_action_value.shape, dtype=int)
        self.policy = np.zeros((self.track_size[0], self.track_size[1], 6, 6))
        self.policy[:, :, :, :] = 4
        self.position = [0, 0]
        self.velocity = [0, 0]
        self.action = [0, 0]
        self.racing = True

        self.energy_save_factor = energy_save_factor
        self.discount = 1
        self.policy_optimized = False
        self.start_point = []

    def start(self):
        """Initialize the race"""
        self.racing = True
        self.policy_optimized = False
        if self.start_point:
            self.position = self.start_point
            self.start_point = []
        else:
            idx = np.random.choice(6)
            self.position = self.start_positions[idx]
        self.velocity = [0, 0]
        self.action = [0, 0]

    def random_start(self):
        """Place the car in random state"""
        while self.position[0] >= 6 and self.position[1] <= 23:
            self.position = [np.random.choice(self.track_size[0]), np.random.choice(self.track_size[1])]
        self.velocity = [np.random.choice(self.speed), np.random.choice(self.speed)]

    def move(self):
        """Update state of race car based on pos, vel, acc. Return reward."""
        reward = - sum(self.action) * self.energy_save_factor
        pos_x, pos_y = self.position
        vel_x, vel_y = self.velocity
        acc_x, acc_y = self.action
        pos_old = (pos_x, pos_y)

        vel_x = max(self.speed_lower_limit, min(self.speed_upper_limit, vel_x + acc_x))
        vel_y = max(self.speed_lower_limit, min(self.speed_upper_limit, vel_y + acc_y))

        pos_x += vel_x
        pos_y += vel_y
        pos_new = (pos_x, pos_y)

        trajectory = LineString([pos_old, pos_new])
        if trajectory.intersects(self.left_bound) or trajectory.intersects(self.right_bound_1) or \
                trajectory.intersects(self.lower_bound_2) or trajectory.intersects(self.upper_bound):
            """
            Rather than start at random position at on the starting line, end the game when the car hit the 
            boundary to decrease run time.
            """
            # self.velocity = [0, 0]
            # self.position = self.start_positions[np.random.choice(len(self.start_positions))]

            self.position = [pos_x, pos_y]
            self.velocity = [vel_x, vel_y]
            self.racing = False
            return reward - 1000

        elif trajectory.intersects(self.end_line):
            self.racing = False
            return reward

        elif vel_x == 0 and vel_y == 0:
            self.racing = False
            return reward - 1000

        else:
            self.position = [pos_x, pos_y]
            self.velocity = [vel_x, vel_y]
            return reward - 1

    def race(self, policy, stochastic=True, output=False, random_start=True):
        """Simulate the full process of one race"""
        states = []
        actions = []
        rewards = []
        step = 0
        self.start()
        if random_start:
            self.random_start()
        while self.racing:
            pos_x, pos_y = self.position
            vel_x, vel_y = self.velocity
            if stochastic:
                action_idx = np.random.choice(np.arange(9), p=policy[pos_x, pos_y, vel_x, vel_y])
            else:
                action_idx = policy[pos_x, pos_y, vel_x, vel_y]
            self.action = self.actions[action_idx]
            states.append([pos_x, pos_y, vel_x, vel_y])
            actions.append(action_idx)
            rewards.append(self.move())

            if output:
                print(f'\nStep {step+1}:')
                print(f'pos: {[pos_x, pos_y]} -> {self.position}')
                print(f'vel: {[vel_x, vel_y]} -> {self.velocity}')
                print(f'acc: {self.action}')
                if rewards[-1] > -1 and not self.racing:
                    print('Reached end line!')
                elif not self.racing:
                    print('Ran off track.')
                step += 1

        return states, actions, rewards

    def behavior_policy(self, eps=0.2):
        """Generate behavior policy with equal possibility of actions in all states"""
        b = np.zeros((self.track_size[0], self.track_size[1], 6, 6, 9))
        b[:, :, :, :, :] = 1/9
        return b

    def off_policy(self, episodes, plot=False):
        """Find optimum policy with off-policy MC control"""
        b_policy = self.behavior_policy()
        for _ in trange(episodes):
            states, actions, rewards = self.race(b_policy, stochastic=True, output=False, random_start=True)
            returns = 0
            weight = 1
            for t in range(len(states) - 1, -1, -1):
                returns = self.discount * returns + rewards[t]
                pos_x, pos_y, vel_x, vel_y = states[t]
                action = actions[t]
                # Unweighted rewards
                self.state_action_count_unweighted[pos_x, pos_y, vel_x, vel_y, action] += 1
                self.state_action_value_unweighted[pos_x, pos_y, vel_x, vel_y, action] += \
                    (returns - self.state_action_value_unweighted[pos_x, pos_y, vel_x, vel_y, action]) / \
                    self.state_action_count_unweighted[pos_x, pos_y, vel_x, vel_y, action]
                # weighted rewards
                self.state_action_count[pos_x, pos_y, vel_x, vel_y, action] += weight
                self.state_action_value[pos_x, pos_y, vel_x, vel_y, action] += \
                    (returns - self.state_action_value[pos_x, pos_y, vel_x, vel_y, action]) * weight / \
                    self.state_action_count[pos_x, pos_y, vel_x, vel_y, action]

                self.policy[pos_x, pos_y, vel_x, vel_y] = \
                    np.argmax(self.state_action_value[pos_x, pos_y, vel_x, vel_y])
                if action != self.policy[pos_x, pos_y, vel_x, vel_y]:
                    break
                weight /= b_policy[pos_x, pos_y, vel_x, vel_y, action]

        if plot:
            for a in self.speed:
                for b in self.speed:
                    player.plot_policy([a, b])

    def find_optimum_policy(self):
        self.policy = np.argmax(self.state_action_value, axis=-1)
        print("Policy optimization complete")
        self.policy_optimized = True

    def optimum_route(self, start_point):
        self.start_point = start_point
        if not self.policy_optimized:
            self.find_optimum_policy()
        self.race(self.policy, stochastic=False, output=True, random_start=False)

    def plot_policy(self, vel):
        if not self.policy_optimized:
            self.find_optimum_policy()

        vel_x, vel_y = vel
        policy = self.policy[:, :, vel_x, vel_y]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        nrows, ncols = self.track_size[1], self.track_size[0]
        width, height = 1 / ncols, 1 / nrows

        for (i, j), val in np.ndenumerate(policy):
            if i < 6 or j > 23:
                tb.add_cell(j, i, width, height, text=self.actions[val], loc='center', facecolor='white')
            else:
                tb.add_cell(j, i, width, height, loc='center', facecolor='white')

        for i in range(self.track_size[1]):
            tb.add_cell(i, -1, width, height, text=i + 1, loc='right', edgecolor='none', facecolor='none')
        for i in range(self.track_size[0]):
            tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center', edgecolor='none', facecolor='none')
        ax.add_table(tb)
        plt.savefig(f'images/race track/race_track_{vel_x}_{vel_y}.png')
        plt.close()


if __name__ == "__main__":
    player = RaceCar()
    player.off_policy(1000000, plot=True)
    player.optimum_route([1, 0])
