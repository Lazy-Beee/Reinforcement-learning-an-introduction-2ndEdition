"""Exercise 5.12"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from shapely.geometry import LineString

class RaceCar:
    """A race car traveling on race track in Figure 5.5"""
    def __init__(self, energy_save_factor=0):
        self.acc = [1, 0, -1]
        self.actions = [[i, j] for i in self.acc for j in self.acc]
        self.speed_lower_limit = 0
        self.speed_upper_limit = 5

        # Race track boundaries. Datum is at bottom left
        self.track_size = [12, 30]
        self.left_bound = LineString([(-0.5, -0.5), (-0.5, 29.5)])
        self.right_bound_1 = LineString([(5.5, -0.5), (5.5, 23.5)])
        self.right_bound_2 = LineString([(11.5, 23.5), (11.5, 29.5)])
        self.upper_bound = LineString([(-0.5, 29.5), (11.5, 29.5)])
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
        self.position = [0, 0]
        self.velocity = [0, 0]
        self.action = [0, 0]
        self.target_policy = np.zeros((12, 30, 6, 6))
        self.racing = True

        self.energy_save_factor = energy_save_factor
        self.discount = 1

    def start(self):
        """Initialize the race"""
        self.racing = True
        idx = np.random.choice(6)
        self.position = self.start_positions[idx]
        self.velocity = [0, 0]
        self.action = [0, 0]

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
            self.velocity = [0, 0]
            self.position = np.random.choice(self.start_positions)
            return reward - 1
        elif trajectory.intersects(self.end_line):
            self.racing = False
            return reward
        else:
            self.position = [pos_x, pos_y]
            self.velocity = [vel_x, vel_y]
            return reward - 1

    def race(self, policy, stochastic=True):
        """Simulate the full process of one race"""
        states = []
        actions = []
        rewards = []

        self.start()
        while self.racing:
            pos_x, pos_y = self.position
            vel_x, vel_y = self.velocity
            if stochastic:
                action_idx = np.random.choice(np.arange(9), p=policy[pos_x, pos_y, vel_x, vel_y])
                # action_idx = np.random.choice(9)
            else:
                action_idx = policy[pos_x, pos_y, vel_x, vel_y]
            self.action = self.actions[action_idx]
            states.append([pos_x, pos_y, vel_x, vel_y])
            actions.append(action_idx)
            rewards.append(self.move())

        return states, actions, rewards

    def behavior_policy(self):
        """Generate behavior policy"""
        b = np.zeros((self.track_size[0], self.track_size[1], 6, 6, 9))

        b[0:6, 0:24, :, :, :] = 0.0625
        b[0:6, 0:24, :, :, 4] = 0.5

        b[0:6, 24:30, :, :, :] = 0.0625
        b[0:6, 24:30, :, :, 1] = 0.5

        b[6:12, 0:24, :, :, :] = 0.0625
        b[6:12, 0:24, :, :, 2] = 0.5
        return b

    def off_policy(self, episodes):
        """Find optimum policy with off-policy MC control"""
        b_policy = self.behavior_policy
        for _ in trange(episodes):
            states, actions, rewards = self.race(b_policy)
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

    def optimum_route(self, start_point):
        self.start()
        self.position = start_point
        self.policy = np.argmax(self.state_action_value, axis=-1)
        states, actions, rewards = self.race(self.policy, stochastic=False)
        for i in range(rewards):
            print(f'\nStep {i}:')
            print(f'pos: {[states[i][0], states[i][1]]}')
            print(f'vel: {[states[i][2], states[i][3]]}')
            action = self.actions[actions[i]]
            print(f'vel: {action}')


if __name__ == "__main__":
    player = RaceCar()
    player.off_policy(10000)
    player.optimum_route([0, 3])
