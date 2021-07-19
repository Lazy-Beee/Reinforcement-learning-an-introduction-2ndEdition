"""Cliff walking with n-step methods"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class Cliff:
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
        self.b_policy = [1/len(self.actions)] * len(self.actions)

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
        """Determine action with eps-greedy, update target policy."""
        self.policy = np.argmax(self.state_action_value, axis=-1)
        if np.random.random() < self.eps:
            self.action = np.random.choice(len(self.actions))
        else:
            self.action = self.policy[self.position[0], self.position[1]]

    def behavior_policy(self):
        """Determine action with behavior policy, update target policy."""
        self.action = np.random.choice(len(self.actions), p=self.b_policy)

    def update_policy(self):
        """Determine action with behavior policy, update target policy."""
        self.policy = np.argmax(self.state_action_value, axis=-1)

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

    def n_step_sarsa(self, step):
        """Apply n-step Sarsa (on-policy TD control)."""
        self.reset()
        for _ in range(self.episodes):
            self.start()
            self.eps_greedy()
            terminal = float('inf')
            time, update_time = 0, 0
            states = [self.position]
            rewards = [0]
            actions = [self.action]
            while True:
                if time < terminal:
                    rewards.append(self.step())
                    states.append(self.position)
                    self.sum_of_rewards += rewards[-1]
                    if not self.moving:
                        terminal = time + 1
                    else:
                        self.eps_greedy()
                        actions.append(self.action)
                update_time = time - step + 1
                if update_time >= 0:
                    returns = 0
                    for j in range(update_time + 1, min(update_time + step, terminal) + 1):
                        returns += (self.discount ** (j - update_time - 1)) * rewards[j]
                    if update_time + step < terminal:
                        returns += (self.discount ** step) * self.state_action_value[states[update_time + step][0],
                                                                                     states[update_time + step][1],
                                                                                     actions[update_time + step]]
                    state, action = states[update_time], actions[update_time]
                    self.state_action_value[state[0], state[1], action] += \
                        self.step_size * (returns - self.state_action_value[state[0], state[1], action])
                if update_time == terminal - 1:
                    break
                time += 1
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

    def n_step_sarsa_off_policy(self, step):
        """Apply n-step Sarsa (off-policy TD control)."""
        self.reset()
        for _ in range(self.episodes):
            self.start()
            self.behavior_policy()
            terminal = float('inf')
            time, update_time = 0, 0
            states = [self.position]
            rewards = [0]
            actions = [self.action]
            while True:
                if time < terminal:
                    rewards.append(self.step())
                    states.append(self.position)
                    if not self.moving:
                        terminal = time + 1
                    else:
                        self.behavior_policy()
                        actions.append(self.action)
                update_time = time - step + 1
                if update_time >= 0:
                    returns = 0
                    weight = 1
                    for i in range(update_time + 1, min(update_time + step, terminal)):
                        state, action = states[i], actions[i]
                        weight *= self.policy[state[0], state[1]] / self.b_policy[action]
                    for j in range(update_time + 1, min(update_time + step, terminal) + 1):
                        returns += (self.discount ** (j - update_time - 1)) * rewards[j]
                    if update_time + step < terminal:
                        returns += (self.discount ** step) * self.state_action_value[states[update_time + step][0],
                                                                                     states[update_time + step][1],
                                                                                     actions[update_time + step]]
                    state, action = states[update_time], actions[update_time]
                    self.state_action_value[state[0], state[1], action] += \
                        self.step_size * weight * (returns - self.state_action_value[state[0], state[1], action])
                    self.update_policy()
                if update_time == terminal - 1:
                    break
                time += 1

            # Evaluate policy
            while self.moving:
                self.eps_greedy()
                reward = self.step()
                self.sum_of_rewards += reward

            self.episodes_record.append(self.sum_of_rewards)
            self.sum_of_rewards = 0
        return self.episodes_record

    def plot_rewards(self):
        """Plot route based on policy."""
        n = np.power(2, np.arange(0, 4))
        repeat = self.repeat
        rewards_sarsa = np.zeros(self.episodes)
        rewards_n_sarsa = np.zeros((len(n), self.episodes))
        # rewards_n_o_sarsa = np.zeros(self.episodes)
        rewards_q = np.zeros(self.episodes)
        for _ in trange(repeat):
            rewards_sarsa += np.array(self.sarsa())
            for i, step in enumerate(n):
                rewards_n_sarsa[i] += np.array(self.n_step_sarsa(step))
            # rewards_n_o_sarsa += np.array(self.n_step_sarsa_off_policy(10))
            rewards_q += np.array(self.q_learning())
        rewards_sarsa /= repeat
        rewards_n_sarsa /= repeat
        # rewards_n_o_sarsa /= repeat
        rewards_q /= repeat

        # Remove rewards less than -100
        rewards_sarsa = [max(i, -100) for i in rewards_sarsa]
        rewards_n_sarsa = [[max(i, -100) for i in row] for row in rewards_n_sarsa]
        # rewards_n_o_sarsa = [max(i, -100) for i in rewards_n_sarsa]
        rewards_q = [max(i, -100) for i in rewards_q]

        plt.figure()
        plt.plot(rewards_sarsa, label="Sarsa(0)")
        for i, step in enumerate(n):
            plt.plot(rewards_n_sarsa[i], label=f"Sarsa(n={step})")
        # plt.plot(rewards_n_sarsa, label="Off-policy Sarsa(n)")
        plt.plot(rewards_q, label="Q-learning")
        plt.title(f'Cliff Walking results (runs={self.repeat})')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards during episode')
        plt.legend(loc='lower right')
        plt.savefig('images/cliff walk - n-step Sarsa')
        plt.close()


if __name__ == '__main__':
    agent = Cliff(episodes=200, repeat=10000)
    agent.plot_rewards()
    # print(agent.n_step_sarsa(10))





