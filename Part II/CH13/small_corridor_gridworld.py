"""Example 13.1"""

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


class GridWorld:
    def __init__(self):
        """Small corridor grid world."""
        # Process Parameters
        self.state = 0
        self.action = 0
        self.reached_terminal = False

        # Settings
        self.world_size = 4
        self.states = np.arange(self.world_size)
        self.normal_states = [0, 2]
        self.reversed_states = [1]
        self.end_state = self.states[-1]
        self.actions_move = [-1, 1]
        self.n_actions = 2
        self.actions = np.arange(self.n_actions)

    def start_new_episode(self):
        """Setup environment for new episode."""
        self.state = 0
        self.action = 0
        self.reached_terminal = False

    def move(self, state=None, action=None):
        """Move one step. Return reward and next state."""
        if state is None:
            state = self.state
        if action is None:
            action = self.action
        if state in self.normal_states:
            self.state = max(state + self.actions_move[action], self.states[0])
        elif state in self.reversed_states:
            self.state = state - self.actions_move[action]
        else:
            print(f"invalid state idx: {state}")
            quit()
        if self.state == self.end_state:
            self.reached_terminal = True
        return -1, self.state

    def full_episode(self, prob):
        """Run full episode with given probability of actions"""
        self.start_new_episode()
        total_reward = 0
        while not self.reached_terminal:
            self.action = np.random.choice(self.actions, p=prob)
            reward, _ = self.move()
            total_reward += reward
        return total_reward


class MCPolicyGradientControl:
    def __init__(self, step_size, environment=GridWorld()):
        """Monte-Carlo Policy-Gradient Control (episodic)."""
        # Settings
        self.env = environment
        self.step_size = step_size
        self.feature = [[1, 0], [0, 1]]
        self.discount = 1
        self.eps = 0.05

        # records
        self.theta = [0, 0]

    def feature_vector(self, state=None, action=None):
        """Return feature vector of state-action pair."""
        if state is None:
            state = self.env.state
        if action is None:
            action = self.env.action
        return self.feature[action]

    def preferences_func(self, state=None, action=None):
        """Calculate parameterized numerical preferences."""
        return np.dot(self.theta, self.feature_vector(state, action))

    def policy_prob(self, state=None):
        """Calculate policy with softmax."""
        prob = []
        den = 0
        for a in self.env.actions:
            den += np.exp(self.preferences_func(state, a))
        for action in self.env.actions:
            num = np.exp(self.preferences_func(state, action))
            prob.append(num / den)
        if min(prob) < self.eps:
            min_idx = np.argmin(prob)
            prob[:] = [1 - self.eps] * self.env.n_actions
            prob[min_idx] = self.eps
        return prob

    def get_action(self, state=None):
        """Determine action according to policy"""
        self.env.action = np.random.choice(self.env.actions, p=self.policy_prob(state))
        return self.env.action

    def full_episode(self):
        """Return full episode states, actions, rewards"""
        self.env.start_new_episode()
        states, actions, rewards = [], [], [0]
        while not self.env.reached_terminal:
            states.append(self.env.state)
            actions.append(self.get_action())
            rewards.append(self.env.move()[0])
        return states, actions, rewards

    def learn(self, states, actions, rewards):
        """Update policy using one full episode"""
        terminal = len(states)
        for t in range(terminal):
            state, action = states[t], actions[t]
            g = 0
            for k in range(t+1, terminal+1):
                g += np.power(self.discount, k-t-1) * rewards[k]
            gradient_ln_policy = self.feature_vector(state, action) - np.dot(self.feature, self.policy_prob(state))
            self.theta += self.step_size * np.power(self.discount, t) * g * gradient_ln_policy

    def policy_gradient_control(self, episode):
        """Optimize policy"""
        self.theta = [-1.47, 1.47]
        rewards_record = []
        for _ in range(episode):
            states, actions, rewards = self.full_episode()
            self.learn(states, actions, rewards)
            rewards_record.append(np.sum(rewards))
        return rewards_record


def example_13_1():
    start_time = time.time()
    n_points = 1000
    repeat = 1000
    right_prob = np.linspace(0.01, 0.99, n_points)
    env = GridWorld()
    data = np.zeros(n_points)
    for _ in trange(repeat):
        for i, prob in enumerate(right_prob):
            data[i] += env.full_episode([1-prob, prob])
    data /= repeat
    plt.figure(figsize=(10, 8))
    plt.plot(right_prob, data)
    plt.xlabel('Probability of moving right')
    plt.ylabel('Value')
    plt.ylim(-100, 0)
    plt.title(f'repeat = {repeat} t = {int(time.time() - start_time)}s')
    plt.savefig('images/example 13.1.png')
    plt.close()


def figure_13_1():
    """Plot figure 13.1"""
    start_time = time.time()
    episode = 1000
    repeat = 100
    alpha_power = [12, 13, 14]
    data = np.zeros((len(alpha_power), episode))
    plt.figure(figsize=(10, 8))

    for _ in trange(repeat):
        for j in range(len(alpha_power)):
            agent = MCPolicyGradientControl(1/np.power(2, alpha_power[j]))
            data[j, :] += agent.policy_gradient_control(episode)
    data /= repeat
    for i in range(len(alpha_power)):
        plt.plot(data[i, :], label=f'step_size = 2^{-alpha_power[i]}')
    plt.xlabel('Episode')
    plt.ylabel('Total reward on episode')
    plt.ylim(-60, 0)
    plt.legend(loc='lower right')

    plt.title(f'repeat = {repeat} t = {int(time.time() - start_time)}s')
    plt.savefig('images/figure 12.6.png')
    plt.close()


if __name__ == "__main__":
    # example_13_1()
    figure_13_1()




