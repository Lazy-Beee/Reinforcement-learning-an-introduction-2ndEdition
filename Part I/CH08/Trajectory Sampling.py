"""Figure 8.8"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import time


def argmax(value):
    max_q = np.max(value)
    return np.random.choice([a for a, q in enumerate(value) if q == max_q])


class Environment:
    def __init__(self, n_states, b, max_steps):
        self.n_states = n_states
        self.b = b
        self.max_steps = max_steps

        self.n_actions = 2
        self.termination_prob = 0.1

        self.map = np.random.randint(self.n_states, size=(self.n_states, self.n_actions, self.b))
        self.reward = np.random.randn(self.n_states, self.n_actions, self.b)

    def step(self, state, action):
        """Move in environment according to map, return reward and next state."""
        action = int(action)
        if np.random.random() < self.termination_prob:
            return self.n_states, 0
        else:
            next_state_b = np.random.randint(self.b)
            return self.map[state, action, next_state_b], self.reward[state, action, next_state_b]

    def next(self, state, action):
        """Return all possible rewards and next states."""
        return self.map[state, action, :], self.reward[state, action, :]


class Agent:
    def __init__(self, n_states, b, max_steps):
        self.n_states = n_states
        self.b = b
        self.max_steps = max_steps

        self.discount = 1
        self.eps = 0.1
        self.envi = Environment(self.n_states, self.b, self.max_steps)
        self.eval_frequency = max_steps / 50

        self.policy = np.zeros(self.n_states, dtype=int)
        self.state_action_values = np.zeros((self.n_states, self.envi.n_actions))
        self.action = 0
        self.state = 0

    def reset(self):
        """Reset class."""
        self.policy = np.zeros(self.n_states)
        self.state_action_values = np.zeros((self.n_states, self.envi.n_actions))
        self.action = 0
        self.state = 0

    def update_policy(self):
        """Update policy according to state action values"""
        for state in range(self.n_states):
            self.policy[state] = argmax(self.state_action_values[state, :])

    def evaluate_policy(self, runs=100):
        """Evaluate the value of first state under current policy."""
        self.update_policy()
        returns = []
        for _ in range(runs):
            state = 0
            rewards = 0
            while state != self.n_states:
                action = self.policy[state]
                state, reward = self.envi.step(state, action)
                rewards += reward
            returns.append(rewards)
        return np.average(returns)

    def eps_greedy(self):
        """Decide action based on eps-greedy."""
        self.update_policy()
        if np.random.random() < self.eps:
            self.action = np.random.randint(self.envi.n_actions)
        else:
            self.action = argmax(self.state_action_values[self.state])
        return self.action

    def uniform(self, random_sweep=True):
        """Apply uniform sampling"""
        self.reset()
        steps = []
        performance = []
        for step in range(self.max_steps):
            if random_sweep:
                state = np.random.randint(self.n_states)
                action = np.random.randint(self.envi.n_actions)
            else:
                state = step // self.envi.n_actions % self.n_states
                action = step % self.envi.n_actions

            next_states, rewards = self.envi.next(state, action)
            # Average all possible next_state_values since they have equal possibility of occurrence.
            self.state_action_values[state, action] = (1 - self.envi.termination_prob) * \
                np.average(rewards + self.discount * np.max(self.state_action_values[next_states, :], axis=1))

            if (step + 1) % self.eval_frequency == 0:
                steps.append(step)
                performance.append(self.evaluate_policy())
        return steps, performance

    def on_policy(self):
        """Apply uniform sampling"""
        self.reset()
        steps = []
        performance = []
        for step in range(self.max_steps):
            state = self.state
            action = self.eps_greedy()
            next_state, _ = self.envi.step(state, action)
            self.state = next_state % self.n_states

            next_states, rewards = self.envi.next(state, action)
            # Average all possible next_state_values since they have equal possibility of occurrence.
            self.state_action_values[state, action] = (1 - self.envi.termination_prob) * \
                np.average(rewards + self.discount * np.max(self.state_action_values[next_states, :], axis=1))

            if (step + 1) % self.eval_frequency == 0:
                steps.append(step)
                performance.append(self.evaluate_policy())

        return steps, performance


def figure():
    num_states = [100, 200]
    times = 20
    branch = [1, 10]
    repeat = 100
    start_time = time.time()

    plt.figure(figsize=(10, 10 * len(num_states)))
    for i, n in enumerate(num_states):
        plt.subplot(len(num_states), 1, i + 1)
        for b in branch:
            s = None
            value_u, value_op = [], []
            for _ in trange(repeat):
                agent = Agent(n_states=n, b=b, max_steps=n*times)
                s, p_u = agent.uniform(random_sweep=True)
                s, p_op = agent.on_policy()
                value_u.append(p_u)
                value_op.append(p_op)
            value_u = np.mean(np.asarray(value_u), axis=0)
            value_op = np.mean(np.asarray(value_op), axis=0)
            plt.plot(s, value_u, label=f'b = {b}, uniform')
            plt.plot(s, value_op, label=f'b = {b}, on-policy')
        plt.title(f'{n} states')
        plt.ylabel('value of start state')
        plt.legend()

    plt.subplot(len(num_states), 1, len(num_states))
    plt.xlabel('computation time, in expected updates')
    plt.suptitle(f'n = {repeat}, run time = {int(time.time()-start_time)} s')
    plt.savefig('images/figure_8_8.png')
    plt.close()


if __name__ == '__main__':
    figure()