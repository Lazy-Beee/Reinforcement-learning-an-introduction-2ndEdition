"""Example 10.2"""

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def argmax(value):
    max_q = np.max(value)
    return np.random.choice([a for a, q in enumerate(value) if q == max_q])


class ServerOperator:
    """10 servers available!"""

    def __init__(self):
        """Initiate class"""
        # Process Parameters
        self.action = True
        self.customer = 0
        self.n_server_free = 0
        self.average_reward = 0

        # Settings
        self.n_server = 10
        self.p_sever_turn_free = 0.06
        self.small_threshold = 1e-4
        self.max_steps = 1000
        self.eps = 0.1
        self.step_size_alpha = 0.01
        self.step_size_beta = 0.01

        self.actions = [True, False]
        self.sever_states = [i for i in range(self.n_server + 1)]
        self.customer_states = [1, 2, 4, 8]

        # Records
        self.state_action_value = np.zeros((len(self.sever_states), len(self.customer_states), len(self.actions)))

    def set(self):
        """Reset/Set class parameters"""
        self.action = True
        self.customer = 0
        self.n_server_free = 0
        self.average_reward = 0

        self.state_action_value = np.zeros((len(self.sever_states), len(self.customer_states), len(self.actions)))

    def get_idx(self, n_server_free, customer, action):
        sever_idx = self.sever_states.index(n_server_free)
        customer_idx = self.customer_states.index(customer)
        action_idx = self.actions.index(action)
        return sever_idx, customer_idx, action_idx

    def eps_greedy(self):
        """Determine action according to eps-greedy policy"""
        if self.n_server_free == 0:
            self.action = False
        else:
            if np.random.random() < self.eps:
                self.action = np.random.choice(self.actions)
            else:
                sever_idx = self.sever_states.index(self.n_server_free)
                customer_idx = self.customer_states.index(self.customer)
                self.action = self.actions[argmax(self.state_action_value[sever_idx, customer_idx])]
        return self.action

    def prepare_next_time_step(self):
        """Determine server status and next customer"""
        self.customer = np.random.choice(self.customer_states)
        sever_turn_free = 0
        for _ in range(self.n_server - self.n_server_free):
            sever_turn_free += np.random.binomial(1, self.p_sever_turn_free)
        self.n_server_free += sever_turn_free
        return self.n_server_free, self.customer

    def move_one_time_step(self):
        """Move one time step, return reward"""
        # Deal with current customer
        if self.action:
            self.n_server_free -= 1
            reward = self.customer
        else:
            reward = 0
        return reward

    def differential_semi_gradient_sarsa(self):
        """Apply Differential Semi-gradient Sarsa"""
        self.set()
        self.prepare_next_time_step()
        self.eps_greedy()
        for _ in trange(self.max_steps):
            sever_state, customer_state, action = self.n_server_free, self.customer, self.action
            reward = self.move_one_time_step()
            new_sever_state, new_customer_state = self.prepare_next_time_step()
            new_action = self.eps_greedy()
            sever_idx, customer_idx, action_idx = self.get_idx(sever_state, customer_state, action)
            new_sever_idx, new_customer_idx, new_action_idx = self.get_idx(new_sever_state, new_customer_state,
                                                                           new_action)
            value = self.state_action_value[sever_idx, customer_idx, action_idx]
            new_value = self.state_action_value[new_sever_idx, new_customer_idx, new_action_idx]
            change = reward - self.average_reward + new_value - value
            self.average_reward += self.step_size_beta * change
            self.state_action_value[sever_idx, customer_idx, action_idx] += self.step_size_alpha * change * 1

    def figure_10_5(self):
        start_time = time.time()
        self.max_steps = 2000000
        self.differential_semi_gradient_sarsa()
        print(np.argmax(self.state_action_value[1:], axis=-1).transpose())
        values = np.max(self.state_action_value[1:], axis=-1).transpose()

        plt.figure()
        for i, value in enumerate(values):
            plt.plot(value, label=f'Priority = {self.customer_states[i]}')
        plt.xlabel('# of free servers')
        plt.ylabel('Value of best action')
        plt.legend()
        plt.suptitle(f'runs = {self.max_steps}, run time = {int(time.time() - start_time)} s')
        plt.savefig(f'images/figure 10.5.png')
        plt.close()

    def figure_10_5_2(self):
        start_time = time.time()
        plt.figure(figsize=(20, 40))
        for i, steps in enumerate(2 * np.power(10, np.arange(0, 8))):
            self.max_steps = steps
            self.differential_semi_gradient_sarsa()
            values = np.max(self.state_action_value[1:], axis=-1).transpose()
            plt.subplot(4, 2, i+1)
            for j, value in enumerate(values):
                plt.plot(value, label=f'Priority = {self.customer_states[j]}')
            plt.xlabel('# of free servers', fontsize=15)
            plt.ylabel('Value of best action', fontsize=15)
            plt.title(f'Steps = {steps}')
            plt.legend()
        plt.suptitle(f'run time = {int(time.time() - start_time)} s', fontsize=20)
        plt.savefig(f'images/figure 10.5.2.png')
        plt.close()


if __name__ == "__main__":
    a = ServerOperator()
    a.figure_10_5_2()
