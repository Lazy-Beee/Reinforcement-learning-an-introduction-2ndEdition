"""Example 10.2"""

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def argmax(value):
    max_q = np.max(value)
    return np.random.choice([x for x, q in enumerate(value) if q == max_q])


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
        self.p_server_turn_free = 0.06
        self.small_threshold = 1e-4
        self.max_steps = 1000
        self.eps = 0.1
        self.step_size_alpha = 0.01
        self.step_size_beta = 0.01

        self.actions = [True, False]
        self.server_states = [i for i in range(self.n_server + 1)]
        self.customer_states = [1, 2, 4, 8]

        # Records
        self.state_action_value = np.zeros((len(self.server_states), len(self.customer_states), len(self.actions)))

    def set(self):
        """Reset/Set class parameters"""
        self.action = True
        self.customer = 0
        self.n_server_free = 0
        self.average_reward = 0

        self.state_action_value = np.zeros((len(self.server_states), len(self.customer_states), len(self.actions)))

    def get_idx(self, n_server_free, customer, action):
        server_idx = self.server_states.index(n_server_free)
        customer_idx = self.customer_states.index(customer)
        action_idx = self.actions.index(action)
        return server_idx, customer_idx, action_idx

    def eps_greedy(self):
        """Determine action according to eps-greedy policy"""
        if self.n_server_free == 0:
            self.action = False
        else:
            if np.random.random() < self.eps:
                self.action = np.random.choice(self.actions)
            else:
                server_idx = self.server_states.index(self.n_server_free)
                customer_idx = self.customer_states.index(self.customer)
                self.action = self.actions[argmax(self.state_action_value[server_idx, customer_idx])]
        return self.action

    def prepare_next_time_step(self):
        """Determine server status and next customer"""
        self.customer = np.random.choice(self.customer_states)
        server_turn_free = 0
        for _ in range(self.n_server - self.n_server_free):
            server_turn_free += np.random.binomial(1, self.p_server_turn_free)
        self.n_server_free += server_turn_free
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
            server_state, customer_state, action = self.n_server_free, self.customer, self.action
            reward = self.move_one_time_step()
            new_server_state, new_customer_state = self.prepare_next_time_step()
            new_action = self.eps_greedy()
            server_idx, customer_idx, action_idx = self.get_idx(server_state, customer_state, action)
            new_server_idx, new_customer_idx, new_action_idx = self.get_idx(new_server_state, new_customer_state,
                                                                            new_action)
            value = self.state_action_value[server_idx, customer_idx, action_idx]
            new_value = self.state_action_value[new_server_idx, new_customer_idx, new_action_idx]
            change = reward - self.average_reward + new_value - value
            self.average_reward += self.step_size_beta * change
            self.state_action_value[server_idx, customer_idx, action_idx] += self.step_size_alpha * change * 1
    
    def differential_semi_gradient_n_step_sarsa(self, n):
        """Apply Differential Semi-gradient Sarsa"""
        self.set()
        self.prepare_next_time_step()
        self.eps_greedy()
        server_state, customer_state, action = self.n_server_free, self.customer, self.action
        server_idx, customer_idx, action_idx = self.get_idx(server_state, customer_state, action)
        div = n + 1
        server_ids = [server_idx] * div
        customer_ids = [customer_idx] * div
        action_ids = [action_idx] * div
        rewards = [0] * div
        for t in trange(self.max_steps):
            reward = self.move_one_time_step()
            new_server_state, new_customer_state = self.prepare_next_time_step()
            new_action = self.eps_greedy()
            new_server_idx, new_customer_idx, new_action_idx = self.get_idx(new_server_state, new_customer_state,
                                                                            new_action)
            server_ids[(t + 1) % div] = new_server_idx
            customer_ids[(t + 1) % div] = new_customer_idx
            action_ids[(t + 1) % div] = new_action_idx
            rewards[(t + 1) % div] = reward

            update_t = t - n + 1
            if update_t >= 0:
                change = 0
                for i in range(update_t + 1, update_t + n + 1):
                    change += rewards[i % div] - self.average_reward
                server_idx, next_server_idx = server_ids[update_t % div], server_ids[(update_t + n) % div]
                customer_idx, next_customer_idx = customer_ids[update_t % div], customer_ids[(update_t + n) % div]
                action_idx, next_action_idx = action_ids[update_t % div], action_ids[(update_t + n) % div]
                value = self.state_action_value[server_idx, customer_idx, action_idx]
                next_value = self.state_action_value[next_server_idx, next_customer_idx, next_action_idx]
                change += next_value - value
                self.average_reward += self.step_size_beta * change
                self.state_action_value[server_idx, customer_idx, action_idx] += self.step_size_alpha * change * 1
    
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

    def figure_10_5_3(self):
        start_time = time.time()
        plt.figure(figsize=(20, 30))
        ns = [10, 20, 40, 80]
        for i, steps in enumerate(2 * np.power(10, np.arange(1, 7))):
            self.max_steps = steps
            time_mark = time.time()
            self.differential_semi_gradient_sarsa()
            time_passed = int(time.time() - time_mark)
            values = np.max(self.state_action_value[1:], axis=-1).transpose()
            plt.subplot(3, 2, i+1)
            plt.plot(values[-1], '--', label=f'Priority = {self.customer_states[-1]}, Sarsa(0), {time_passed}s')
            for j, n in enumerate(ns):
                time_mark = time.time()
                self.differential_semi_gradient_n_step_sarsa(n)
                time_passed = int(time.time() - time_mark)
                values = np.max(self.state_action_value[1:], axis=-1).transpose()
                plt.plot(values[-1], label=f'{ns[j]}-step Sarsa, {time_passed}s')
            plt.xlabel('# of free servers', fontsize=25)
            plt.ylabel('Value of best action', fontsize=25)
            plt.title(f'Steps = {steps}')
            plt.legend(fontsize=15)
        plt.suptitle(f'run time = {int(time.time() - start_time)} s', fontsize=30)
        plt.savefig(f'images/figure 10.5.3.png')
        plt.close()


if __name__ == "__main__":
    a = ServerOperator()
    a.figure_10_5_3()
