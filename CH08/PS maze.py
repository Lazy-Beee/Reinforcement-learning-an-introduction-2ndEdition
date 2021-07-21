"""Example 8.4"""

import collections
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class Maze:
    """A 6*9 maze."""
    def __init__(self, repeat, size, n):
        """Initialize class, datum is at bottom left"""
        self.repeat = repeat
        self.size = size
        self.n = n

        self.world_size = [size, size]
        self.start_state = [0, 0]
        self.goal_state = [self.world_size[0] - 1, self.world_size[1] - 1]
        self.obstacles = []

        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        self.states = [[i, j] for i in range(self.world_size[0]) for j in range(self.world_size[1])]

        self.state_action_value = np.zeros((self.world_size[0], self.world_size[1], len(self.actions)))
        self.state_action_model = np.zeros((self.world_size[0], self.world_size[1], len(self.actions), 3))
        self.state_action_seen = np.zeros((len(self.states), len(self.actions)))
        self.policy = np.zeros((self.world_size[0], self.world_size[1]))
        self.state = self.start_state
        self.action = 0

        self.discount = 0.95
        self.eps = 0.1
        self.step_size = 0.1
        self.small_threshold = 1e-4
        self.total_steps = 0
        self.episode_finished = False
        self.p_queue = dict()
        self.predecessors = dict()

    def reset(self):
        """Reset class"""
        self.state_action_value = np.zeros((self.world_size[0], self.world_size[1], len(self.actions)))
        self.state_action_model = np.zeros((self.world_size[0], self.world_size[1], len(self.actions), 3))
        self.state_action_seen = np.zeros((len(self.states), len(self.actions)))
        self.policy = np.zeros((self.world_size[0], self.world_size[1]))
        self.state = self.start_state
        self.action = 0
        self.episode_finished = False
        self.total_steps = 0
        self.p_queue = dict()
        self.predecessors = dict()

    def start(self):
        """Start a new episode."""
        self.state = self.start_state
        self.episode_finished = False

    def move(self):
        """Move on grid according to action. Return reward."""
        pos_x, pos_y = self.state
        action_x, action_y = self.actions[self.action]
        new_pos_x = max(0, min(self.world_size[0] - 1, pos_x + action_x))
        new_pos_y = max(0, min(self.world_size[1] - 1, pos_y + action_y))
        new_state = [new_pos_x, new_pos_y]

        if new_state == self.goal_state:
            self.state = self.start_state
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
        return self.action

    def check_path(self):
        """Check if policy is optimized"""
        self.policy = np.argmax(self.state_action_value, axis=-1)
        state = self.start_state
        for _ in range(2 * self.size - 2):
            action = self.actions[self.policy[state[0], state[1]]]
            new_pos_x = max(0, min(self.world_size[0] - 1, state[0] + action[0]))
            new_pos_y = max(0, min(self.world_size[1] - 1, state[1] + action[1]))
            state = [new_pos_x, new_pos_y]
            if state == self.goal_state:
                return True
        return False

    def insert_pq(self, priority, state, action):
        """Insert state-action pairs into P-queue"""
        if priority > self.small_threshold:
            if [state[0], state[1], action] in self.p_queue.values():
                for k, v in self.p_queue.items():
                    if v == [state[0], state[1], action]:
                        del self.p_queue[k]
                        break
            self.p_queue[priority] = [state[0], state[1], action]

    def dyna_q(self):
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

            self.dyna_q_model_learning(self.n)

            if self.episode_finished:
                if self.check_path():
                    break
                self.start()
            # print(self.total_steps)
        return self.total_steps

    def dyna_q_model_learning(self, n):
        """Model learning in Dyna-Q process."""
        for _ in range(n):
            self.total_steps += 1

            state_idx = np.random.choice([i for i in range(len(self.state_action_seen))
                                          if sum(self.state_action_seen[i]) > 0])
            state = self.states[state_idx]
            action = np.random.choice([i for i in range(len(self.state_action_seen[state_idx]))
                                       if self.state_action_seen[state_idx][i] > 0])
            new_state = [10, 10]
            reward, new_state[0], new_state[1] = self.state_action_model[state[0], state[1], action]
            self.state_action_value[state[0], state[1], action] += self.step_size * \
                (reward + self.discount * np.max(self.state_action_value[int(new_state[0]), int(new_state[1])]) -
                 self.state_action_value[state[0], state[1], action])

    def prioritized_sweeping(self):
        """Apply Prioritized Sweeping"""
        self.reset()
        self.start()
        self.p_queue = collections.defaultdict(list)
        while True:
            state = self.state
            action = self.eps_greedy()
            new_state, reward = self.move()
            self.state_action_model[state[0], state[1], action] = [reward, new_state[0], new_state[1]]
            priority = np.abs(reward + self.discount * np.max(self.state_action_value[new_state[0], new_state[1], :])
                              - self.state_action_value[state[0], state[1], action])

            if tuple(new_state) not in self.predecessors.keys():
                self.predecessors[tuple(new_state)] = set()
            self.predecessors[tuple(new_state)].add((tuple(state), action))

            self.insert_pq(priority, state, action)
            self.ps_model_learning(self.n)

            if self.episode_finished:
                if self.check_path():
                    break
                self.start()
            # print(self.total_steps)
        return self.total_steps

    def ps_model_learning(self, n):
        """Model learning in Prioritized Sweeping."""
        if self.p_queue:
            for _ in range(n):
                self.total_steps += 1

                state = [0, 0]
                self.p_queue = dict(sorted(self.p_queue.items()))
                state[0], state[1], action = list(self.p_queue.values())[-1]
                del self.p_queue[list(self.p_queue.keys())[-1]]

                new_state = [0, 0]
                reward, new_state[0], new_state[1] = self.state_action_model[state[0], state[1], action]
                self.state_action_value[state[0], state[1], action] += self.step_size * \
                    (reward + self.discount * np.max(self.state_action_value[int(new_state[0]), int(new_state[1])])
                     - self.state_action_value[state[0], state[1], action])

                for prev in self.predecessors[tuple(state)]:
                    prev_state = list(prev[0])
                    prev_action = prev[1]
                    prev_reward, _, _ = self.state_action_model[prev_state[0], prev_state[1], prev_action]
                    priority = np.abs(prev_reward
                                      + self.discount * np.max(self.state_action_value[state[0], state[1], :])
                                      - self.state_action_value[prev_state[0], prev_state[1], prev_action])
                    self.insert_pq(priority, prev_state, prev_action)

                if not self.p_queue:
                    break

    def output(self):
        """Output average steps till optimal"""
        dyna_q, ps = 0, 0
        for _ in trange(self.repeat):
            dyna_q += self.dyna_q()
            ps += self.prioritized_sweeping()
        dyna_q /= self.repeat
        ps /= self.repeat

        return dyna_q, ps


def plot():
    repeat = 20
    n = 50
    sizes = np.power(2, np.arange(1, 6))
    dq = np.zeros(len(sizes))
    ps = np.zeros(len(sizes))
    for i, size in enumerate(sizes):
        agent = Maze(repeat=repeat, size=size, n=n)
        dq[i], ps[i] = agent.output()
    dq = np.log(dq)
    ps = np.log(ps)

    plt.figure()
    plt.plot(sizes, dq, label=f"Dyna-Q(n={n})")
    plt.plot(sizes, ps, label=f"Prioritized(n={n})")
    plt.title(f'runs={repeat}')
    plt.xlabel('Grid Size')
    plt.ylabel('Total Updates (10^)')
    plt.legend(loc='upper left')
    plt.savefig('images/maze-DQ-PS')
    plt.close()


if __name__ == '__main__':
    # agent = Maze(repeat=10, size=10, n=50)
    # print('Dyna-Q | Prioritized Sweeping')
    # print(agent.output())

    plot()