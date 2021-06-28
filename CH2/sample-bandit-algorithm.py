"""
Sample bandit algorithm
Page 32: Pseudocode for a complete bandit algorithm using incrementally computed sample
averages and "eps-greedy action selection is shown in the box below. The function bandit(a)
is assumed to take an action and return a corresponding reward.
"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class SampleMachine:
    """
    When activated, for a probability of eps, select a random option.
    Otherwise select the option with largest estimated reward.
    """

    def __init__(self, eps=0.1, c=2, arm_num=10, params=(20, 50, 5),
                 gen_fig=False, ucb_used=False):
        self.q = [0] * arm_num
        self.n = [0] * arm_num
        self.r = []
        self.options_count = arm_num
        self.eps = eps
        self.c = c
        self.gen_fig = gen_fig
        self.ucb_used = ucb_used
        self.runs = 0
        self.avg_r = 0
        self.best_action_count = 0
        self.record = {"q": [], "n": [], "bac": [], "avg_r": []}

        self.bandit_machine = MultiArmBandit(arm_num=arm_num, params=params)
        # self.best_action = self.bandit_machine.find_best_action()

    def act_eps_greedy(self):
        """Act based on possibilities, return the index of the chosen option"""
        if np.random.random() >= self.eps:
            return self._random_max_index(self.q)
        else:
            return np.random.randint(0, self.options_count)

    def act_ucb(self):
        """Return the index of the largest ucb value"""
        ucb_q = [0] * self.options_count
        for i in range(len(ucb_q)):
            ucb_q[i] = self.q[i] + self.c * np.sqrt(np.log(self.runs) / (self.n[i] + 1e-5))
        return self._random_max_index(ucb_q)

    def step(self, curr_arm):
        """Perform an act and update parameters"""
        if curr_arm == self.bandit_machine.find_best_action():
            self.best_action_count += 1
        self.r.append(self.bandit_machine.bandit(curr_arm))
        self.avg_r = sum(self.r)/self.runs
        self.n[curr_arm] += 1
        self.q[curr_arm] += (self.r[-1] - self.q[curr_arm]) / self.n[curr_arm]

    def simulate(self, runs=1000):
        """Simulate, count and return how many times each option is chosen."""
        for i in trange(runs):
            self.runs += 1
            if self.ucb_used:
                curr_arm = self.act_ucb()
            else:
                curr_arm = self.act_eps_greedy()
            self.step(curr_arm)
            self._record_data()

    def disp_result(self):
        """Output result to user"""
        result_distribution = (np.array(self.n)/self.runs).tolist()
        result_distribution = [float('%.2f' % elem) for elem in result_distribution]
        q = [float('%.2f' % elem) for elem in self.q]

        print(f"Iter: {self.runs}, \u03B5 = {self.eps}: {self.eps}")
        print(f"Actions: {self.n}")
        print(f"Distribution: {result_distribution}")
        print(f"Quality estimate: {q}")
        print(f"Actually quality: {self.bandit_machine.arms}")

        if self.gen_fig:
            steps = [i + 1 for i in range(self.runs)]

            # best option percentage vs. steps
            bac_percentage = [self.record["bac"][i]*100/(i+1) for i in range(self.runs)]
            plt.plot(steps, bac_percentage)
            plt.xlabel('steps')
            plt.ylabel('optimal action (%)')
            plt.savefig('image/optimal action vs steps.png')
            plt.show()
            plt.close()

            # best option percentage vs. steps
            plt.plot(steps, self.record["avg_r"])
            plt.xlabel('steps')
            plt.ylabel('average reward')
            plt.savefig('image/average reward vs steps.png')
            plt.show()
            plt.close()

    def _record_data(self):
        """Record data for each action"""
        self.record["q"].append(self.q)
        self.record["n"].append(self.n)
        self.record["bac"].append(self.best_action_count)
        self.record["avg_r"].append(self.avg_r)

    def _random_max_index(self, values):
        max_val = np.max(values)
        i, ind = 0, []
        for val in values:
            if val == max_val:
                ind.append(i)
            i = i + 1
        if len(ind) > 1:
            ind = np.random.choice(ind)
        else:
            ind = ind[0]
        return ind


class MultiArmBandit:
    """
    A multi-arm bandit (arm 0 to n-1) with random parameters.
    For each arm pressed, returns an integer based on normal distribution of arm parameter.
    """
    def __init__(self, arm_num, params):
        self.arm_num = arm_num
        self.arms = [0] * arm_num
        # params = (lower boundary, upper boundary, standard division) of arm results
        self.lower_bound = params[0]
        self.upper_bound = params[1]
        self.range = params[2]
        self._init_arm_params()

    def _init_arm_params(self):
        """Generate arm parameters"""
        for i in range(self.arm_num):
            self.arms[i] = np.random.randint(self.lower_bound, self.upper_bound)

    def bandit(self, arm_pressed):
        """Return result of arm_pressed"""
        arm_param = self.arms[arm_pressed]
        return np.random.randint(arm_param - self.range, arm_param + self.range + 1)

    def find_best_action(self):
        """Return arm with highest reward"""
        return self.arms.index(max(self.arms))


if __name__ == "__main__":
    test_machine = SampleMachine(gen_fig=True, ucb_used=True)
    test_machine.simulate(runs=10000)
    test_machine.disp_result()
