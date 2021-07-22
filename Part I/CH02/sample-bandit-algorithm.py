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

    def __init__(self, eps=0, alpha=0.1, c=2, arm_num=10, params=(10, 10, 1),
                 gen_fig=True, method_used='', incremental_update=False, baseline=False):
        self.q = [0] * arm_num
        self.n = [0] * arm_num
        self.r = []
        self.options_count = arm_num
        self.eps = eps
        self.alpha = alpha
        self.c = c
        self.gen_fig = gen_fig
        # Method: ucb, gradient
        self.method_used = method_used.lower()
        self.incremental_update = incremental_update
        self.baseline = baseline
        self.runs = 0
        self.avg_r = 0
        self.best_action_count = 0
        self.action_prob = [0] * arm_num
        self.record = {"q": [], "n": [], "bac": [], "avg_r": [], "r": []}

        self.bandit_machine = MultiArmBandit(arm_num=arm_num, params=params)
        self.check_inputs()

    def check_inputs(self):
        if self.incremental_update and self.baseline:
            print("Can not simultaneously apply: <incremental update> & <baseline>")
            quit()

        if self.method_used != "ucb" and self.method_used != "gradient" and self.method_used != '':
            print(f"Can recognize method: {self.method_used}")
            quit()

        if self.method_used == "ucb" or self.method_used == "gradient":
            self.eps = 0

    def act(self):
        """Act based on possibilities, return the index of the chosen option"""
        if np.random.random() < self.eps:
            return np.random.choice(np.arange(self.options_count))
        elif self.method_used == 'ucb':
            return self.act_ucb()
        elif self.method_used == 'gradient':
            return self.act_gradient()
        else:
            return self._random_max_index(self.q)

    def act_ucb(self):
        """Return the index of the largest ucb value"""
        ucb_q = [0] * self.options_count
        for i in range(len(ucb_q)):
            ucb_q[i] = self.q[i] + self.c * np.sqrt(np.log(self.runs) / (self.n[i] + 1e-5))
        return self._random_max_index(ucb_q)

    def act_gradient(self):
        exp_p = np.exp(self.q)
        self.action_prob = exp_p / np.sum(exp_p)
        return np.random.choice(np.arange(self.options_count), p=self.action_prob)

    def step(self, curr_arm):
        """Perform an act and update parameters"""
        if curr_arm == self.bandit_machine.find_best_action():
            self.best_action_count += 1
        self.r.append(self.bandit_machine.bandit(curr_arm))
        self.avg_r += (self.r[-1] - self.avg_r) / self.runs
        self.n[curr_arm] += 1
        if self.incremental_update:
            self.q[curr_arm] += self.alpha * (self.r[-1] - self.q[curr_arm])
        elif self.baseline:
            one_hot = np.zeros(self.options_count)
            one_hot[curr_arm] = 1
            baseline = self.avg_r
            self.q += self.alpha * (self.r[-1] - baseline) * (one_hot - self.action_prob)
        else:
            self.q[curr_arm] += (self.r[-1] - self.q[curr_arm]) / self.n[curr_arm]

    def simulate(self, runs=1000):
        """Simulate, count and return how many times each option is chosen."""
        for _ in trange(runs):
            self.runs += 1
            curr_arm = self.act()
            self.step(curr_arm)
            self._record_data()

    def disp_result(self):
        """Output result to user"""
        result_distribution = (np.array(self.n) / self.runs).tolist()
        result_distribution = [float('%.1f' % (elem * 100)) for elem in result_distribution]
        q = [float('%.1f' % elem) for elem in self.q]
        bandit_machine_arms = [float('%.1f' % elem) for elem in self.bandit_machine.arms]

        print(f"Method: {self.method_used}")
        print(f"Iter: {self.runs}, \u03B5 = {self.eps}, \u03B1 = {self.alpha}")
        print(f"Actions:          {self.n}")
        print(f"Distribution(%):  {result_distribution}")
        print(f"Quality estimate: {q}")
        print(f"Actually quality: {bandit_machine_arms}")

        if self.gen_fig:
            steps = [i + 1 for i in range(self.runs)]

            # best option percentage vs. steps
            bac_percentage = [self.record["bac"][i]*100/(i+1) for i in range(self.runs)]
            plt.plot(steps, bac_percentage)
            plt.xlabel('steps')
            plt.ylabel('optimal action (%)')
            plt.savefig('images/optimal action vs steps.png')
            plt.show()
            plt.close()

            # best option percentage vs. steps
            plt.plot(steps, self.record["avg_r"])
            plt.xlabel('steps')
            plt.ylabel('average reward')
            plt.savefig('images/average reward vs steps.png')
            plt.show()
            plt.close()

    def _record_data(self):
        """Record data for each action"""
        self.record["q"].append(self.q)
        self.record["n"].append(self.n)
        self.record["bac"].append(self.best_action_count)
        self.record["avg_r"].append(self.avg_r)

    def _random_max_index(self, values):
        q_best = np.max(values)
        return np.random.choice(np.where(values == q_best)[0])


class MultiArmBandit:
    """
    A multi-arm bandit (arm 0 to n-1) with random parameters.
    For each arm pressed, returns an integer based on normal distribution of arm parameter.
    """
    def __init__(self, arm_num, params):
        self.arm_num = arm_num
        self.arms = [0] * arm_num
        # params = (mean, std of arm value, standard division of arm output)
        self.mean = params[0]
        self.std = params[1]
        self.range = params[2]
        self._init_arm_params()

    def _init_arm_params(self):
        """Generate arm parameters"""
        for i in range(self.arm_num):
            self.arms[i] = np.random.normal(self.mean, self.std)

    def bandit(self, arm_pressed):
        """Return result of arm_pressed"""
        arm_param = self.arms[arm_pressed]
        return np.random.normal(arm_param, self.range)

    def find_best_action(self):
        """Return arm with highest reward"""
        return self.arms.index(max(self.arms))


if __name__ == "__main__":
    test_machine = SampleMachine(gen_fig=True, method_used='gradient', incremental_update=False,
                                 baseline=True)
    test_machine.simulate(runs=1000)
    test_machine.disp_result()
