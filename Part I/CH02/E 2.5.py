"""
Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the
difficulties that sample-average methods have for non-stationary problems. Use a modified
version of the 10-armed testbed in which all the q*(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean 0
and standard deviation 0.01 to all the q*(a) on each step). Prepare plots like Figure 2.2
for an action-value method using sample averages, incrementally computed, and another
action-value method using a constant step-size parameter, alpha = 0.1. Use " eps = 0.1 and
longer runs, say of 10,000 steps.
"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class EpsGreedyMachine:
    """
    When activated, for a probability of eps, select a random option.
    Otherwise select the option with largest estimated reward.
    """

    def __init__(self, eps=0.1, alpha=0.1, arm_num=10, params=(20, 50, 0.01), gen_fig=False):
        self.q = [0] * arm_num
        self.n = [0] * arm_num
        self.r = []
        self.options_count = arm_num
        self.eps = eps
        self.alpha = alpha
        self.gen_fig = gen_fig
        self.runs = 0
        self.avg_r = 0
        self.best_action_count = 0
        self.record = {"q": [], "n": [], "bac": [], "avg_r": []}

        self.bandit_machine = MultiArmBandit(arm_num=arm_num, params=params)
        # Code suspended: arm values are constantly changing
        # self.best_action = self.bandit_machine.find_best_action()

    def act(self):
        """Act based on possibilities, return the index of the chosen option"""
        if np.random.random() >= self.eps:
            max_val = np.max(self.q)
            i, ind = 0, []
            for val in self.q:
                if val == max_val:
                    ind.append(i)
                i = i + 1
            if len(ind) > 1:
                ind = np.random.choice(ind)
            else:
                ind = ind[0]
            return ind
        else:
            return np.random.randint(0, self.options_count)

    def step(self):
        """Perform an act and update parameters"""
        curr_arm = self.act()
        if curr_arm == self.bandit_machine.find_best_action():
            self.best_action_count += 1
        self.r.append(self.bandit_machine.bandit(curr_arm))
        self.avg_r = sum(self.r)/self.runs
        self.n[curr_arm] += 1
        self.q[curr_arm] += self.alpha * (self.r[-1] - self.q[curr_arm]) / self.n[curr_arm]
        return self.act()

    def simulate(self, runs=1000):
        """Simulate, count and return how many times each option is chosen."""
        for i in trange(runs):
            self.runs = i + 1
            self.step()
            self._record_data()

    def disp_result(self):
        """Output result to user"""
        result_distribution = (np.array(self.n)/self.runs).tolist()
        result_distribution = [float('%.2f' % (elem*100)) for elem in result_distribution]
        q = [float('%.2f' % elem) for elem in self.q]
        bandit_machine_arms = [float('%.2f' % elem) for elem in self.bandit_machine.arms]

        print(f"Iter: {self.runs}, \u03B5 = {self.eps}, \u03B1 = {self.alpha}")
        print(f"Actions: {self.n}")
        print(f"Distribution(%): {result_distribution}")
        print(f"Quality estimate: {q}")
        print(f"Actually quality: {bandit_machine_arms}")

        if self.gen_fig:
            steps = [i + 1 for i in range(self.runs)]

            # best option percentage vs. steps
            bac_percentage = [self.record["bac"][i]*100/(i+1) for i in range(self.runs)]
            plt.plot(steps, bac_percentage)
            plt.xlabel('steps')
            plt.ylabel('optimal action (%)')
            plt.savefig('images/E2.5 optimal action vs steps.png')
            plt.show()
            plt.close()

            # best option percentage vs. steps
            plt.plot(steps, self.record["avg_r"])
            plt.xlabel('steps')
            plt.ylabel('average reward')
            plt.savefig('images/E2.5 average reward vs steps.png')
            plt.show()
            plt.close()

    def _record_data(self):
        """Record data for each action"""
        self.record["q"].append(self.q)
        self.record["n"].append(self.n)
        self.record["bac"].append(self.best_action_count)
        self.record["avg_r"].append(self.avg_r)


class MultiArmBandit:
    """
    A multi-arm bandit (arm 0 to n-1) with random parameters.
    For each arm pressed, returns an integer based on normal distribution of arm parameter.
    """
    def __init__(self, arm_num=10, params=(20, 50, 0.01)):
        self.arm_num = arm_num
        self.arms = [0] * arm_num
        # params = (lower boundary, upper boundary, standard division) of arm results
        self.lower_bound = params[0]
        self.upper_bound = params[1]
        self.walk_std = params[2]
        self._init_arm_params()

    def _init_arm_params(self):
        """Generate arm parameters"""
        for i in range(self.arm_num):
            self.arms[i] = np.random.randint(self.lower_bound, self.upper_bound)

    def bandit(self, arm_pressed):
        """Apply random walk to each arm, then return result of arm_pressed"""
        for i in range(self.arm_num):
            self.arms[i] += np.random.normal(0, self.walk_std)
        return self.arms[arm_pressed]

    def find_best_action(self):
        """Return arm with highest reward"""
        return self.arms.index(max(self.arms))


if __name__ == "__main__":
    test_machine = EpsGreedyMachine(gen_fig=True)
    test_machine.simulate(runs=10000)
    test_machine.disp_result()

