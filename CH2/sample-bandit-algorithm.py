"""
Sample bandit algorithm
Page 32: Pseudocode for a complete bandit algorithm using incrementally computed sample
averages and "eps-greedy action selection is shown in the box below. The function bandit(a)
is assumed to take an action and return a corresponding reward.
"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class EpsGreedyMachine:
    """
    When activated, for a probability of eps, select a random option.
    Otherwise select the option with largest reward.
    """

    def __init__(self, eps=0.1, arm_num=10, params=(20, 50, 5), gen_fig=False):
        self.q = [0] * arm_num
        self.n = [0] * arm_num
        self.r = []
        self.options_count = arm_num
        self.eps = eps
        self.gen_fig = gen_fig
        self.runs = 0
        self.avg_r = 0
        self.best_action_count = 0
        self.record = {"q": [], "n": [], "bac": [], "avg_r":[]}

        self.bandit_machine = MultiArmBandit(arm_num=arm_num, params=params)
        self.best_action = self.bandit_machine.find_best_action()

    def act(self):
        """Act based on possibilities, return the index of the chosen option"""
        if np.random.random() >= self.eps:
            return self.q.index(max(self.q))
        else:
            return np.random.randint(0, self.options_count)

    def step(self):
        """Perform an act and update parameters"""
        curr_arm = self.act()
        if curr_arm == self.best_action:
            self.best_action_count += 1
        self.r.append(self.bandit_machine.bandit(curr_arm))
        self.avg_r += sum(self.q)/len(self.r)
        self.n[curr_arm] += 1
        self.q[curr_arm] += (self.r[-1] - self.q[curr_arm]) / self.n[curr_arm]
        return self.act()

    def simulate(self, runs=1000000):
        """Simulate, count and return how many times each option is chosen."""
        self.runs = runs
        for i in trange(self.runs):
            self.step()
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
            plt.show()
            plt.savefig('optimal action vs steps.png')
            plt.close()

            # best option percentage vs. steps
            plt.plot(steps, self.record["avg_r"])
            plt.xlabel('steps')
            plt.ylabel('average reward')
            plt.show()
            plt.savefig('average reward vs steps.png')
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
    def __init__(self, arm_num=10, params=(20, 50, 5)):
        self.arm_num = arm_num
        self.arms = [0] * arm_num
        # params = (lower boundary, upper boundary, standard division) of arm results
        self.lower_bound = params[0]
        self.upper_bound = params[1]
        self.std_div = params[2]
        self._init_arm_params()

    def _init_arm_params(self):
        """Generate arm parameters"""
        for i in range(self.arm_num):
            self.arms[i] = np.random.randint(self.lower_bound, self.upper_bound)

    def bandit(self, arm_pressed):
        """Return result of arm_pressed"""
        arm_param = self.arms[arm_pressed]
        return np.random.randint(arm_param - 10, arm_param + 11)

    def find_best_action(self):
        """Return arm with highest reward"""
        return self.arms.index(max(self.arms))


if __name__ == "__main__":
    test_machine = EpsGreedyMachine(gen_fig=True)
    test_machine.simulate(runs=50000)
    test_machine.disp_result()
