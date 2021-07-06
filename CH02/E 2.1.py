"""
6/26/2021 Jun Fan
Exercise 2.1 In "eps-greedy action selection, for the case of two actions and " = 0.5, what is
the probability that the greedy action is selected?
"""
import numpy as np
from tqdm import trange


class EpsGreedyMachine:
    """
    When activated, for a possibility of [eps], select a random option.
    Otherwise select the largest option in options.
    """

    def __init__(self, options=(0, 1), eps=0.5):
        self.options = options
        self.options_count = len(options)
        self.eps = eps
        self.runs = 0

    def act(self):
        """Act based on possibilities, return the index of the chosen option"""
        if np.random.random() >= self.eps:
            return self.options.index(max(self.options))
        else:
            return np.random.randint(0, self.options_count)

    def simulate(self, runs=1000):
        """Simulate, count and return how many times each option is chosen."""
        self.runs = runs
        result = [0] * self.options_count
        for i in trange(runs):
            result[self.act()] += 1
        option_possibility = (np.array(result)/runs).tolist()
        return result, option_possibility


if __name__ == "__main__":
    test_machine = EpsGreedyMachine()
    _, [_, greedy_possibility] = test_machine.simulate(runs=1000000)
    print(f"\nGreedy action selected out of {test_machine.runs} trails (\u03B5 = 0.5):"
          f"\t{greedy_possibility*100}%")