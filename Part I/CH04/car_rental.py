"""
Example 4.2 / Exercise 4.7
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson
from tqdm import trange

matplotlib.use('Agg')

# Probability for poisson distribution
# @lam: lambda should be less than 10 for this function
poisson_cache = dict()


def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]


class CarRental:

    def __init__(self, problem_modified=False, constant_return=False):
        """Initialize parameters"""
        # Problem settings
        self.MAX_CARS = 20
        self.MAX_MOVE_OF_CARS = 5
        self.RENTAL_REQUEST_FIRST = 3
        self.RENTAL_REQUEST_SECOND = 4
        self.RETURNS_FIRST = 3
        self.RETURNS_SECOND = 2
        self.DISCOUNT = 0.9
        self.RENTAL_CREDIT = 10
        self.MOVE_CAR_COST = 2
        self.POISSON_UPPER_BOUND = 11
        self.NIGHT_PARK_LIMIT = 10
        self.NIGHT_PARK_COST = 4

        self.actions = np.arange(-self.MAX_MOVE_OF_CARS, self.MAX_MOVE_OF_CARS + 1)
        self.iter = 0
        self.value = np.zeros((self.MAX_CARS + 1, self.MAX_CARS + 1))
        self.policy = np.zeros(self.value.shape, dtype=int)
        self.modified = problem_modified
        self.constant_return = constant_return
        self.value_change_threshold = 1e-4

    def expected_return(self, state, action, state_value):
        """Calculate expected return under specific state and action"""
        if self.modified and action >= 1:
            returns = - self.MOVE_CAR_COST * (action - 1)
        else:
            returns = - self.MOVE_CAR_COST * abs(action)
        num_of_cars_first = min(state[0] - action, self.MAX_CARS)
        num_of_cars_second = min(state[1] + action, self.MAX_CARS)

        for i in range(self.POISSON_UPPER_BOUND):
            for j in range(self.POISSON_UPPER_BOUND):
                prob_rental = poisson_probability(i, self.RENTAL_REQUEST_FIRST) * \
                              poisson_probability(j, self.RENTAL_REQUEST_SECOND)
                valid_rental_first = min(num_of_cars_first, i)
                valid_rental_second = min(num_of_cars_second, j)

                reward = (valid_rental_first + valid_rental_second) * self.RENTAL_CREDIT
                num_of_cars_first_after = num_of_cars_first - valid_rental_first
                num_of_cars_second_after = num_of_cars_second - valid_rental_second

                if not self.constant_return:
                    for returned_cars_first in range(self.POISSON_UPPER_BOUND):
                        for returned_cars_second in range(self.POISSON_UPPER_BOUND):
                            prob_return = poisson_probability(returned_cars_first, self.RETURNS_FIRST) * \
                                          poisson_probability(returned_cars_second, self.RETURNS_SECOND)
                            prob_total = prob_rental * prob_return
                            num_of_cars_first_after = min(num_of_cars_first_after + returned_cars_first, self.MAX_CARS)
                            num_of_cars_second_after = min(num_of_cars_second_after + returned_cars_second, self.MAX_CARS)
                            if self.modified and num_of_cars_first_after > self.NIGHT_PARK_LIMIT:
                                reward -= self.NIGHT_PARK_COST
                            if self.modified and num_of_cars_second_after > self.NIGHT_PARK_LIMIT:
                                reward -= self.NIGHT_PARK_COST
                            returns += prob_total * (reward + self.DISCOUNT * state_value[num_of_cars_first_after,
                                                                                          num_of_cars_second_after])
                else:
                    num_of_cars_first_after = min(num_of_cars_first_after + self.RETURNS_FIRST, self.MAX_CARS)
                    num_of_cars_second_after = min(num_of_cars_second_after + self.RETURNS_SECOND, self.MAX_CARS)
                    if self.modified and num_of_cars_first_after > self.NIGHT_PARK_LIMIT:
                        reward -= self.NIGHT_PARK_COST
                    if self.modified and num_of_cars_second_after > self.NIGHT_PARK_LIMIT:
                        reward -= self.NIGHT_PARK_COST
                    returns += prob_rental * (reward + self.DISCOUNT * state_value[num_of_cars_first_after,
                                                                                   num_of_cars_second_after])

        return returns

    def update_policy(self):
        """Find optimum policy"""
        self.iter = 1
        while True:
            while True:
                old_value = self.value.copy()
                for i in trange(self.MAX_CARS + 1):
                    for j in range(self.MAX_CARS + 1):
                        self.value[i, j] = self.expected_return([i, j], self.policy[i, j], self.value)
                max_value_change = abs(old_value - self.value).max()
                print(f'iter {self.iter} max value change {max_value_change}')
                if max_value_change < self.value_change_threshold:
                    break

            policy_stable = True
            for i in trange(self.MAX_CARS + 1):
                for j in range(self.MAX_CARS + 1):
                    old_action = self.policy[i, j]
                    action_returns = []
                    for action in self.actions:
                        if -j <= action <= i:
                            action_returns.append(self.expected_return([i, j], action, self.value))
                        else:
                            action_returns.append(-np.inf)
                    self.policy[i, j] = self.actions[np.argmax(action_returns)]
                    if policy_stable and old_action != self.policy[i, j]:
                        policy_stable = False
            print('policy stable {}'.format(policy_stable))
            print("\nPOLICY:")
            print(self.policy)
            print("\nVALUE:")
            print(self.value.round())
            if policy_stable:
                break
            self.iter += 1

    def plot_policy(self):
        """plot policy"""
        _, axes = plt.subplots(1, 2, figsize=(40, 20))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        axes = axes.flatten()

        fig = sns.heatmap(np.flipud(self.policy), cmap="YlGnBu", ax=axes[0])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(self.MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('policy', fontsize=30)

        fig = sns.heatmap(np.flipud(self.value), cmap="YlGnBu", ax=axes[1])
        fig.set_ylabel('# cars at first location', fontsize=30)
        fig.set_yticks(list(reversed(range(self.MAX_CARS + 1))))
        fig.set_xlabel('# cars at second location', fontsize=30)
        fig.set_title('optimal value', fontsize=30)

        if self.modified and self.constant_return:
            plt.savefig('images/car_rental_value_constant_return_modified.png')
        elif self.modified:
            plt.savefig('images/car_rental_value_modified.png')
        elif self.constant_return:
            plt.savefig('images/car_rental_value_constant_return.png')
        else:
            plt.savefig('images/car_rental_value.png')
        plt.close()


if __name__ == "__main__":
    jack = CarRental(problem_modified=True, constant_return=False)
    jack.update_policy()
    jack.plot_policy()
