"""
Example 4.2: Jack’s Car Rental Jack manages two locations for a nationwide car
rental company. Each day, some number of customers arrive at each location to rent cars.
If Jack has a car available, he rents it out and is credited $10 by the national company.
If he is out of cars at that location, then the business is lost. Cars become available for
renting the day after they are returned. To help ensure that cars are available where
they are needed, Jack can move them between the two locations overnight, at a cost of
$2 per car moved. We assume that the number of cars requested and returned at each
location are Poisson random variables, meaning that the probability that the number is
n is "nn! e#", where " is the expected number. Suppose " is 3 and 4 for rental requests at
the first and second locations and 3 and 2 for returns. To simplify the problem slightly,
we assume that there can be no more than 20 cars at each location (any additional cars
are returned to the nationwide company, and thus disappear from the problem) and a
maximum of five cars can be moved from one location to the other in one night. We take
the discount rate to be # = 0.9 and formulate this as a continuing finite MDP, where
the time steps are days, the state is the number of cars at each location at the end of
the day, and the actions are the net numbers of cars moved between the two locations
overnight. Figure 4.2 shows the sequence of policies found by policy iteration starting
from the policy that never moves any cars.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

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

    def __init__(self):
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

        # Possible car movement from location 1 to 2
        self.actions = np.arange(-self.MAX_MOVE_OF_CARS, self.MAX_MOVE_OF_CARS + 1)

    def expected_return(self, state, action, state_value):
        """Calculate expected return under state and action"""
        returns = - self.MOVE_CAR_COST * abs(action)
        num_of_cars_first = min(state[0] - action, self.MAX_CARS)
        num_of_cars_second = min(state[1] + action, self.MAX_CARS)

        for rental_request_first in range(self.POISSON_UPPER_BOUND):
            for rental_request_second in range(self.POISSON_UPPER_BOUND):
                prob_rental = poisson_probability(rental_request_first,
                                                  self.RENTAL_REQUEST_FIRST) * \
                              poisson_probability(rental_request_second,
                                                  self.RENTAL_REQUEST_SECOND)
                valid_rental_first = min(num_of_cars_first, rental_request_first)
                valid_rental_second = min(num_of_cars_second, rental_request_second)

                reward = (valid_rental_first + valid_rental_second) * self.RENTAL_CREDIT
                num_of_cars_first -= valid_rental_first
                num_of_cars_second -= valid_rental_second

                for returned_cars_first in range(self.POISSON_UPPER_BOUND):
                    for returned_cars_second in range(self.POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(returned_cars_first,
                                                          self.RETURNS_FIRST) * \
                                      poisson_probability(returned_cars_second,
                                                          self.RETURNS_SECOND)
                        prob_total = prob_rental * prob_return
                        num_of_cars_first = min(num_of_cars_first + returned_cars_first,
                                                self.MAX_CARS)
                        num_of_cars_second = min(num_of_cars_second + returned_cars_second,
                                                 self.MAX_CARS)

                        returns += prob_total * (reward + self.DISCOUNT *
                                                 state_value[num_of_cars_first, num_of_cars_second])

        return returns