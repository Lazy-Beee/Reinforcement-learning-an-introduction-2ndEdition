"""Example 5.1: Blackjack"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import trange


class Environment:
    """The dealer of the game"""

    def __init__(self):
        """Initialize class settings"""
        self.goal = 21
        self.cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.stick_threshold = 17
        self.ace_use = 0
        self.show_card = self.hit()
        self.dealer_cards = [self.show_card, self.hit()]

    def reset_game(self):
        """Reset cards for a new game"""
        self.show_card = self.hit()
        self.dealer_cards = [self.show_card, self.hit()]
        self.ace_use = 0

    def hit(self):
        """Returns a random card from infinite deck"""
        return np.random.choice(self.cards)

    def card_value(self, cards):
        """Calculate total value of cards"""
        point = sum(cards)
        if 1 in cards and point < 12:
            point += 10
            self.ace_use = 1
        else:
            self.ace_use = 0
        return point

    def dealer_turn(self, player_point):
        """Simulate dealer's turn, return game result"""
        if player_point > 21:
            return -1
        while self.card_value(self.dealer_cards) < self.stick_threshold:
            self.dealer_cards.append(self.hit())
        dealer_point = self.card_value(self.dealer_cards)
        if dealer_point > 21 or dealer_point < player_point:
            return 1
        elif dealer_point == player_point:
            return 0
        else:
            return -1


class Agent:
    """The player of the game"""

    def __init__(self):
        """Initialize class settings"""
        self.goal = 21
        # Player sum: 12-21; Dealer show card: 1-10; Usable ace: no, yes; Action: hit, stick
        self.state_action_values = np.zeros((10, 10, 2, 2))
        self.state_action_values_unweighted = np.zeros((10, 10, 2, 2))
        self.state_action_count = np.zeros((10, 10, 2, 2), dtype=int)
        self.state_action_count_unweighted = np.zeros((10, 10, 2, 2), dtype=int)

        self.policy = np.zeros((10, 10, 2), dtype=int)

        self.state_values_ace = np.zeros((10, 10))
        self.state_values_no_ace = np.zeros((10, 10))
        self.state_count_ace = np.zeros((10, 10), dtype=int)
        self.state_count_no_ace = np.zeros((10, 10), dtype=int)

        self.player_cards = []
        self.dealer_show_card = 0
        self.discount = 1
        self.ace_use = 0
        # hit-0, stick-1
        self.actions = [0, 1]
        self.dealer = Environment()

    def start_game(self):
        """Reset game"""
        self.dealer.reset_game()
        self.ace_use = 0
        self.dealer_show_card = self.dealer.show_card
        self.player_cards = [self.dealer.hit(), self.dealer.hit()]
        # determine whether the player has a usable ace
        self.card_value(self.player_cards)

    def hit(self):
        """Get a card from dealer"""
        self.player_cards.append(self.dealer.hit())

    def card_value(self, cards):
        """Calculate total value of cards"""
        point = sum(cards)
        if 1 in cards and point < 12:
            point += 10
            self.ace_use = 1
        else:
            self.ace_use = 0
        return point

    def hit_on_less_than_20(self):
        """Apply policy of always hit before 20. Return states, actions, rewards"""
        self.start_game()
        states, actions, rewards = [], [], []

        while True:
            player_sum = self.card_value(self.player_cards)
            if player_sum < 12:
                self.hit()
            else:
                states.append([player_sum - 12, self.dealer_show_card - 1, self.ace_use])
                if player_sum < 20:
                    self.hit()
                    actions.append(0)
                    if self.card_value(self.player_cards) > 21:
                        rewards.append(-1)
                        break
                    else:
                        rewards.append(0)
                else:
                    actions.append(1)
                    rewards.append(self.dealer.dealer_turn(player_sum))
                    break
        return states, actions, rewards

    def first_visit_mc_prediction(self, episodes, record_points):
        """Apply Monte Carlo prediction on given policy. Return average returns at certain episodes"""
        state_value_record_ace = []
        state_value_record_no_ace = []
        for i in trange(episodes):
            states, _, rewards = self.hit_on_less_than_20()
            returns = 0
            for t in range(len(states) - 1, -1, -1):
                returns = self.discount * returns + rewards[t]
                player_point, dealer_show, ace = states[t]
                if ace == 0:
                    self.state_count_no_ace[player_point, dealer_show] += 1
                    self.state_values_no_ace[player_point, dealer_show] += (returns - self.state_values_no_ace[
                        player_point, dealer_show]) / self.state_count_no_ace[player_point, dealer_show]
                else:
                    self.state_count_ace[player_point, dealer_show] += 1
                    self.state_values_ace[player_point, dealer_show] += (returns - self.state_values_ace[
                        player_point, dealer_show]) / self.state_count_ace[player_point, dealer_show]

            if i + 1 in record_points:
                state_value_record_ace.append(self.state_values_ace)
                state_value_record_no_ace.append(self.state_values_no_ace)
        return state_value_record_ace, state_value_record_no_ace

    def figure_5_1(self):
        """Generate figure as figure 5.1 in the book"""
        state_value_record_ace, state_value_record_no_ace = self.first_visit_mc_prediction(500000, [10000, 500000])
        states = [state_value_record_ace[0],
                  state_value_record_no_ace[0],
                  state_value_record_ace[1],
                  state_value_record_no_ace[1]]

        titles = ['Usable Ace, 10000 Episodes',
                  'Usable Ace, 500000 Episodes',
                  'No Usable Ace, 10000 Episodes',
                  'No Usable Ace, 500000 Episodes']

        _, axes = plt.subplots(2, 2, figsize=(40, 30))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        axes = axes.flatten()

        for state, title, axis in zip(states, titles, axes):
            fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                              yticklabels=list(reversed(range(12, 22))))
            fig.set_ylabel('player sum', fontsize=30)
            fig.set_xlabel('dealer showing', fontsize=30)
            fig.set_title(title, fontsize=30)

        plt.savefig('images/figure_5_1.png')
        plt.close()

    def play_game_es(self, init_action):
        """Play the game according to current policy. Return states, actions, rewards"""
        self.start_game()
        states, actions, rewards = [], [], []
        first_step = True
        game_continue = True
        while game_continue:
            player_sum = self.card_value(self.player_cards)
            state = [player_sum - 12, self.dealer_show_card - 1, self.ace_use]
            if first_step:
                action = init_action
                if action == 0:
                    self.hit()
                    reward, game_continue = self.burst_check()
                else:
                    game_continue = False
                    reward = self.dealer.dealer_turn(player_sum)
                first_step = False
            else:
                action = self.policy[state[0], state[1], state[2]]
                if action == 0:
                    self.hit()
                    reward, game_continue = self.burst_check()
                else:
                    game_continue = False
                    reward = self.dealer.dealer_turn(player_sum)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        return states, actions, rewards

    def burst_check(self):
        """Check if player sum is over 21. Return reward and game continue"""
        if self.card_value(self.player_cards) > 21:
            return -1, False
        else:
            return 0, True

    def monte_carlo_es(self, episodes):
        """Monte Carlo with Exploring Starts"""
        self.policy[8:10, :, :] = 1
        for _ in trange(episodes):
            states, actions, rewards = self.play_game_es(np.random.choice(self.actions))
            returns = 0
            for t in range(len(states) - 1, -1, -1):
                returns = self.discount * returns + rewards[t]
                player_sum, dealer_show, ace = states[t]
                action = actions[t]
                self.state_action_count[player_sum, dealer_show, ace, action] += 1
                self.state_action_values[player_sum, dealer_show, ace, action] += \
                    (returns - self.state_action_values[player_sum, dealer_show, ace, action]) / \
                    self.state_action_count[player_sum, dealer_show, ace, action]
                self.policy[player_sum, dealer_show, ace] = \
                    np.argmax(self.state_action_values[player_sum, dealer_show, ace, :])

    def figure_5_2(self):
        """Generate figure as figure 5.2 in the book"""
        self.monte_carlo_es(1000000)

        state_value_no_usable_ace = np.max(self.state_action_values[:, :, 0, :], axis=-1)
        state_value_usable_ace = np.max(self.state_action_values[:, :, 1, :], axis=-1)
        _, action_usable_ace, action_no_usable_ace = self.find_optimum_policy()

        images = [action_usable_ace,
                  state_value_usable_ace,
                  action_no_usable_ace,
                  state_value_no_usable_ace]

        titles = ['Optimal policy with usable Ace',
                  'Optimal value with usable Ace',
                  'Optimal policy without usable Ace',
                  'Optimal value without usable Ace']

        _, axes = plt.subplots(2, 2, figsize=(40, 30))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        axes = axes.flatten()

        for image, title, axis in zip(images, titles, axes):
            fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                              yticklabels=list(reversed(range(12, 22))))
            fig.set_ylabel('player sum', fontsize=30)
            fig.set_xlabel('dealer showing', fontsize=30)
            fig.set_title(title, fontsize=30)

        plt.savefig('images/figure_5_2.png')
        plt.close()

    def play_game_b(self, b):
        """Play the game with policy b. Return states, actions, rewards"""
        self.start_game()
        states, actions, rewards = [], [], []
        game_continue = True
        while game_continue:
            player_sum = self.card_value(self.player_cards)
            state = [player_sum - 12, self.dealer_show_card - 1, self.ace_use]
            action = np.random.choice([0, 1], p=b)
            if action == 0:
                self.hit()
                reward, game_continue = self.burst_check()
            else:
                game_continue = False
                reward = self.dealer.dealer_turn(player_sum)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        return states, actions, rewards

    def off_policy(self, episodes, policy_control=True):
        """Off-policy MC prediction/control with behavior policy of [0.5, 0.5]"""
        b_policy = [0.5, 0.5]
        for _ in trange(episodes):
            states, actions, rewards = self.play_game_b(b_policy)
            returns = 0
            weight = 1
            for t in range(len(states) - 1, -1, -1):
                returns = self.discount * returns + rewards[t]
                player_sum, dealer_show, ace = states[t]
                action = actions[t]
                # Unweighted rewards
                self.state_action_count_unweighted[player_sum, dealer_show, ace, action] += 1
                self.state_action_values_unweighted[player_sum, dealer_show, ace, action] += \
                    (returns - self.state_action_values_unweighted[player_sum, dealer_show, ace, action]) / \
                    self.state_action_count_unweighted[player_sum, dealer_show, ace, action]
                # weighted rewards
                self.state_action_count[player_sum, dealer_show, ace, action] += weight
                self.state_action_values[player_sum, dealer_show, ace, action] += \
                    (returns - self.state_action_values[player_sum, dealer_show, ace, action]) * weight / \
                    self.state_action_count[player_sum, dealer_show, ace, action]

                if policy_control:
                    self.policy[player_sum, dealer_show, ace] = \
                        np.argmax(self.state_action_values[player_sum, dealer_show, ace, :])
                    if action != self.policy[player_sum, dealer_show, ace]:
                        break
                    weight /= b_policy[action]
                else:
                    weight *= self.policy[player_sum, dealer_show, ace] / b_policy[action]
                    if weight == 0:
                        break

    def find_optimum_policy(self):
        """Find optimum policy based on state-action values"""
        action = np.argmax(self.state_action_values[:, :, :, :], axis=-1)
        action_no_usable_ace = np.argmax(self.state_action_values[:, :, 0, :], axis=-1)
        action_usable_ace = np.argmax(self.state_action_values[:, :, 1, :], axis=-1)

        return action, action_usable_ace, action_no_usable_ace

    def figure_5_3(self):
        """Generate figure 5.3"""
        self.off_policy(1000000)

        state_value_no_usable_ace = np.max(self.state_action_values[:, :, 0, :], axis=-1)
        state_value_usable_ace = np.max(self.state_action_values[:, :, 1, :], axis=-1)
        _, action_usable_ace, action_no_usable_ace = self.find_optimum_policy()

        images = [action_usable_ace,
                  state_value_usable_ace,
                  action_no_usable_ace,
                  state_value_no_usable_ace]

        titles = ['Optimal policy with usable Ace',
                  'Optimal value with usable Ace',
                  'Optimal policy without usable Ace',
                  'Optimal value without usable Ace']

        _, axes = plt.subplots(2, 2, figsize=(40, 30))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        axes = axes.flatten()

        for image, title, axis in zip(images, titles, axes):
            fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                              yticklabels=list(reversed(range(12, 22))))
            fig.set_ylabel('player sum', fontsize=30)
            fig.set_xlabel('dealer showing', fontsize=30)
            fig.set_title(title, fontsize=30)

        plt.savefig('images/figure_5_3.png')
        plt.close()


if __name__ == "__main__":
    player = Agent()
    # player.figure_5_1()
    # player.figure_5_2()
    player.figure_5_3()

