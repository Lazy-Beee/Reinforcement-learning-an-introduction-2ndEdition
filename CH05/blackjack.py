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
        # 0-9: points 12-21; 0-9: cards 0-10; 0-1: no ace, ace; 0-1: hit, stick
        self.state_action_values = np.zeros((10, 10, 2, 2))
        self.state_action_count = np.zeros((10, 10, 2, 2))

        self.state_values_ace = np.zeros((10, 10))
        self.state_values_no_ace = np.zeros((10, 10))
        self.state_count_ace = np.zeros((10, 10))
        self.state_count_no_ace = np.zeros((10, 10))

        self.policy = np.zeros((10, 10, 2))
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
                    self.state_values_no_ace[player_point, dealer_show] += returns
                else:
                    self.state_count_ace[player_point, dealer_show] += 1
                    self.state_values_ace[player_point, dealer_show] += returns

            if i + 1 in record_points:
                state_value_record_ace.append(self.state_values_ace/self.state_count_ace)
                state_value_record_no_ace.append(self.state_values_no_ace/self.state_count_no_ace)
        return state_value_record_ace, state_value_record_no_ace

    def figure_5_1(self):
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


if __name__ == "__main__":
    player = Agent()
    player.figure_5_1()

    # print(player.hit_on_less_than_20())
    # print(player.player_cards, player.dealer.dealer_cards)
