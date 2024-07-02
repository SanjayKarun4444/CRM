import numpy as np
from numpy.random import choice


class RPSTrainer:
    def __init__(self):

        self.NUM_ACTIONS = 3
        self.possible_actions = np.arange(self.NUM_ACTIONS)
        self.actionUtility = np.array([
            [0,-1,1],
            [1,0,-1],
            [-1,1,0]
        ])
        self.regretSum = np.zeros(self.NUM_ACTIONS)
        self.strategySum = np.zeros(self.NUM_ACTIONS)

        self.oppregretSum = np.zeros(self.NUM_ACTIONS)
        self.oppstrategySum = np.zeros(self.NUM_ACTIONS)

    def getStrategy(self, regret_sum):
        regret_sum[regret_sum<0] = 0
        normalizing_sum = sum(regret_sum)
        strategy = regret_sum
        for a in range(self.NUM_ACTIONS):
            if normalizing_sum > 0:
                strategy[a] /= normalizing_sum
            else:
                strategy[a] = 1.0/self.NUM_ACTIONS

        return strategy

    def getAverageStrategy(self, strategySum):
        average_strategy = [0,0,0]
        normalizing_sum = sum(strategySum)
        for a in range(self.NUM_ACTIONS):
            if normalizing_sum > 0:
                average_strategy[a] = strategySum[a] / normalizing_sum
            else:
                average_strategy[a] = 1.0/self.NUM_ACTIONS

        return average_strategy

    def get_action(self, strategy):
        return choice(self.possible_actions, p=strategy)

    def get_reward(self, myAction, opponentAction):
        return self.actionUtility[myAction, opponentAction]

    def train(self, iterations):

        for i in range(iterations):
            strategy = self.getStrategy(self.regretSum)
            oppStrategy = self.getStrategy(self.oppregretSum)

            self.strategySum += strategy
            self.oppstrategySum += oppStrategy

            opponent_action = self.get_action(oppStrategy)
            my_action = self.get_action(strategy)

            my_reward = self.get_reward(my_action, opponent_action)
            opp_reward = self.get_reward(opponent_action, my_action)

            for a in range(self.NUM_ACTIONS):
                my_regret = self.get_reward(a,opponent_action) - my_reward
                opp_regret = self.get_reward(a, my_action) - opp_reward
                self.regretSum[a] += my_regret
                self.oppregretSum[a] += opp_regret


def main():
    trainer = RPSTrainer()
    trainer.train(1000)
    target_policy = trainer.getAverageStrategy(trainer.strategySum)
    opp_target_policy = trainer.getAverageStrategy(trainer.oppstrategySum)

    print('Target Policy: %s' % (target_policy))

if __name__ == "__main__":
    main()