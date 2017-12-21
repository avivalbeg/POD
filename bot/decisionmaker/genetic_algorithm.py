'''
Assesses the log file and checks how the parameters in strategies.xml need to be adjusted to optimize playing
'''

import logging
from configobj import ConfigObj
from bot.tools.mongo_manager import GameLogger,StrategyHandler

class GeneticAlgorithm(object):
    """
    When this class is instantiated with a StrategyHandler object,
    it attempts to improve it.
    This is done in TableScreenBased.call_genetic_algorithm, which 
    is called every time 
    """
    
    def __init__(self, write_update, gameLogger):
        self.logger = logging.getLogger('genetic_algo')
        self.logger.setLevel(logging.DEBUG)
        self.output = ''
        p = StrategyHandler()
        p.read_strategy()
        strategyName = p.current_strategy
        self.logger.debug("Strategy to analyse: "+strategyName)
        self.load_log(strategyName, gameLogger)
        # Genetic algo applies here
        self.improve_strategy(gameLogger, p)
        
        # If the strategy was modified, and write_updates is true,
        # then save it to server 
        # (when this method is called from TableScreenBased, write_update is set to True)
        if (self.modified and write_update==True) or write_update=="Force":
            # Save strategy
            p.save_strategy_genetic_algorithm()
            
            # Update reference in config object
            config = ConfigObj("config.ini")
            config['last_strategy'] = p.current_strategy
            config.write()
            self.logger.info("Genetic algorithm: New strategy saved")

    def get_results(self):
        return self.output

    def load_log(self, strategyName, gameLogger):
        """
        Update GameLog gameLogger with data from all
        games in which a strategy with name strategyName was used.
        Data is retrieved from the mongodb.
        """
        self.gameResults = {}
        gameLogger.get_stacked_bar_data('Template', strategyName, 'stackedBar')
        self.recommendation = dict()

    def assess_call(self, p, gameLogger, decision, stage, coeff1, coeff2, coeff3, coeff4, change):
        A = gameLogger.d[decision, stage, 'Won'] > gameLogger.d[decision, stage, 'Lost'] * coeff1  # Call won > call lost * c1
        B = gameLogger.d[decision, stage, 'Lost'] > gameLogger.d['Fold', stage, 'Lost'] * coeff2  # Call Lost > Fold lost
        C = gameLogger.d[decision, stage, 'Won'] + gameLogger.d['Bet', stage, 'Won'] < gameLogger.d[
                                                                         'Fold', stage, 'Lost'] * coeff3  # Fold Lost*c3 > Call won + bet won
        if A and B:
            self.recommendation[stage, decision] = "ok"
        elif A and B == False and C:
            self.recommendation[stage, decision] = "more agressive"
            p.modify_strategy(stage + 'MinCallEquity', -change)
            p.modify_strategy(stage + 'CallPower', -change * 25)
            self.changed += 1
        elif A == False and B == True:
            self.recommendation[stage, decision] = "less agressive"
            p.modify_strategy(stage + 'MinCallEquity', +change)
            p.modify_strategy(stage + 'CallPower', +change * 25)
            self.changed += 1
        else:
            self.recommendation[stage, decision] = "inconclusive"
        self.logger.info(stage + " " + decision + ": " + self.recommendation[stage, decision])
        self.output += stage + " " + decision + ": " + self.recommendation[stage, decision] + '\n'

    def assess_bet(self, p, gameLogger, decision, stage, coeff1, change):
        A = gameLogger.d['Bet', stage, 'Won'] > (gameLogger.d['Bet', stage, 'Lost']) * coeff1  # Bet won bigger Bet lost
        B = gameLogger.d['Check', stage, 'Won'] > gameLogger.d['Check', stage, 'Lost']  # check won bigger check lost
        C = gameLogger.d['Bet', stage, 'Won'] < (gameLogger.d['Bet', stage, 'Lost']) * 1  # Bet won bigger Bet lost

        if A and not B:
            self.recommendation[stage, decision] = "ok"
        elif A and B:
            self.recommendation[stage, decision] = "more agressive"
            p.modify_strategy(stage + 'MinBetEquity', -change)
            p.modify_strategy(stage + 'BetPower', -change * 25)
            self.changed += 1
        elif C and not B:
            self.recommendation[stage, decision] = "less agressive"
            p.modify_strategy(stage + 'MinBetEquity', +change)
            p.modify_strategy(stage + 'BetPower', +change * 25)
            self.changed += 1
        else:
            self.recommendation[stage, decision] = "inconclusive"
        self.logger.info(stage + " " + decision + ": " + self.recommendation[stage, decision])
        self.output += stage + " " + decision + ": " + self.recommendation[stage, decision] + '\n'

    def improve_strategy(self, gameLogger, p):
        """
        Run the genetic algo on given strategy p (type StrategyHandler)
        and based on info from logger gameLogger (type GameLogger).
        """
        self.modified=False
        self.changed = 0
        maxChanges = 2
        if self.changed <= maxChanges:
            coeff1 = 2
            coeff2 = 1
            coeff3 = 2
            coeff4 = 1
            stage = 'River'
            decision = 'Call'
            change = 0.02
            self.assess_call(p, gameLogger, decision, stage, coeff1, coeff2, coeff3, coeff4, change)

        if self.changed < maxChanges:
            coeff1 = 2
            coeff2 = 1.5
            coeff3 = 2
            stage = 'Turn'
            decision = 'Call'
            change = 0.02
            self.assess_call(p, gameLogger, decision, stage, coeff1, coeff2, coeff3, coeff4, change)

        if self.changed < maxChanges:
            coeff1 = 2
            coeff2 = 1.5
            coeff3 = 2
            stage = 'Flop'
            decision = 'Call'
            change = 0.01
            self.assess_call(p, gameLogger, decision, stage, coeff1, coeff2, coeff3, coeff4, change)

        if self.changed < maxChanges:
            coeff1 = 2
            coeff2 = 2.5
            coeff3 = 2
            stage = 'PreFlop'
            decision = 'Call'
            change = 0.03
            self.assess_call(p, gameLogger, decision, stage, coeff1, coeff2, coeff3, coeff4, change)

        if self.changed>0: self.modified=True
        self.changed = 0

        if self.changed < maxChanges:
            coeff1 = 2
            stage = 'River'
            decision = 'Bet'
            change = 0.02
            self.assess_bet(p, gameLogger, decision, stage, coeff1, change)

        if self.changed < maxChanges:
            coeff1 = 2
            stage = 'Turn'
            decision = 'Bet'
            change = 0.02
            self.assess_bet(p, gameLogger, decision, stage, coeff1, change)

        if self.changed < maxChanges:
            coeff1 = 2
            stage = 'Flop'
            decision = 'Bet'
            change = 0.02
            self.assess_bet(p, gameLogger, decision, stage, coeff1, change)

        if self.changed < maxChanges:
            coeff1 = 2
            stage = 'PreFlop'
            decision = 'Bet'
            change = 0.02
            self.assess_bet(p, gameLogger, decision, stage, coeff1, change)

        if self.changed > 0: self.modified = True


def run_genetic_algorithm(write, logger):
    logger.info("===Running genetic algorithm===")
    gameLogger = GameLogger()
    GeneticAlgorithm(write, gameLogger)


if __name__ == '__main__':
    # Run the genetic algorithm on 
    import logging

    logger = logging
    logger.basicConfig(level=logging.DEBUG)
    run_genetic_algorithm(False, logger)

    user_input = input("Run again and modify (Y)=Force / N? ")
    if user_input.upper() == "Y":
        run_genetic_algorithm("Force", logger)
