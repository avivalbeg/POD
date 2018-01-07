import os

import itertools

import sys
from itertools import product


from constants import *
from util import *
from train.data_parsing import DeepMindPokerDataLoader

runBenchmark = False
runAnn = True


def collectCardDistributions():
    """
    Collect the distribution of cards from the poker bot's
    database.
    """
    cardCounts = defaultdict(lambda: defaultdict(lambda: 0))

    gl = GameLogger()
    i = 0
    for game in gl.mongodb.games.find():
        i += 1
        if game["rounds"]:
            firstRoundVals = game["rounds"][0]["round_values"]
            endRoundVals = game["rounds"][-1]["round_values"]
            myCards = firstRoundVals['PlayerCardList'].split(" ")

            cardsOnTable = endRoundVals['cardsOnTable'].split(" ")
            cards = endRoundVals['PlayerCardList_and_others'].split(" ")

            for card in findCards(cardsOnTable):
                cardCounts[firstRoundVals["pokerSite"]][card] += 1
            if i % 500 == 0:
                pprint(cardCounts)
                print()
    pprint(cardCounts)


def countGamesWithAllCardsShown():
    gameSets = set()
    total = 0.
    withCards = 0.
    parser = IrcHoldemDataParser()
    for game in parser.iterGames():
        total += 1
        if all([player.cards for player in game.players]):

            allActions = "".join(
                [player.preflopActions + player.flopActions + player.turnActions + player.riverActions for player in
                 game.players]
            )
            if NA in allActions and not ALL_IN in allActions:
                withCards += 1
                print(game)
                print(game.gameId, game.timeStamp, game.gameSetId, withCards)
                gameSets.add(game.gameSetId)

                if total % 500 == 0:
                    print(gameSets)




def train(data):
    """Run various models on a dataset, using the models package."""
    from models.DataLoader import TextVectorDataLoader, SKLearnDataLoader, \
        RandomDataLoader
    from models.eval_tools import Config, compareModels
    import itertools
    from models.params import REG_VALS, LRS, BATCH_SIZE, N_EPOCHS
    from models.Model import SVMModel, LogisticRegressionModel, KMeansModel, \
        KNeighborsModel, SoftmaxANN, DoubleLayerSoftmaxANN, TripleLayerSoftmaxANN
    from sklearn import datasets



    print("Training with %d examples" % len(data))
    if runBenchmark:
        configs = [Config(data, reg, BATCH_SIZE, N_EPOCHS, 0) \
                   for reg in REG_VALS]
        compareModels((
            SVMModel,
            LogisticRegressionModel,
            KMeansModel,
            KNeighborsModel,
        ),
            configs,
            debug=False)

    if runAnn:
        configs = [Config(data, reg, BATCH_SIZE, N_EPOCHS, lr) \
                   for reg, lr in product(REG_VALS, LRS)]

        compareModels((
            #             SoftmaxANN,
            #             DoubleLayerSoftmaxANN,
            TripleLayerSoftmaxANN,
        ),
            configs,
            debug=True)



def trainLstm(data):
    """Train an LSTM on a dataset.
    This is different from the train function above because an LSTM is
    time sensitive, which means that the data format is 2D matrices
    rather than 1D vectors. Each 2D matrix is a stack of vectors that are
    ordered in time. In our case, each vector is a game round."""

    from models.Model import LstmClassifier
    from models.eval_tools import AnnConfig

    nDims = len(DEEPMIND_GAME_FEATURES) + len(FEATURE_ACTIONS) - 1
    nEpochs = 35

    regs = sorted(np.logspace(-4, 2, num=4, base=10))
    batchSizes = [10, 3, 32]
    hidSizes = [1000, 400]
    layerCounts = [3, 2, 4]
    dropouts = [.5, .2, .7]
    activations = ['sigmoid', 'tanh', 'relu']

    configs = [AnnConfig(data, reg, batchSize, nEpochs, 0,
                         hidSize, nLayers, dropout, activation)
               for reg, batchSize, hidSize, nLayers, dropout, activation in
               itertools.product(regs, batchSizes, hidSizes, layerCounts,
                                 dropouts, activations)]

    maxAcc, bestConfig = 0, None

    for config in configs:
        model = LstmClassifier(config)
        print("Trying config:")
        print([config])
        print()
        acc, nEpochs = model.trainGraded(verbose=0)
        print()
        print("Achieved accuracy of %s within %s epochs" % (str(acc),
                                                             nEpochs))
        if acc >= maxAcc:
            config.nEpochs = nEpochs
            maxAcc, bestConfig = acc, config

    print("Best config with accuracy %d:" % maxAcc)
    print(config)


def main(args):
    #     Watchuwanna do?
    #     countGamesWithAllCardsShown()
    #     collectCardDistributions()
    #     getIrcData()
    #     dlPokerBotData()
    #     mineGameData(debugMode=False)
    # buildIrcVectors(debug=True)
    #     mineHandData()
    #     train1(IrcDataLoader())
    trainLstm(DeepMindPokerDataLoader(2000))

    pass


if __name__ == '__main__':
    main(sys.argv)
