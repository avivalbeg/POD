import os

import itertools

import sys
from itertools import product

from os.path import join
from train.get_data import *
from train.data_parsing import *
from constants import *
from util import *


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


def train(data, runBenchmark=True, runAnn=True):
    """Run various models on a dataset and compare their
    performance, using the models package."""
    from models.eval_tools import Config, compareModels
    from models.params import REG_VALS, LRS, BATCH_SIZE, N_EPOCHS
    from models.Model import SVMModel, LogisticRegressionModel, KMeansModel, \
        KNeighborsModel, SoftmaxANN, DoubleLayerSoftmaxANN, TripleLayerSoftmaxANN


    print("Training with %d examples" % len(data))
    if runBenchmark:
        configs = [Config(data, reg, BATCH_SIZE, N_EPOCHS, 0) \
                   for reg in [.001]]
        compareModels((
            LogisticRegressionModel,
            KMeansModel,
            SVMModel,
            KNeighborsModel,
        ),
            configs,
            debug=True)

    if runAnn:
        configs = [Config(data, reg, BATCH_SIZE, N_EPOCHS, lr) \
                   for reg, lr in product(REG_VALS, LRS)]

        compareModels((
            SoftmaxANN,
            DoubleLayerSoftmaxANN,
            # TripleLayerSoftmaxANN,
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
    nEpochs = 10

    regs = sorted(np.logspace(-4, 2, num=3, base=10))
    batchSizes = [64, 32, 16]
    hidSizes = [100]# [1000, 400]
    layerCounts = [1] #3, 4, 2]
    dropouts = [.5, .2, .7]
    lrs = [.005, .01, .001]
    activations = ['sigmoid', 'tanh', 'relu']

    configs = [AnnConfig(data, reg, batchSize, nEpochs, lr,
                         hidSize, nLayers, dropout, activation)
               for reg, batchSize, hidSize, nLayers, dropout, activation, lr in
               itertools.product(regs, batchSizes, hidSizes, layerCounts,
                                 dropouts, activations, lrs)]

    maxAcc, bestConfig, bestNEpochs = 0, None, 0
    for config in configs:
        model = LstmClassifier(config)
        print("A priori accuracy:", model.eval(config.data.devX, config.data.dev_y))
        print("Training with config:")
        print([config])
        print()
        acc, nEpochs = model.trainGraded(step=1, verbose=1)
        print()
        print("Achieved accuracy of %s within %s epochs" % (str(acc),
                                                            nEpochs))
        if acc >= maxAcc:
            maxAcc, bestConfig, bestNEpochs = acc, config, nEpochs

    bestConfig.nEpochs = bestNEpochs
    bestModel = LstmClassifier(bestConfig)
    testAcc = bestModel.eval(bestConfig.data.testX, bestConfig.data.test_y)
    print("Best config (dev accuracy: %s, test accuracy: %s):" % (str(maxAcc), str(testAcc)))
    print(bestConfig)


def main(args):
    np.set_printoptions(suppress=True,edgeitems=1000,linewidth=140)
    #     Watchuwanna do?
    #     countGamesWithAllCardsShown()
    #     collectCardDistributions()
    #     getIrcData()
    #     dlPokerBotData()
    #     mineGameData(debugMode=False)
    #        mineHandData()
    #     train1(IrcDataLoader())
    # trainLstm(DeepMindPokerDataLoader(1000))
    # buildIrcHandVectors(debug=False, cap=2000)
    # train(IrcHandVecsDataLoader())
    trainLstm(IrcHandMatsDataLoader(0, float('inf')))
    # for stage in range(0,4):
    #     for cutoff in RANK_CUTOFFS:
    #         print("Training for stage:", stage)
    #         trainLstm(IrcHandDataLoader(stage, cutoff))
    pass


if __name__ == '__main__':
    main(sys.argv)
