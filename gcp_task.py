import numpy as np
import keras
import pymongo
import pandas
import itertools, re
from pymongo import MongoClient

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from constants import *


def oneHotEncCat(cat, cats):
    """One-hot encode a category from a list of possible categories."""
    ind = cats.index(cat)
    vec = np.zeros(len(cats))
    vec[ind] = 1
    return vec


def isNumStr(x):
    return type(x) == type("") and re.match("-?\d*((\.\d*)|\d+)", x)


def pad(matrix, n, nDims):
    """Pad a 2D matrix with NaN vectors up to certain size."""
    for _ in range(max([0, n - len(matrix)])):
        emptyRound = np.empty(nDims)
        emptyRound[:] = NAN_NUM  # Fill it with nothing
        matrix = np.vstack((matrix, emptyRound))
    return matrix


def featurize(value, strDic={}):
    """Turn some value into a numerical feature."""

    if isNumStr(value):
        return eval(value)

    if type(value) == type(""):
        if value in ("nan", "NaN", "None"):
            return NAN_NUM
        elif value in ("True", "False"):
            return int(eval(value))
        elif not value:
            return 0
        else:
            return strDic[value]

    if type(value) in (type(np.nan), type(None)):
        return NAN_NUM

    if type(value) == type(False):
        return int(value)

    return value


def getOrDefault(obj, item, dflt):
    try:
        ret = obj[item]
        if type(ret) == type(u""):
            ret = ret.encode()
        return ret
    except (KeyError, IndexError):
        return dflt


def makeRoundVector(roundValues):
    """Creates a vector representation of one Deep Mind Poker Bot
    round."""
    # Reduce number of decisions
    action = roundValues["decision"]
    if "bet" in action.lower() or "all" in action.lower(): action = "Raise"
    if "check" in action.lower() or "call" in action.lower(): action = "Call"
    if "fold" in action.lower(): action = "Raise"  # Dummy; folded rounds won't be considered
    # One hot encode decision
    decisionVector = oneHotEncCat(action, FEATURE_ACTIONS)

    # Insert all values to vector
    vector = np.zeros(len(DEEPMIND_GAME_FEATURES) - 1)
    for i, feat in enumerate(DEEPMIND_GAME_FEATURES):
        if i < len(vector):
            vector[i] = featurize(getOrDefault(roundValues, feat, np.nan),
                                  strDic=
                                  {x.encode(): j for j, x in
                                   list(enumerate(DEEPMIND_ACTIONS)) + list(enumerate(GameStages))})
    return np.hstack((vector, decisionVector))


class DummyData:
    pass


class DummyConfig:
    def __init__(self, data, reg, batchSize, nEpochs, lr,
                 hidSize, nLayers, dropout, activation):
        self.data, self.reg, self.batchSize, self.nEpochs, self.lr, self.hidSize, self.nLayers, self.dropout, self.activation = data, reg, batchSize, nEpochs, lr, hidSize, nLayers, dropout, activation
        self.inputShape = self.data.trainX.shape[1:]

    strBlacklist = ["data"]

    def __str__(self):
        return "\n".join([
            k + " : " + str(v) for k, v in vars(self).items() if k not in self.strBlacklist
        ])

    def __repr__(self):
        return "-".join([
            k + ":" + str(v) for k, v in vars(self).items() if k not in self.strBlacklist
        ])


def main():
    mongoclient = MongoClient('mongodb://guest:donald@dickreuter.com:27017/POKER')
    mongodb = mongoclient.POKER  # This is the database of games
    games = mongodb.games.find()

    classes = ["Lost", "Won"]
    prevRoundId = (np.nan, np.nan, np.nan)

    gameMats = []
    i = 0
    for game in games:
        print(i)
        if i > 7000:
            break
        i += 1
        outcome = game["FinalOutcome"]

        if outcome in ("Neutral",):
            continue

        rounds = sorted(game["rounds"],
                        key=lambda x: (GameStages.index(x["round_values"]["gameStage"]), x["round_number"]))

        if not rounds:
            continue

        gameVec = []
        for j, roundData in enumerate(rounds):
            roundVec = makeRoundVector(roundData["round_values"])
            roundVec = np.hstack((roundVec, np.array([classes.index(outcome)])))
            gameVec.append(roundVec)
            if roundData["round_values"]["decision"] != "Fold":
                i += 1
                gameMats.append(pad(np.array(gameVec), MAX_N_ROUNDS, len(roundVec)))
    gameMats = np.array(gameMats)
    np.random.shuffle(gameMats)

    train, test = np.array_split(gameMats, 2)

    data = DummyData()
    data.trainX, data.train_y = np.split(train, (-1,), 2)
    data.testX, data.test_y = np.split(test, (-1,), 2)
    data.train_y = np.max(np.squeeze(data.train_y, 2), 1)
    data.test_y = np.max(np.squeeze(data.test_y, 2), 1)

    nEpochs = 40
    step = 5
    regs = sorted(np.logspace(-4, 2, num=4, base=10))
    batchSizes = [32, 16]
    hidSizes = [1000, 400]
    layerCounts = [3, 2, 4]
    dropouts = [.5, .2, .7]
    activations = ['sigmoid', 'tanh', 'relu']

    configs = [DummyConfig(data, reg, batchSize, nEpochs, 0,
                           hidSize, nLayers, dropout, activation)
               for reg, batchSize, hidSize, nLayers, dropout, activation in
               itertools.product(regs, batchSizes, hidSizes, layerCounts,
                                 dropouts, activations)]
    maxAcc, bestConfig, bestNEpochs = 0, None, 0

    for config in configs:

        print("Trying config:")
        print([config])
        print()

        # Build model:
        model = Sequential()

        # Input layer
        model.add(Dense(config.hidSize, input_shape=config.inputShape,
                        kernel_regularizer=keras.regularizers.l2(config.reg),
                        activity_regularizer=keras.regularizers.l1(config.reg)
                        ))

        lstmLayer = lambda retSeqs: keras.layers.LSTM(config.hidSize,
                                                      kernel_regularizer=keras.regularizers.l2(config.reg),
                                                      recurrent_regularizer=keras.regularizers.l2(config.reg),
                                                      bias_regularizer=keras.regularizers.l1(config.reg),
                                                      activity_regularizer=keras.regularizers.l1(config.reg),
                                                      dropout=config.dropout,
                                                      recurrent_dropout=0.,  # vary?
                                                      return_sequences=retSeqs)
        # Add lstm layers
        for _ in range(config.nLayers - 1):
            model.add(lstmLayer(True))
        model.add(lstmLayer(False))

        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        # Train
        for i in range(1, int(nEpochs / step) + 1):
            model.fit(data.trainX, data.train_y, epochs=step, batch_size=config.batchSize)
            # Final evaluation of the model
            _, acc = model.evaluate(data.testX, data.test_y, verbose=1)
            if acc > maxAcc:
                maxAcc, bestConfig, bestNEpochs = acc, config, i * step
            print("Currently best config (accuracy %s, %d epochs):" % (str(maxAcc), bestNEpהochs))
            print([bestConfig])


if __name__ == '__main__':
    main()
