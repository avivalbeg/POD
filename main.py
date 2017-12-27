

import os

from constants import *
from util import *
from itertools import product
import sys
from mining.get_data import dlPokerBotData, mineGameData
from bot.tools.mongo_manager import GameLogger
from pprint import pprint
import re
from _collections import defaultdict
from mining.data_parsing import IrcHoldemDataParser, IrcDataParser
import pandas as pd
from copy import copy
from sklearn.cross_validation import train_test_split
from models.Model import Model, LstmClassifier
import tensorflow as tf
from models.utils import sample_batch

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

runBenchmark = False
runAnn = True



def collectCardDistributions():
    """
    Collect the distribution of cards from the poker bot's
    database.
    """
    cardCounts = defaultdict(lambda:defaultdict(lambda:0))

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
                [player.preflopActions + player.flopActions + player.turnActions + player.riverActions for player in game.players]
                )
            if NA in allActions and not ALL_IN in allActions:
                withCards += 1
                print(game)
                print(game.gameId, game.timeStamp, game.gameSetId, withCards)
                gameSets.add(game.gameSetId)
                
                if total % 500 == 0:
                    print(gameSets)


def trainEquityPrediction():
    """
    Runs the machine learning models from the ml directory.
    Notice that there is also a train.py script in the bot directory, but that script is different because it trains the bot to make actual decisions. This script on the other hand just creates statistical models.
    
    
    Best results are achieved with a logistic regression model:
    
    'LogisticRegressionModel': (79.014084507042256, # Test accuracy
                                 reg:0.1-nSamples:2272-nFeats:22-nClasses:5-batchSize:64-nEpochs:10000-lr:0),
    
    Best results I got with a neural network:
    
    {'DoubleLayerSoftmaxANN': (0.6915493, # Test accuracy
                               reg:3.16227766017-nSamples:2272-nFeats:22-nClasses:5-batchSize:64-nEpochs:100000-lr:0.00501187233627)}
    
    """ 
    
    from models.DataLoader import TextVectorDataLoader, SKLearnDataLoader, \
    RandomDataLoader
    from models.eval_tools import Config, compareModels
    import itertools
    from models.params import REG_VALS, LRS, BATCH_SIZE, N_EPOCHS
    from models.Model import SVMModel, LogisticRegressionModel, KMeansModel, \
        KNeighborsModel, SoftmaxANN, DoubleLayerSoftmaxANN, TripleLayerSoftmaxANN
    from sklearn import datasets
    
    data = TextVectorDataLoader(HAND_DATA_PATH, nClasses=5)

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


FEATURE_ACTIONS = ["Call","Raise"] # The actions that are considered for features


def roundVector(roundValues):
    # Reduce number of decisions
    action = roundValues["decision"]
    if "Bet" in action: action = "Raise"
    if "Check" in action: action = "Call"
    if "Fold" in action: action = "Raise" # Dummy; folded rounds won't be considered
    # One hot encode decision
    decisionVector = oneHotEncCat(action, FEATURE_ACTIONS)
    
    # Insert all values to vector
    vector = np.zeros(len(DEEPMIND_GAME_FEATURES)-1)
    for i,feat in enumerate(DEEPMIND_GAME_FEATURES): 
        if i<len(vector):
            vector[i] = featurize(getOrDefault(roundValues, feat, np.nan),
                                  strDic=
                                  {x:i for i, x in 
                                   list(enumerate(DEEPMIND_ACTIONS)) + list(enumerate(GameStages))})
    return np.hstack((vector,decisionVector))


def trainDecisionMaking():
    
    gameLogger = GameLogger()   
    games = gameLogger.mongodb.games.find()
    classes = ["Lost","Won"]
    gameVecs = []
    prevRoundId = (np.nan, np.nan, np.nan)
    gameVec = np.nan
    i = 0
    nSamples = 5
    maxNRounds = 10
    for game in games:
        outcome = game["FinalOutcome"]
        if outcome in ("Neutral",): continue
        rounds = sorted(game["rounds"], key=lambda x:(GameStages.index(x["round_values"]["gameStage"]), x["round_number"]))
        if not rounds: continue
        gameVec = []
        for j,roundData in enumerate(rounds):
            roundVec = roundVector(roundData["round_values"])
            roundVec = np.hstack((roundVec,np.array([classes.index(outcome)])))
            gameVec.append(roundVec)
            if roundData["round_values"]["decision"]!="Fold":
                gameVecs.append(pad(np.array(gameVec),maxNRounds,len(roundVec)))
        i+=1
        if i>200:
            break
    gameVecs = np.array(gameVecs)
#     np.random.shuffle(gameVecs)
    train,test = np.array_split(gameVecs,2)
    print(train.shape)
    print(test.shape)
    X_train,y_train = np.split(train,(-1,),2)
    X_test,y_test = np.split(test,(-1,),2)
    y_train = np.max(np.squeeze(y_train, 2),1)
    y_test = np.max(np.squeeze(y_test, 2),1)
    print(y_train)
    print(X_test)
    print(y_test)
    print(test[0])
    print(test[1])
    print(test[-2])
    print(y_test[0])
    print(y_test[1])
    print(y_test[-2])
    print(np.unique(y_test))
    print(X_train.shape)
    print(y_train.shape)
    print(y_test.shape)
    nSamples = len(gameVecs)
    nDims = len(DEEPMIND_GAME_FEATURES)+1
    
    # truncate and pad input sequences
    print(X_test)
    for m in X_train:
        print(m.shape)
    for m in X_test:
        print(m.shape)
    batch_size = 3
    # create the model
    
    model = Sequential()
    model.add(Dense(batch_size, input_shape=(maxNRounds,nDims)))
    model.add(LSTM(1000))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    print(model.output_shape)
    print(model.input_shape)
    
    
    for i in range(1,10):
        model.fit(X_train, y_train, nb_epoch=10, batch_size=batch_size)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
    
    
def main(args):
    
#     Watchuwanna do?
#     countGamesWithAllCardsShown()
#     collectCardDistributions()
#     getIrcData()
#     dlPokerBotData()
#     mineGameData(debugMode=False)
#     mineHandData()
#     trainEquityPrediction()
    trainDecisionMaking()
    
    pass

if __name__ == '__main__':
    main(sys.argv)
