"""Script for managing downloading, saving and scraping data.
The actual work isn't done here, rather, these are scripts
that manage objects that do the actual work."""
from keras.utils.np_utils import to_categorical
from numpy import random
from os import listdir
from sklearn.cross_validation import train_test_split


from bot.tools.mongo_manager import GameLogger
from models.DataLoader import DataLoader
from util import encodeRec, DummyLogger

from bs4 import BeautifulSoup
import requests
import os
from os.path import join, exists
from copy import copy
import sys
from bot.decisionmaker.montecarlo_python import MonteCarlo
import time
import re
import numpy as np
from pprint import pprint
import pandas as pd
from train.data_parsing import IrcHoldemDataParser
from constants import *
from util import *


def find_files(url):
    soup = BeautifulSoup(requests.get(url).text)

    for a in soup.find_all('a'):
        yield a['href']


def getIrcData():
    if not exists("data"):
        os.mkdir("data")

    if not exists(IRC_DATA_PATH):
        os.mkdir(IRC_DATA_PATH)

    url = 'https://web.archive.org/web/20051029232106/http://games.cs.ualberta.ca:80/poker/IRC/IRCdata/'

    links = []
    for link in find_files(url):
        links.append(link)

    links = links[4:]
    image_url = links[0]
    for elem in links[1:]:
        r = requests.get(
            'https://web.archive.org/web/20051029232106/http://games.cs.ualberta.ca:80/poker/IRC/IRCdata/' + elem)  # create HTTP response object
        with open(join("data", elem), 'wb') as encodeRec:
            encodeRec.write(r.content)
            encodeRec.close()


def getEquity(mycards, tableCards, nActivePlayers):
    tup = ((tuple(sorted(mycards)),),
           tuple(sorted(tableCards)),
           nActivePlayers)

    mc = MonteCarlo()
    timeout = time.time() + 5
    mc.run_montecarlo(DummyLogger(),
                      [mycards],
                      tableCards,
                      player_amount=len(nActivePlayers),
                      ui=None,
                      timeout=timeout,
                      maxRuns=10000,
                      ghost_cards="",
                      opponent_range=0.25)
    return mc.equity


featsToIgnore = [
    "logger",
    "logging_timestamp",
    "ip",
    "version",
    "computername",
    "_id",
    "GameID",
    "uploader",

]


def cleanMdbDic(dic):
    cleaned = {}
    for field, val in dic.items():

        if type(val) == type(b""):
            val = val.decode(sys.stdout.encoding)

        if val == "False": val = 0
        if val == "True": val = 1
        if val == "nan" or not val: val = -1
        if val in GameStages: val = GameStages.index(val)

        if field == "other_players": val = str(val)
        if field == "PlayerCardList": val = " ".join(eval(val))
        if field == "PlayerCardList_and_others": val = " ".join(re.findall("[a-zA-Z\d]+", val))
        cleaned[field] = [val]

    return cleaned


def dlPokerBotData():
    """
    Download and save the data stored on the poker bot's server.
    Each file corresponds to a game, and each column of a game is a roundData.
    I only found games with up to 4 rounds, so maybe each roundData is 
    actually a game stage? 
    """

    if not exists("data"):
        os.mkdir("data")
    if not exists(POKER_BOT_DATA_PATH):
        os.mkdir(POKER_BOT_DATA_PATH)

    gameLogger = GameLogger()
    games = gameLogger.mongodb.games.find()

    prevRoundId = (np.nan, np.nan, np.nan)
    gameDf = np.nan

    for game in games:
        rounds = sorted(game["rounds"],
                        key=lambda x: (GameStages.index(x["round_values"]["gameStage"]), x["round_number"]))

        print([(x["round_values"]["gameStage"],
                x["round_number"]) for x in rounds])

        if not rounds: continue
        gameDf = pd.DataFrame(cleanMdbDic(encodeRec(rounds[0]["round_values"])))
        for roundData in rounds[1:]:
            roundDf = pd.DataFrame(cleanMdbDic(encodeRec(roundData["round_values"])))
            thisRoundId = (roundDf["GameID"][0],
                           roundDf["gameStage"][0],
                           eval(roundDf["round_number"][0]))

            # Append current roundData to game
            gameDf = pd.concat((gameDf, roundDf), axis=0)
            prevRoundId = copy(thisRoundId)
        if type(gameDf) != type(np.isnan):
            blacklist = (
                "rounds", "GameID", "ComputerName", "Template", "FinalDecision", "_id", "ip", "logging_timestamp")
            fileName = game["GameID"] + "-" + "-".join([str(v) for k, v in game.items() if not k in blacklist])

            gameDf.transpose().to_csv(join(POKER_BOT_DATA_PATH, fileName + ".csv"))


def mineGameData(ircDataPath=IRC_DATA_PATH,
                 gameDataPath=IRC_GAME_VECTORS_PATH,
                 debugMode=False):
    """Collect and store vector representation of 
    game rounds from the IRC dataset from the 
    perspective of players that made it to showdown."""

    ircParser = IrcHoldemDataParser(ircDataPath)

    # Counters that get set back to 0 every stage        
    _stageCallCounter = 0  # How many calls so far in this stage
    _stageRaiseCounter = 0  # How many raises so far in this stage
    _stageBetCounter = 0  # How many bets so far in this stage
    if not exists(gameDataPath):
        os.mkdir(gameDataPath)

    fileCounter = 1

    print("Creating file #%d" % fileCounter)
    open(join(gameDataPath, "%d.csv" % fileCounter), "w").close()
    i = 0
    outFile = open(join(gameDataPath, "%d.txt" % fileCounter), "a")
    for game in ircParser.iterGames():
        if debugMode:
            print(game)

        i += 1
        # Choose game and go over all players with cards            
        game = ircParser.nextGame()
        players = [player for player in game.players if player.cards]
        for player in players:
            for vector in game.roundVectors(player, debugMode):
                print(vector)
                quit()
                outFile.write("\t".join([str(x) for x in vector]))

        # Create a new file
        if i % 500 == 0:
            outFile.close()
            fileCounter += 1
            print("Creating file #%d" % fileCounter)
            open(join(gameDataPath, "%d.txt" % fileCounter), "w").close()
            outFile = open(join(gameDataPath, "%d.csv" % fileCounter), "a")

    outFile.close()


class DeepMindPokerDataLoader(DataLoader):
    """Loads an array of 2D matrices, each representing
     a Deep Mind Poker Bot game. Each row of a matrix is one round vector.
     The last column of each round vector is a binary indicating whether the
     round was won or lost, and the last two columns indicate the choice that
     was made in that round."""

    def __init__(self, cap=float('inf')):
        gameLogger = GameLogger()
        games = gameLogger.mongodb.games.find()
        classes = ["Lost", "Won"]
        prevRoundId = (np.nan, np.nan, np.nan)

        gameMats = []
        allVecs = []
        i = 0
        for game in games:
            print(i)
            if i >= cap:
                break

            outcome = game["FinalOutcome"]

            if outcome in ("Neutral",):
                continue

            rounds = sorted(game["rounds"],
                            key=lambda x: (GameStages.index(x["round_values"]["gameStage"]), x["round_number"]))

            if not rounds:
                continue

            gameVec = []
            for j, roundData in enumerate(rounds):
                roundVec = self.makeRoundVector(roundData["round_values"])
                roundVec = np.hstack((roundVec, np.array([classes.index(outcome)])))
                gameVec.append(roundVec)
                allVecs.append(roundVec)
                if roundData["round_values"]["decision"] != "Fold":
                    i += 1
                    gameMats.append(pad(np.array(gameVec), MAX_N_ROUNDS, len(roundVec)))

        gameMats = np.array(gameMats)
        np.random.shuffle(gameMats)
        # No dev set. This is good for Keras-based models which use some of
        # the train set for dev

        train, test = train_test_split(gameMats)

        self.trainX, self.train_y = np.split(train, (-1,), 2)
        self.testX, self.test_y = np.split(test, (-1,), 2)

        self.train_y = np.max(np.squeeze(self.train_y, 2), 1)
        self.test_y = np.max(np.squeeze(self.test_y, 2), 1)

    def makeRoundVector(self, roundValues):
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
                                      {x: i for i, x in
                                       list(enumerate(DEEPMIND_ACTIONS)) + list(enumerate(GameStages))})
        return np.hstack((vector, decisionVector))

    def nClasses(self):
        return len([x for x in np.unique(self.train_y) if x >= 0])


class IrcHandDataLoader(DataLoader):
    """A class for loading the hand rank data created and saved by the
    'buildIrcHandVectors' function. Used to train an LSTM that predicts
    hand strength. The number of hand strenght buckets used is defined in
    the constant N_HAND_CLASSES."""
    def __init__(self):
        # Load all game matrices from all files
        arr = None
        for f in listdir(HAND_DATA_PATH):
            curArr = np.load(join(HAND_DATA_PATH, f))
            if type(arr) == type(None):
                arr = curArr
            else:
                arr = np.vstack((arr, curArr))
        self.trainX, self.train_y, self.devX, self.dev_y, self.testX, self.test_y = trainDevTestPrep(arr)

        # Cluster the labels
        _, y = divXy(arr, 2)
        counts = defaultdict(lambda: 0)
        for x in y:
            counts[x] += 1
        counts = list(counts.items())
        kmeansModel = KMeans(n_clusters=N_HAND_CLASSES, random_state=70)
        kmeansModel.fit([[x] for x in y])

        # Create a function that maps ranks into their classes according to the clusters
        # It maps ranks that are 0 or less into unique classes, rather than their clusters
        # This is so because 0 and less corresponds to folding at different stages of the game
        toClasses = lambda ranks: to_categorical([kmeansModel.predict([[rank]])[0] if rank>0 else N_HAND_CLASSES-rank for rank in ranks])

        ## Show label distribution where the clusters are color coded
        yPred = kmeansModel.predict([[x] for x in y])
        colors = "bgrcmykw"
        clusterToCounts = defaultdict(lambda: [])
        for i in range(len(y)):
            rank = y[i]
            clusterToCounts[yPred[i]].append(rank)
        for cluster, vals in clusterToCounts.items():
            pyplot.scatter([x for x in vals], [random.choice(range(1000)) for _ in vals],
                           color=colors[cluster % len(colors)])
        pyplot.show()

        # Divide labels into classes
        self.train_y = toClasses(self.train_y)
        self.dev_y = toClasses(self.dev_y)
        self.test_y = toClasses(self.test_y)


def buildIrcHandVectors(ircDataPath=IRC_DATA_PATH,
                        outPath=HAND_DATA_PATH,
                        debug=False,
                        maxIterations = 600):
    """Iterate features of IRC games and save them as vectors."""
    parser = IrcHoldemDataParser(ircDataPath)

    fileCounter = 1
    path = join(outPath, "%d.npy" % fileCounter)

    mats = None
    outFile = makeFile(path)
    i = 1
    for game in parser.iterGames():
        if game.nPlayers > MAX_N_PLAYERS:
            continue

        i += 1
        # Stop here
        if i > maxIterations:
            outFile.close()
            os.remove(path)
            break

        # Save data periodically
        if i % 200 == 0:
            print("Saving to file #" + str(fileCounter))
            np.save(outFile, mats)
            mats = None
            fileCounter += 1
            path = join(outPath, "%d.npy" % fileCounter)
            outFile = makeFile(path)

        playerToMat = {}
        for player, gameState in game.roundVectors(debug=debug):

            # Note: equity is modeled as plain rank for now. Rank is how strong
            # the hand is. The greater the number the higher the rank. I'm using
            # the eval7 package.
            label = gameState.get("equity", player.pos)  # We predict equity
            vec = gameState.asVector(playerFeatsBlackList=["equity", "pos"]) # Get vector without equity
            vec.append(label)

            # Build rounds matrix
            if player.pos in playerToMat:
                mat = playerToMat[player.pos]
                mat = np.vstack((mat, vec))
            else:
                mat = np.expand_dims(vec, 0)

            playerToMat[player.pos] = mat

            newMat = np.expand_dims(padSequence(mat, MAX_N_ROUNDS), 0)
            if type(mats) == type(None):
                mats = newMat
            else:
                mats = np.vstack((mats, newMat))
