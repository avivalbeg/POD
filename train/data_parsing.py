"""
A module for parsing the IRC poker data.

For detais about IRC data format see:
https://web.archive.org/web/20100607041834/http://games.cs.ualberta.ca:80/poker/IRC/IRCdata/README.TXT
"""

import os
import glob
import re
from os.path import join, isdir, splitext, exists
from pprint import pprint
from random import choice
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.cross_validation import train_test_split

from constants import *
from util import *
from bot.decisionmaker.montecarlo_python import MonteCarlo
import time
import pandas as pd

import numpy as np

from bot.tools.mongo_manager import GameLogger
from models.DataLoader import DataLoader

from util import *
from constants import *


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


class IrcDataLoader(DataLoader):
    def __init__(self):
        quit()


class GameState:
    """An object to represent a game state. The data is stored in a numpy array
    in which every index corresponds to the feature with the same index.
    The __getitem__ method takes a feature name as a string and returns the value stored
    in the vector's position which corresponds to the feature name's index in the
    feature list. Player feature positions are multiplied by the player's position."""

    def __init__(self):
        self._vector = np.zeros(len(OUR_FEATURES) + MAX_N_PLAYERS * len(PLAYER_FEATURES))

    def asVector(self):
        return self._vector

    def get(self, item, nPlayer=0):
        value = None
        if item in OUR_FEATURES:
            if nPlayer:
                raise ValueError("This is not a player feature, yet player number was specified.")
            value = self._vector[OUR_FEATURES.index(item)]
        elif nPlayer and item in PLAYER_FEATURES:
            if nPlayer > MAX_N_PLAYERS:
                raise ValueError("Player number %d exceeds maximal number of players (%d)" % (nPlayer, MAX_N_PLAYERS))
            value = self._vector[len(OUR_FEATURES) + (nPlayer - 1) * len(PLAYER_FEATURES) + PLAYER_FEATURES.index(item)]
        else:
            raise KeyError(str(item))
        return float(value)

    def set(self, item, newValue, nPlayer=0):
        if item in OUR_FEATURES:
            if nPlayer:
                raise ValueError("This is not a player feature, yet player number was specified.")
            self._vector[OUR_FEATURES.index(item)] = newValue
        elif nPlayer and item in PLAYER_FEATURES:
            if nPlayer > MAX_N_PLAYERS:
                raise ValueError("Player number %d exceeds maximal number of players (%d)" % (nPlayer, MAX_N_PLAYERS))
            self._vector[
                len(OUR_FEATURES) + (nPlayer - 1) * len(PLAYER_FEATURES) + PLAYER_FEATURES.index(item)] = newValue
        else:
            raise KeyError(str(item))

    def inc(self, item, value, nPlayer=0):
        self.set(item, self.get(item, nPlayer) + value, nPlayer)

    _headersAndPlayerNumber = [(feat, 0) for feat in OUR_FEATURES] + joinLists(
        [[(feat, i + 1) for feat in PLAYER_FEATURES] for i in range(MAX_N_PLAYERS)])

    def __str__(self):
        out = ""
        for header, i in self._headersAndPlayerNumber:
            nStr = ""
            if i:
                nStr = "(%d)" % i
            out += tableRowTitle(header + nStr) + " " + str(self.get(header, i)) + "\n"
        return out

    def __repr__(self):
        return str(self)


class Game(object):
    def __init__(self, timeStamp,
                 gameSetId,
                 gameId,
                 nPlayers,
                 flopNPls,
                 flopPot,
                 turnNPls,
                 turnPot,
                 riverNPls,
                 riverPot,
                 showdownNPls,
                 showdownPot,
                 boardCards,
                 players):
        for player in players:
            if player.timeStamp != timeStamp:
                raise IOError("PlayerInGame timestamp is different from game timestamp")

        self.timeStamp, self.gameSetId, self.gameId, self.nPlayers, self.flopNPls, self.flopPot, self.turnNPls, self.turnPot, self.riverNPls, self.riverPot, self.showdownNPls, self.showdownPot, self.boardCards, self.players \
            = timeStamp, gameSetId, gameId, nPlayers, flopNPls, flopPot, turnNPls, turnPot, riverNPls, riverPot, showdownNPls, showdownPot, boardCards, players

        self.preflopPot = 0
        self.preflopNPls = nPlayers

        # Count number of raises for each stage
        counts = {"flop": 0, "turn": 0, "river": 0, "showdown": 0}
        for player in players:
            counts["flop"] += len(re.findall("[Arb]", player.preflopActions))
            counts["turn"] += len(re.findall("[Arb]", player.flopActions))
            counts["river"] += len(re.findall("[Arb]", player.turnActions))
            counts["showdown"] += len(re.findall("[Arb]", player.riverActions))
        self.preflopNraises = counts["flop"] + 2  # for blinds
        self.flopNraises = counts["turn"]
        self.turnNraises = counts["river"]
        self.riverNraises = counts["showdown"]
        self.showdownNraises = 0

        self.data = [timeStamp, gameSetId, gameId, nPlayers,
                     flopNPls, flopPot,
                     turnNPls, turnPot,
                     riverNPls, riverPot,
                     showdownNPls, showdownPot,
                     boardCards, players]
        self.raiseEsts = {"preflop": [], "flop": [], "turn": [], "river": []}

    def getWinner(self):
        winnings = [player.winnings for player in self.players]
        if (not 0 in winnings) or not any(winnings):
            return np.nan  # Draw
        return self.players[np.argmax(winnings)]

    def getFinalPot(self):
        return sum([player.winnings for player in self.players])

    def initGameState(self):
        """Initiate an empty game state."""
        gs = GameState()
        gs.set('nPlayers', self.nPlayers)
        gs.set('nActivePlayers', self.nPlayers)
        for i, player in enumerate(self.players):
            i += 1  # Because player index starts from 1
            gs.set('pos', player.pos, nPlayer=i)
            gs.set('funds', player.bankroll, nPlayer=i)
        return gs

    def roundVectors(self, debug=False):
        gameState = self.initGameState()
        cardsOnTable = []

        if debug:
            print(self)

        # Go through all stages
        while gameState.get("gameStage") != len(GameStages) - 1:

            # Update each player's equity (right now it's rank)
            for player in self.players:
                gameState.set("equity", getRank(player.cards+cardsOnTable), player.pos)

            someonePlayed = True  # Tells us if there are still players who didn't quit/fold
            gameState.set("stagePotValue", 0)  # re-init every stage
            while someonePlayed:
                someonePlayed = False
                for player in self.playRound(gameState, debug):
                    if player:
                        yield player, gameState
                        someonePlayed = True
                gameState.inc("roundNumber", 1)

            # Re-init stage variables
            gameState.set("roundNumber", 0)
            gameState.set("lastBetValue", 0)
            gameState.inc("gameStage", 1)

            # Expose cards
            if gameState.get("gameStage") == GameStages.index(Flop):
                cardsOnTable = list(map(ircCardToBotCard, self.boardCards[:3]))
            if gameState.get("gameStage") == GameStages.index(Turn):
                cardsOnTable = list(map(ircCardToBotCard, self.boardCards[:4]))
            if gameState.get("gameStage") == GameStages.index(River):
                cardsOnTable = list(map(ircCardToBotCard, self.boardCards))

            # Update absolute equity (right now, it's not really equity, just hand rank)
            gameState.set("absEquity", getRank(cardsOnTable))

    def playRound(self, gameState, debug=False):
        """Play one round of this game, starting from the given game state, and going through
        all the players in the table. For each player, if that player is still in the game,
        yield True, otherwise, yield false."""

        gameStageStr = GameStages[int(gameState.get("gameStage"))]
        nextStage = GameStages[GameStages.index(gameStageStr) + 1]
        gameStageStr = gameStageStr.lower()

        # global bet: divide money from this stage by number of raising/calling actions in this stage
        # Each bet/call action performed in this stage is then taken to be this amount, because the IRC
        # records don't contain documentation of exact betting amounts.
        betAvg = divOr0(
            (getattr(self, nextStage.lower() + "Pot") - getattr(self, gameStageStr + "Pot")),
            sum(list(map(
                lambda plr: len(re.findall("[BbrcA]", getattr(plr, gameStageStr + "Actions"))),
                self.players
            )))
        )
        betAvg = int(betAvg)  # Might want to remove this sometimes

        for curPlayer in self.players:
            player = curPlayer

            pos = curPlayer.pos
            # Find action
            actions = getattr(curPlayer, gameStageStr + "Actions")
            action = getOrDefault(actions,
                                  int(gameState.get("roundNumber")),
                                  NA)

            # Update first and second raiser and caller
            if action in (RAISE, BET, ALL_IN):
                gameState.set("lastRaiser", pos)
                if not gameState.get("firstRaiser"):
                    gameState.set("firstRaiser", pos)
                elif (gameState.get("firstRaiser")
                      and not gameState.get("secondRaiser")):
                    gameState.set("secondRaiser", pos)
            if action == CALL:
                gameState.set("lastCaller", pos)
                if not gameState.get("firstCaller"):
                    gameState.set("firstCaller", pos)
                elif (gameState.get("firstCaller")
                      and not gameState.get("secondCaller")):
                    gameState.set("secondCaller", pos)

            # If even one player did something,
            # this game stage continues
            if action not in (NA, QUIT, KICKED):
                if debug:
                    print(curPlayer.name, action, int(betAvg), int(gameState.get("totalPotValue")),
                          gameState.get("gameStage"),
                          gameState.get("roundNumber"))

            # Update table based on action
            if action == CALL:
                gameState.inc("funds", -betAvg, pos)
                gameState.inc("nCalls", 1, pos)
                gameState.inc("callSum", betAvg, pos)

                gameState.inc("totalPotValue", betAvg)
                gameState.inc("stagePotValue", betAvg)

            # I still haven't handled all in
            elif action in (BLIND, BET, RAISE, ALL_IN):
                gameState.set("lastBet", betAvg, pos)
                gameState.inc("funds", -betAvg, pos)
                gameState.inc("nRaises", 1, pos)
                gameState.inc("raiseSum", betAvg, pos)

                gameState.set("lastBetValue", betAvg)
                gameState.inc("totalPotValue", betAvg)
                gameState.inc("stagePotValue", betAvg)

            elif action == CHECK:
                gameState.inc("nChecks", 1, pos)

            elif action in (FOLD, KICKED, QUIT):
                gameState.inc("nActivePlayers", -1)

            elif action == NA:
                player = None

            else:
                raise ValueError("Unexpected action " + action)

            if debug and action in (RAISE, BET, ALL_IN, CALL, BLIND):
                print()
                print(gameState)
            yield player

    def featurizeRound(self):
        pass

    def getEquity(self, table):

        tup = ((tuple(sorted(table.mycards)),),
               tuple(sorted(table.cardsOnTable)),
               len(table.other_active_players))

        if tup in relEquityCache.keys():
            return relEquityCache[tup]
        mc = MonteCarlo()
        timeout = time.time() + 5
        mc.run_montecarlo(DummyLogger(),
                          [table.mycards],
                          table.cardsOnTable,
                          player_amount=len(table.other_active_players),
                          ui=None,
                          timeout=timeout,
                          maxRuns=10000,
                          ghost_cards="",
                          opponent_range=0.25)

        relEquityCache[tup] = mc.equity
        return mc.equity

    def getAbsEquity(self, cards, nActivePlayers):
        tup = (tuple(sorted(cards)), nActivePlayers)
        if tup in globalEquityCache.keys():
            return globalEquityCache[tup]
        mc = MonteCarlo()
        timeout = time.time() + 5
        mc.run_montecarlo(DummyLogger(),
                          [cards[:2]],
                          cards[2:],
                          player_amount=len(table.other_active_players),
                          ui=None,
                          maxRuns=10000,
                          timeout=timeout,
                          ghost_cards="",
                          opponent_range=0.25)

        # Cache and return
        globalEquityCache[tup] = mc.equity
        return mc.equity

    def getAbsEquity(self, cardsOnTable, nActivePlayers):
        tup = (tuple(sorted(cardsOnTable)), nActivePlayers)
        if tup in globalEquityCache.keys():
            return globalEquityCache[tup]
        mc = MonteCarlo()
        timeout = time.time() + 5
        mc.run_montecarlo(DummyLogger(),
                          [cardsOnTable[:2]],
                          cardsOnTable[2:],
                          player_amount=nActivePlayers,
                          ui=None,
                          maxRuns=10000,
                          timeout=timeout,
                          ghost_cards="",
                          opponent_range=0.25)

        # Cache and return
        globalEquityCache[tup] = mc.equity
        return mc.equity

    def newTable(self, player, h):

        # Init history (we don't really need it)
        h.round_number = 0
        preflop_url = "bot/decisionmaker/preflop.xlsx"  # This path is relative to the POD directory; if you're running it from elsewhere you might have to adjust it
        h.preflop_sheet_name = "preflop.xlsx"
        h.preflop_sheet = pd.read_excel(preflop_url, sheetname=None)
        h.myLastBet = 0

        # Init table 
        table = DummyTable()
        table.myPos = player.pos
        table.gameStage = PreFlop
        table.other_active_players = set([plyr.name for plyr in self.players if plyr != player])

        table.equity = 0
        table.global_equity = 0
        table.nMyRaises = 0
        table.nMyCalls = 0
        table.nMyChecks = 0
        table.myRaiseSum = 0
        table.myCallSum = 0

        table.first_raiser = np.nan
        table.second_raiser = np.nan
        table.first_caller = np.nan
        table.second_caller = np.nan
        table.first_raiser_utg = np.nan
        table.first_caller_utg = np.nan
        table.second_raiser_utg = np.nan

        table.mycards = [ircCardToBotCard(card) for card in player.cards]
        table.other_players = [othrPlyr for othrPlyr in self.players if othrPlyr.name != player.name]
        table.round_pot_value = 0
        table.currentCallValue = 0
        table.currentBetValue = 0
        table.relative_equity = 0
        table.global_equity = 0
        table.cardsOnTable = []
        table.totalPotValue = 0
        table.max_X = 0.86
        table.myFunds = player.bankroll

        return table

    def __str__(self):
        out = "IRC Game Record\n----------------------------------------\n"

        out += tableRowTitle("Timestamp ") + str(self.timeStamp) + "\n"
        out += tableRowTitle("Players (" + str(self.nPlayers) + ")") + " ".join(
            [player.name for player in self.players]) + "\n"
        out += tableRowTitle("Game set # ") + str(self.gameSetId) + "\n"
        out += tableRowTitle("Game # ") + str(self.gameId) + "\n"
        out += tableRowTitle("Board cards ") + " ".join(self.boardCards) + "\n"

        out += "\n\nPhase data  (#plrs/potsize):\n\n"
        out += tableRowTitle("Flop ") + str(self.flopNPls) + "/" + str(self.flopPot) + "\n"
        out += tableRowTitle("Turn ") + str(self.turnNPls) + "/" + str(self.turnPot) + "\n"
        out += tableRowTitle("River ") + str(self.riverNPls) + "/" + str(self.riverPot) + "\n"
        out += tableRowTitle("Showdown ") + str(self.showdownNPls) + "/" + str(self.showdownPot) + "\n"

        out += "\n\nPlayer data:\n\n"

        headers = tableRowFromList("name pos preflop flop turn river bankroll action winnings cards".split(" "))

        out += headers + "\n"
        out += "-" * len(headers) + "\n"
        for player in self.players:
            out += tableRowFromList([str(x) for x in player.data[2:-1]] + [" ".join(player.cards)]) + "\n"

        return out

    def __repr__(self):
        return str(self)


class PlayerInGame(object):
    """
    Simple wrapper class to represent data for one player in one game.
    """

    def __init__(self, timeStamp,
                 nPlayers,
                 name,
                 pos,
                 preflopActions,
                 flopActions,
                 turnActions,
                 riverActions,
                 bankroll,
                 action,
                 winnings,
                 cards):
        self.timeStamp, self.nPlayers, self.name, self.pos, self.preflopActions, self.flopActions, self.turnActions, self.riverActions, self.bankroll, self.action, self.winnings, self.cards \
            = timeStamp, nPlayers, name, pos, preflopActions, flopActions, turnActions, riverActions, bankroll, action, winnings, cards
        self.data = [timeStamp, nPlayers, name, pos, preflopActions, flopActions, turnActions, riverActions, bankroll,
                     action, winnings, cards]

    def __str__(self):
        return " ".join([str(x) for x in self.data])

    def __repr__(self):
        return str(self)

    # Adjustments to fit pokerbot

    def __getitem__(self, item):
        if item == "pot":
            return self.bankroll

    def __eq__(self, o):
        return self.name == o.name

    def __ne__(self, o):
        return not self == o


class Player(object):
    """Represents overall data of a player, read from a player file."""

    def __init__(self, name):
        self._name = name
        self._avgWinnings = 0
        self._avgRaising = 0
        self._nBluffs = 0
        self._nGames = 0

    def update(self, gameData):
        """Update a player's stats based on one game's data."""
        pass


class DataParser:
    pass


class IrcDataParser(DataParser):
    pass


class IrcHoldemDataParser(IrcDataParser):
    # API

    def nextGame(self):
        return next(self._gameIt)

    def nextPlayer(self):
        return next(self._playerIt)

    def getRandomGame(self):
        """Returns a randomly sampled game."""

        headPath = choice(self._headPaths)
        folder = choice(os.listdir(headPath))
        path = (join(headPath, folder))

        try:
            hrosterFile = open(join(path, "hroster"))  # Game metadata (which players)
            hdbFile = open(join(path, "hdb"))  # Game data
            metadataLines, dataLines = hrosterFile.readlines(), hdbFile.readlines()
            hrosterFile.close()
            hdbFile.close()
        except FileNotFoundError:
            return self.getRandomGame()
        # Get random game from folder
        i = float("inf")
        while i >= len(metadataLines) or i >= len(dataLines):
            i = choice(range(len(metadataLines)))
        # Collect and split data lines:
        metadataLine, dataLine = metadataLines[i], dataLines[i]

        game = self._gameFromLines(metadataLine, dataLine, path)

        return game or self.getRandomGame()

    # Private methods

    def __init__(self, dataDirPath=DATA_DIR_PATH, playerSkipProb=0):
        self.nGames = 0
        self.badGames = 0

        # Collect all topmost holdem path
        self._headPaths = [join(dataDirPath, path) for path in os.listdir(dataDirPath) if
                           re.match(HOLDEM_PATHS_REGEX, path)
                           and isdir(join(dataDirPath, path))]

        self._seenPlayers = set()

        self._playerIt = self._makePlayerIterator(playerSkipProb)
        self._gameIt = self._makeGamesDataIterator()

    def _makePlayerIterator(self, skipProb):
        """Returns a generator that iterates over all players
        in the dataset. Skipes a plyer with porbability skipProb.
        skipProb effectively decides what is the fraction of players
        the fraction of players we get at the end."""

        count = 0
        # Go through all game files
        for headPath in self._headPaths:
            for folder in os.listdir(headPath):
                path = (join(headPath, folder))
                if not exists(join(path, "pdb")):
                    continue
                for playerFileName in os.listdir(join(path, "pdb")):

                    # Decide if should skip:
                    _, name = splitext(playerFileName)

                    count += 1

                    self._seenPlayers.add(name)
                    if name in self._seenPlayers \
                            or not (np.random.binomial(1, skipProb)):
                        playerPath = join(path, "pdb", playerFileName)
                        yield self._playerFromFile(playerPath)

    def iterGames(self):
        """Returns a generator object that yields all games
        one by one."""

        # Go through all game files
        for headPath in self._headPaths:
            for folder in os.listdir(headPath):
                path = (join(headPath, folder))

                hrosterFile = open(join(path, "hroster"))  # Game metadata (which players)
                hdbFile = open(join(path, "hdb"))  # Game data
                metadataLines, dataLines = hrosterFile.readlines(), hdbFile.readlines()
                hrosterFile.close()
                hdbFile.close()

                # For each metadata and data line:
                for i in range(len(metadataLines)):
                    # Collect and split data lines:
                    if i < len(metadataLines) and i < len(dataLines):
                        metadataLine, dataLine = metadataLines[i], dataLines[i]

                    game = self._gameFromLines(metadataLine, dataLine, path)
                    if game:
                        yield game

    def _playerFromFile(self, path):
        player = Player(splitext(path)[-1][1:])
        with open(path) as encodeRec:
            for line in encodeRec:
                try:
                    lineElems = splitIrcLine(line)

                    name, timeStamp, nPlayers, pos, preflopActions, flopActions, turnActions, riverActions, bankroll, action, winnings = lineElems[
                                                                                                                                         :11]
                    cards = lineElems[11:]

                    gameData = PlayerInGame(timeStamp, nPlayers, name, eval(pos), preflopActions, flopActions,
                                            turnActions, riverActions, eval(bankroll), eval(action), eval(winnings),
                                            cards)

                    player.update(gameData)
                except:
                    # If this line is bad, just ignore it
                    pass
        return player

    def _gameFromLines(self, metadataLine, dataLine, rootPath):
        """
        Create a game object from all the lines that are relevant
        for that game.
        """
        #         try:
        gameMetaData = splitIrcLine(metadataLine)
        gameData = splitIrcLine(dataLine)

        # Get data from lines:
        timeStamp1, nPlayers, playerNames = gameMetaData[0], eval(gameMetaData[1]), gameMetaData[2:]
        timeStamp2, gameSetId, gameId, nPlayers, flop, turn, river, showdown = \
            gameData[:8]
        boardCards = gameData[8:]
        flopNPls, flopPot = flop.split("/")
        turnNPls, turnPot = turn.split("/")
        riverNPls, riverPot = river.split("/")
        showdownNPls, showdownPot = showdown.split("/")

        if timeStamp1 != timeStamp2:
            return None
        timeStamp = timeStamp1

        # Get data from player files:
        # First, get the files that correspond to each player:

        players = []  # PlayerInGame objects

        for playerName in playerNames:
            playerPath = join(rootPath, "pdb", "pdb." + playerName)
            # Search for game line
            try:
                with open(playerPath) as playerFile:
                    for line in playerFile:
                        lineElems = splitIrcLine(line)
                        if lineElems[1] == timeStamp:
                            break
            except OSError:
                continue
            # parse player data
            name, timeStamp, nPlayers, pos, preflopActions, flopActions, turnActions, riverActions, bankroll, action, winnings = lineElems[
                                                                                                                                 :11]
            cards = lineElems[11:]
            plyr = PlayerInGame(timeStamp, nPlayers, name, eval(pos), preflopActions, flopActions, turnActions,
                                riverActions, eval(bankroll), eval(action), eval(winnings), cards)
            players.append(plyr)
        players.sort(key=lambda plyr: plyr.pos)
        self.nGames += 1

        return Game(timeStamp,
                    eval(gameSetId),
                    eval(gameId),
                    eval(nPlayers),
                    eval(flopNPls),
                    eval(flopPot),
                    eval(turnNPls),
                    eval(turnPot),
                    eval(riverNPls),
                    eval(riverPot),
                    eval(showdownNPls),
                    eval(showdownPot),
                    boardCards,
                    players)

    #         except:
    #             self.badGames+=1
    #             # returns None





def test():
    parser = IrcHoldemDataParser()
    for _ in range(5):
        (parser.getRandomGame())


if __name__ == '__main__':
    test()
