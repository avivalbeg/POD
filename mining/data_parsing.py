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
from constants import *
from util import *
from bot.decisionmaker.montecarlo_python import MonteCarlo
import time
import pandas as pd



class GameTracker(object):
    def __init__(self, game):
        self.activePlayers = set([player.name for player in game.players]) 
        self.stage = GameStages[0]
        self.roundCounter = 0
        self.nCalls = 0
        self.nRaises = 0
        
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
        
        self.timeStamp, self.gameSetId, self.gameId, self.nPlayers, self.flopNPls, self.flopPot, self.turnNPls, self.turnPot, self.riverNPls, self.riverPot, self.showdownNPls, self.showdownPot, self.boardCards, self.players\
 = timeStamp, gameSetId, gameId, nPlayers, flopNPls, flopPot, turnNPls, turnPot, riverNPls, riverPot, showdownNPls, showdownPot, boardCards, players
        
        self.preflopPot = 0
        self.preflopNPls = nPlayers
        
        # Count number of raises for each stage
        counts = {"flop":0, "turn":0, "river":0, "showdown":0}
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
        self.raiseEsts = {"preflop":[], "flop":[], "turn":[], "river":[]}
    
    def getWinner(self):
        winnings = [player.winnings for player in self.players]
        if (not 0 in winnings) or not any(winnings):
            return np.nan # Draw 
        return self.players[np.argmax(winnings)]
        
    def getFinalPot(self):
        return sum([player.winnings for player in self.players])
    
    def roundVectors(self, player, debugMode=False):
        """Returns an iterator over feature vectors of the game's rounds, 
        from the perspective of given player."""
        hist = DummyHistory()
        table = self.newTable(player, hist)
        
        # Go through the game's rounds and save all data
        while table.gameStage != Showdown:
            # Play stage
            if table.gameStage != PreFlop:
                table.max_X = 1
            
            someonePlayed = True  # Tells us if there are still players who didn't quit/fold
            table.round_pot_value = 0 # re-init every stage           
            while someonePlayed:
                raundVector,someonePlayed = self.playRound(player, table, hist, debugMode)
                hist.round_number += 1
                yield raundVector

            # Re-init stage variables
            hist.round_number = 0 
            table.currentBetValue = 0
            table.currentCallValue = 0
            
            table.gameStage = getOrDefault(GameStages,
                                           GameStages.index(table.gameStage) + 1,
                                           None)
            
            # Expose cards
            if table.gameStage == Flop:
                table.cardsOnTable = list(map(ircCardToBotCard, self.boardCards[:3]))
            if table.gameStage == Turn:
                table.cardsOnTable = list(map(ircCardToBotCard, self.boardCards[:4]))
            if table.gameStage == River:
                table.cardsOnTable = list(map(ircCardToBotCard, self.boardCards))
                
            # Update equity
            table.global_equity = self.getAbsEquity(table)
            table.equity = self.getEquity(table)
            
    def playRound(self, player, table, hist, debugMode=False):
        """Play one round of this game, starting from the given table and history states.
        Returns a tuple (np.array,bool). The first element is the feature 
        vector of the round if it was played by player, and None otherwise.
        The second element indicates whether or not someone played during this round."""
        
        
        nextStage = GameStages[GameStages.index(table.gameStage) + 1]
        prevStage = GameStages[GameStages.index(table.gameStage) + 1]
        
        # global bet: divide money from this stage by number of players that put in money this turn
        betAvgGlobal = divOr0((getattr(self, nextStage.lower() + "Pot") - getattr(self, table.gameStage.lower() + "Pot"))\
                    , len(list(filter(lambda plr: re.findall("[BrbcA]",
                                     getattr(plr, table.gameStage.lower() + "Actions"))
                                      , self.players))))
        
        features,someonePlayed = None,False

        for curPlayer in self.players:
            
            # Find action
            actions = getattr(curPlayer, table.gameStage.lower() + "Actions")
            action = getOrDefault(actions,
                                  hist.round_number,
                                  NA)
            
            # Update first and second raiser and caller
            if action == RAISE:
                if np.isnan(table.first_raiser):
                    table.first_raiser = curPlayer.pos
                elif (not np.isnan(table.first_raiser)) \
                    and np.isnan(table.second_raiser):
                        table.second_raiser = curPlayer.pos
            if action == CALL:
                if np.isnan(table.first_caller):
                    table.first_caller = curPlayer.pos
                elif (not np.isnan(table.first_caller)) \
                    and np.isnan(table.second_caller):
                        table.second_caller = curPlayer.pos
                
            # Weigh bet by number of money-placing actions this stage    
            betAvg = divOr0(betAvgGlobal , len(re.findall("[AbrcB]", actions))) 
            
            # If even one player did something, 
            # this game stage continues
            if not action in (NA, QUIT, KICKED):
                someonePlayed = True
                if debugMode:
                    print(curPlayer.name, action, int(betAvg), int(table.totalPotValue), table.gameStage, hist.round_number)
            
            # Update table based on action
            if action == CALL:
                if curPlayer == player:
                    table.myFunds -= betAvg
                    table.nMyCalls += 1
                    table.myCallSum += betAvg
                table.totalPotValue += betAvg
                self._stageCallCounter += 1
                table.round_pot_value += betAvg
            
            # I still haven't handled all in
            elif action in (BLIND, BET, RAISE, ALL_IN):
                if curPlayer == player:
                    hist.myLastBet = betAvg
                    table.myFunds -= betAvg
                    table.nMyRaises += 1
                    table.myRaiseSum += betAvg

                table.currentBetValue = betAvg
                table.currentCallValue = betAvg
                table.totalPotValue += betAvg
                table.round_pot_value += betAvg
                
            elif action == CHECK:
                if curPlayer == player:
                    table.nMyChecks += 1
            
            elif action == FOLD:
                table.other_active_players = table.other_active_players.difference({curPlayer.name})
            elif action == KICKED:
                table.other_active_players = table.other_active_players.difference({curPlayer.name})
            elif action == QUIT:
                table.other_active_players = table.other_active_players.difference({curPlayer.name})
                
            elif action == NA:
                pass
            
            else:
                raise ValueError("Unexpected action " + action)
            
            
            # Note: These features represent the game state
            # AT THE END OF THIS ROUND. So the first feature
            # dictionary that is returned is after the 0th (or first) 
            # round of the preflop
            if curPlayer == player and action != NA:
                features = self.featurizeTable(table, hist, player)
                
                if debugMode:
                    pprint({FEATURES[i]:features[i] for i in range(len(FEATURES))})
            
        return (features, someonePlayed)
    
    
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

    def getAbsEquity(self, table):
        tup = (tuple(sorted(table.cardsOnTable)), len(table.other_active_players))
        if tup in globalEquityCache.keys():
            return globalEquityCache[tup]
        mc = MonteCarlo()
        timeout = time.time() + 5
        mc.run_montecarlo(DummyLogger(),
                          [table.cardsOnTable[:2]],
                          table.cardsOnTable[2:],
                          player_amount=len(table.other_active_players),
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
        out += tableRowTitle("Players (" + str(self.nPlayers) + ")") + " ".join([player.name for player in self.players]) + "\n"
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
        self.data = [timeStamp, nPlayers, name, pos, preflopActions, flopActions, turnActions, riverActions, bankroll, action, winnings, cards]
    
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
                    if name in self._seenPlayers\
                    or not(np.random.binomial(1, skipProb)):
                        
                        playerPath = join(path, "pdb", playerFileName) 
                        yield self._playerFromFile(playerPath)
                                
                    
    def _makeGamesDataIterator(self):
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
    
                    name, timeStamp, nPlayers, pos, preflopActions, flopActions, turnActions, riverActions, bankroll, action, winnings = lineElems[:11]
                    cards = lineElems[11:]
                
                    gameData = PlayerInGame(timeStamp, nPlayers, name, eval(pos), preflopActions, flopActions, turnActions, riverActions, eval(bankroll), eval(action), eval(winnings), cards)
                    
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
                name, timeStamp, nPlayers, pos, preflopActions, flopActions, turnActions, riverActions, bankroll, action, winnings = lineElems[:11]
                cards = lineElems[11:]
                plyr = PlayerInGame(timeStamp, nPlayers, name, eval(pos), preflopActions, flopActions, turnActions, riverActions, eval(bankroll), eval(action), eval(winnings), cards)
                players.append(plyr)
            players.sort(key=lambda plyr:plyr.pos)
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
        
    
    def iterGames(self):
            return self._gameIt
def test():
    parser = IrcHoldemDataParser()
    for _ in range(5):
        (parser.getRandomGame())

if __name__ == '__main__':
    test()
