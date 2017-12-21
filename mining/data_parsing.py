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
import numpy
from constants import *
from util import *


HOLDEM_PATHS_REGEX = r"holdem*"


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
                    or not(numpy.random.binomial(1, skipProb)):
                        
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
        with open(path) as f:
            for line in f:
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
